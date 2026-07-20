# jude 分布式查询 & UDF 基建实现原理

> 面向「一段时间没跟进、想快速搞懂 jude 分布式与 UDF 是怎么跑起来的」的读者。
> 阅读顺序：先读 §1 心智模型，再按需深入。所有 `文件:行号` 均可跳转核对。
> 最后更新：2026-07-19。

---

## 1. 一句话心智模型

**jude = 不 fork 的 DuckDB（干算力） + Rust 大脑（做所有调度决策） + Ray 手脚（只做 RPC）。**

- **DuckDB**：每个 worker 里跑一个**原生 DuckDB**，负责真正的 SQL 执行。jude 不改 DuckDB 源码（对比 Vane fork 了 DuckDB C++）。
- **Rust 大脑**：分区数、分区切法、worker 分配、并发窗口、shuffle 分桶、资源准入、stage 拆分——**所有"怎么调度"的决策都在 Rust**（`src/dist/`），GIL-free。
- **Ray 手脚**：Python 侧 (`python/jude/runners/`) 只把 Rust 的决策**翻译成 Ray RPC**，不含任何调度算法。这就是「尽可能少 py」原则。

```
   用户 API (Relation.collect / map_batches / aggregate ...)
        │
        ▼
   Rust 决策层  src/dist/  ── WorkerManager / StagePlanner / physical / ResourceManager / ClusterScheduler
        │  (返回：分区计划、worker 映射、并发窗口、shuffle 分桶、stage DAG)
        ▼
   Python 薄执行层  python/jude/runners/  ── RayRunner + _ray_shim
        │  (把决策变成 ray.remote 调用；ObjectRef 路由；ray.wait 循环)
        ▼
   Ray Actor (_JudeWorker)  ×N  ── 每个内含一个原生 DuckDB 连接
        │
        ▼
   Arrow 数据在 object store 中零拷贝流动
```

---

## 2. 两条独立的「分布式」轴，别混淆

jude 里有**两个**容易混为一谈的东西：

| | **分布式查询**（关系代数） | **分布式 UDF**（map_batches） |
|---|---|---|
| 干什么 | 把 SQL/关系算子（agg/join/sort/distinct/setop）拆到多机跑 | 把一个 Python 函数并行作用到每个数据分片 |
| 入口 | `Relation.collect()` / `execute_dag()` | `Relation.map_batches(fn, execution_backend=...)` |
| 核心难点 | shuffle（数据要按 key 重分布） | 绕开 GIL（Python 函数并行） |
| 执行体 | worker 里的 DuckDB 跑 SQL | worker 里的 Python 解释器跑 pickle 来的函数 |
| 代码 | `src/dist/` + `runners/ray.py` 的 distributed_* / execute_dag | `src/udf/` + `execution/` + `runners/ray.py` 的 map_relation |

下面 §3 讲第一条轴，§4 讲第二条。

---

## 3. 分布式查询基建

### 3.1 一切从 LogicalPlan 说起

用户调 `.filter().aggregate().join()` 时，jude **不立刻执行**，而是在 Rust 里搭一棵 `LogicalPlan` 树（`src/plan.rs:22`）：

```
Order
 └─ Join
     ├─ Aggregate(group=[region], aggs=[SUM(amt)])
     │   └─ Table("sales")
     └─ Table("names")
```

每个节点知道怎么把自己 lower 成 SQL（`LogicalPlan::to_sql`, `src/plan.rs:137`）。单机执行就是「整棵树 → 一条大 SQL → 丢给 DuckDB」。分布式则要**在 shuffle 边界把树切开**。

### 3.2 什么是 shuffle 边界

有些算子能「各分片各算，结果拼起来就对」——scan/filter/project/map，叫 **partition-wise（分区内）**。
有些算子需要「跨所有分片的全局视角」——aggregate（同 key 要聚一起）、join（同 key 要碰面）、distinct、order、setop——这些叫 **shuffle 边界**（`src/dist/physical.rs:23` `is_shuffle`）。

分布式执行的本质：**在 shuffle 边界把数据按 key 重新分桶（洗牌），让需要碰面的数据落到同一个 worker。**

### 3.3 Rust 大脑：WorkerManager（怎么分区、怎么分配）

`src/dist/worker_manager.rs` 是调度大脑，`RayRunner` 每个决策都问它：

- `target_partitions(nbytes, nrows)` — 该切几个分区？（按数据大小 / worker 数 / `open_cost_bytes`）
- `partition_plan(nrows, nbytes, hint)` — 返回每个分区的 `(起始行, 行数)` 切片计划（Python 只负责按计划 `table.slice`）。
- `worker_for(i)` — 第 i 个任务派给哪个 worker（round-robin）。
- `dispatch_window(n)` — 并发窗口多大（背压）。0=无限，否则 = `max_task_backlog`。
- `shuffle_bucket_count(n)` / `shuffle_bucket_workers(n)` — shuffle 分几个桶、每个桶哪个 worker 当 reducer。

关键：**这些方法都是纯决策，不碰 Ray**。所以能在 Rust 单测里验证（`cargo test --lib dist`），也不受 GIL 影响。

### 3.4 Python 手脚：RayRunner + _ray_shim

`python/jude/runners/ray.py` 的 `RayRunner`：
- `_partition_tables(rel)` — 调 `mgr.partition_plan(...)` 拿切片计划，执行 `table.slice(...).combine_chunks()`。
- `_dispatch_bounded(fns)` — 调 `mgr.dispatch_window(...)` 拿窗口，交给 shim 跑 `ray.wait` 循环（`_ray_shim.run_bounded`, `:258`）。

`python/jude/runners/_ray_shim.py` 的 `_JudeWorker`（`@ray.remote` actor）：
- 每个 actor 内含**一个原生 DuckDB 连接**（`self._conn = jude.connect()`）。
- `run_sql_on_table(table, sql)` — 把 Arrow 分片注册成表 `part`，跑 SQL，返回 Arrow。
- `bucketize(table, key_expr, b)` — 把一个分片按 `hash(key) % b` 切成 b 个桶（shuffle 的 producer 侧）。
- `join_bucket_group` / `sql_on_refs` / `setop_on_refs` / `distinct_bucket` — shuffle 的 reducer 侧：把某个桶的所有分片（从各 producer 来的 ObjectRef）拉过来，拼一起，本地算。

### 3.5 单个 shuffle 怎么跑（以 join 为例）

`distributed_join_streaming`（`ray.py:560`）：
```
左表分片   右表分片
  │每片          │每片
  ▼ bucketize    ▼ bucketize      ← producer：各分片在自己 worker 上按 key 分成 b 个桶
 [L0..Lb][..]   [R0..Rb][..]        (num_returns=b，每个桶是独立 ObjectRef)
        └────┬────┘
             ▼ 按桶路由              ← 桶 k 的所有左右分片 ObjectRef 交给 reducer worker[k]
    join_bucket_group(桶k左片们, 桶k右片们)   ← reducer：拉取+拼接+本地 join
             ▼
        结果分片拼接
```
**要点**：shuffle 数据从 producer worker 直接经 object store 流到 reducer worker，**不经过 driver**（driver 只路由 ObjectRef）。这就是「pipelined / streaming shuffle」——producer 和 consumer 重叠，driver 不成为瓶颈。这是 jude 对标 Vane「Flight exchange」的答案，但在编排层用 Ray object store 实现。

**聚合**用两阶段（`_agg.py` `build_two_phase`）：各分片先算 partial（`SUM→SUM`, `COUNT→SUM(COUNT)`, `AVG→SUM/COUNT`），reducer 再 merge。exact 且省内存。

### 3.6 通用流式 stage-DAG 执行器（最近补的关键件）

**之前的问题**：上面每个 `distributed_*` 只会分布式**一个** shuffle。碰到**嵌套** shuffle（如 `aggregate → join → order`），`collect()` 只把最外层分布式，里层直接 `relation.to_arrow()` **退化成单机**。

**现在的解法**（`src/dist/physical.rs` + `ray.py:execute_dag`）：

1. **Rust 分解**（`physical::peel`）：把计划顶部的 partition-wise 区域「剥」下来（渲染成 over `part` 的 SQL），停在最近的 shuffle 边界，返回：顶部本地 SQL + 该边界类型/keys + 边界下面的**子计划们**。
2. **暴露给 Python**：`Relation.dist_step()`（`src/relation.rs`）返回一个 dict：`{local_sql, pushable, boundary, keys, join_keys, how, children:[子Relation], ...}`。
3. **Python 递归执行**（`_dag_partitions`, `ray.py:648`）：
   - 对每个 child **递归**跑 `_dag_partitions` → 拿到 child 的**输出分片 ObjectRef 列表**；
   - 用这些 ref 作为**本 shuffle 的输入**（join 就 bucketize+join_bucket_group，agg 就 partial+final，order 就本地排序+归并…）；
   - 最后把顶部 `local_sql` 作用上去（pushable 就每分片各作用，否则汇总到一个 reducer 作用一次）。

**关键不变量**：stage 之间用 **ObjectRef 列表**（分片句柄）当交换货币，中间结果**永不汇总回 driver**。所以 `aggregate→join→order` 全程分布式、流式。**没有** Vane 的容错框架（无落盘 spooling、无 attempt 重试）——这是刻意的，用户明确要 streaming 而非 FTE。

对应关系：`stage.rs` 出「DAG 形状」，`physical.rs` 出「单步分解」，`execute_dag` 是「流式运行时」。

### 3.7 一个隐藏大坑：decimal128 对齐

`SUM(int)` 在 DuckDB 里产出 `decimal128(38,0)`，需要 **16 字节对齐**的 buffer。但 Ray 的 plasma object store 返回 **8 字节对齐**的 buffer，`combine_chunks()`/IPC 都只补到 8 字节 → DuckDB 的 arrow-rs C-stream 导入**直接 panic**。

解法：`_ray_shim._realign()` 用 `Table.take(全部索引)`（一个 C gather kernel）强制重新分配对齐 buffer，在每个跨进程 register 点调用。这个修复惠及**所有**跨进程分布式 decimal 算子，不只 DAG 执行器。

### 3.8 到底怎么用 Ray（细节版）

前面讲了「用 Ray」，这节把**具体怎么调 Ray API、每个 Ray 概念对应 jude 的哪块**讲透。jude 只用 Ray 的 4 个原语：**Actor、Task、ObjectRef、`ray.wait`**。

#### 3.8.1 Actor：常驻 worker = 常驻 DuckDB

jude 的 worker 是 **Ray Actor**（`_JudeWorker`, `_ray_shim.py:50` `@ray.remote`），不是无状态 task。为什么用 actor：
- **DuckDB 连接常驻**：actor 在 `__init__` 里建一个 `jude.connect()`（`:65`），后续所有 SQL 复用它，省去每次建连接/注册 UDF 的开销。
- **数据亲和**：actor 是长生命周期的，Ray 会尽量把后续任务和它的数据放在同一节点。

创建（`make_workers`, `:314`）：
```python
_JudeWorker.options(num_gpus=g).remote(num_gpus=g)   # g>0 时声明 GPU 需求
```
- `.options(num_gpus=g)` 告诉 **Ray 调度器**：这个 actor 需要 g 个 GPU，Ray 只会把它放到有空闲 GPU 的节点——**多机放置由 Ray 负责，jude 不管物理节点**。
- `RayRunner._ensure_workers()`（`ray.py:121`）懒创建一次，缓存在 `self._workers`，整个 runner 生命周期复用。actor 数 = `num_workers`（默认 = 集群 CPU 数）。

#### 3.8.2 ObjectRef：分布式数据的「遥控器」

`worker.method.remote(args)` **立刻返回一个 `ObjectRef`**（一个 future/句柄），方法在 actor 上**异步**执行，结果存进该节点的 **plasma object store**（共享内存）。

关键：**Arrow 数据以 ObjectRef 形式在系统里流动，driver 拿的是句柄不是数据本体**。
- 当你把一个 ObjectRef 作为参数传给另一个 `.remote()` 调用，Ray **自动解引用**：如果目标 actor 在同节点，零拷贝共享内存读；跨节点，Ray 自动做 **object transfer**（网络拉取），对 jude 代码透明。
- 这就是 shuffle 数据「不经过 driver」的机制：`join_bucket_group(左片refs, 右片refs)`（`ray.py:610`）传的是 ObjectRef 列表，reducer actor 在**自己**那侧 `ray.get(refs)` 把分片从各 producer 的 object store 拉过来——driver 只做了「把句柄从 producer 的返回值路由到 reducer 的入参」这一步编排。

#### 3.8.3 `num_returns`：一次调用产出多个分片（shuffle 的 fan-out）

普通 `.remote()` 返回一个 ObjectRef。shuffle 的 producer 要把一个分片切成 b 个桶、每个桶是**独立**的 ObjectRef，才能分别路由给 b 个不同 reducer：
```python
w.bucketize.options(num_returns=b).remote(part, key_expr, b)   # ray.py:595
# 返回 [ref_bucket0, ref_bucket1, ..., ref_bucket(b-1)]
```
`_refs_bucketize`（`ray.py:632`）构造出 `refs[分区][桶]` 的二维句柄表；reducer 侧按桶取列：`[refs[p][bkt] for p in 各分区]` 就是「桶 bkt 的所有分片句柄」。**整个洗牌的数据搬运是 Ray 在 producer/reducer 之间直接做的，driver 只在重排一个二维 ObjectRef 数组。**

#### 3.8.4 `ray.wait`：背压 = 有界并发窗口

如果一次性 submit 几千个 task，object store 会被中间结果撑爆。jude 的背压在 `_ray_shim.run_bounded`（`:328`）：
```python
window = mgr.dispatch_window(len(fns))   # 窗口大小由 Rust 决定
# 先填满 window 个 in-flight；然后每 ray.wait 回收一个，就补一个新的
done, _ = ray.wait(list(inflight), num_returns=1)
```
- **策略在 Rust**（`WorkerManager.dispatch_window` 返回 `max_task_backlog` 或 0=无限）；**机制在 Python**（`ray.wait` 循环）。这条分工是「决策在 Rust、手脚在 Ray」的典型。
- 结果按**提交顺序**归位（`results[idx]`），所以 map/scan 的输出顺序稳定。
- GPU 任务走另一条：`_dispatch_admission`（`ray.py`）在 submit 前经 `ResourceManager.try_reserve` 预留 GPU/内存，完成后 release——防止一堆推理 task 把 GPU 超订。

#### 3.8.5 一次分布式 join 的完整 Ray 调用时序

```
driver: 左表 _partition_tables → [L0, L1]              (driver 本地切片)
driver: ray.put? 否——L0/L1 由 run_sql_on_table.remote 产出为 ObjectRef
driver: 对每个左分片  w[i].bucketize.options(num_returns=b).remote(Li) → 左refs[i][0..b]
driver: 对每个右分片  w[j].bucketize.options(num_returns=b).remote(Rj) → 右refs[j][0..b]
        （以上 .remote 全部立即返回句柄，actor 们已并行在跑 bucketize）
driver: for bkt in 0..b:
            reducer = w[bucket_workers[bkt]]
            结果refs[bkt] = reducer.join_bucket_group.remote(
                              [左refs[p][bkt]...], [右refs[p][bkt]...])   ← 传句柄
                            # reducer 内部 ray.get 这些句柄：Ray 自动跨节点拉分片
driver: ray.get(结果refs)  → 拼接                     (只有最终结果回 driver)
```
全程 driver 只做「切片 + 路由句柄 + 收尾 ray.get」，**所有 O(数据量) 的搬运和计算都在 actor 之间**。producer 的 bucketize 和 reducer 的 join **时间上重叠**（pipelined）——某个桶的分片一就绪，reducer 就能开始，不必等所有 producer 完成。

#### 3.8.6 多机放置与观测

- **放置**：jude 从不指定「任务去哪个物理节点」。它只声明资源需求（`num_cpus`/`num_gpus`），Ray 的调度器决定落哪个节点；`ClusterScheduler`（`src/dist/cluster.rs`）做的是**跨查询**的 bin-pack 提示，不是替代 Ray 的放置。
- **多机验证**：`benchmarking/bench_multinode.py` 用 `ray.cluster_utils.Cluster` 起「head + N worker 节点」（各自独立 raylet + object store），证明 actor 池会散布到多节点、吞吐随节点近线性增长。
- **节点观测**：`_ray_shim.cluster_nodes()`（`:297`）读 `ray.nodes()` 的存活节点 + CPU/GPU/内存，喂给 `observe` 和 `ClusterScheduler`。

#### 3.8.7 Ray 是可选依赖

`import` Ray 失败时，`jude.runners.get_or_create_runner()` 回落到 `LocalRunner`（单进程）。所以没装 Ray 也能跑单机；分布式路径按需启用。

---

## 4. UDF 基建（怎么把 Python 函数并行且绕开 GIL）

### 4.1 问题：GIL

`map_batches(fn)` 要把一个 Python 函数作用到每个 batch。Python 的 GIL 让**同进程内**多线程跑 Python 函数无法并行。所以要么多进程、要么多 Ray worker。jude 提供 4 个后端。

### 4.2 四个后端（`execution_backend=`）

| 后端 | 机制 | 何时用 |
|---|---|---|
| `in_process` | 同进程直接调（GIL 串行） | 基线 / 小数据 |
| `subprocess` | Rust 起一池子进程，管道传 Arrow IPC，**释放 GIL** | 单机 CPU 密集 UDF（**最快**，见 bench） |
| `ray_task` | 每 batch 一个无状态 Ray task | 弹性、纯函数 |
| `ray_actor` | 常驻 Ray actor 池，每 actor 加载一次 UDF（含模型权重） | GPU 模型推理、有状态 |

### 4.3 subprocess 池（jude 的招牌，Rust 实现）

`src/udf/subprocess.rs`：
- `SubprocessPool`（`:90`）持有 N 个 `Worker`，每个 Worker 是一个 `Child` 进程（`:19`），stdin/stdout 管道。
- 进程里跑 `python/jude/execution/_worker.py`：unpickle UDF（cloudpickle）→ 循环读 Arrow IPC 帧 → 作用 UDF → 写回 IPC 帧。
- `map_batches`（`:112`）：Rust 侧用 `std::thread::scope` 起线程分发 batch 给各 worker，**关键是分发时释放 GIL**（Python 函数在子进程里跑，主进程无 GIL 争用）。
- **池子缓存**（`:225` `pool_registry`）：按 `(python, num_workers, UDF哈希)` 缓存池，spawn 成本每个 UDF 只付一次。

为什么快：调度在 Rust（无 GIL），执行在子进程（无 GIL），Arrow IPC 零拷贝传输。bench 显示 12 核上 ~7.6x 于单进程，且胜过 Daft 最佳配置 1.6x。

### 4.4 ray_actor 常驻池（GPU 推理用）

`python/jude/execution/udf_ray.py` `RayActorExecutor`：
- `_ACTOR_POOLS`（`execution/__init__.py:26`）按 `(UDF哈希, workers, gpus, call_mode)` **缓存 actor 池**——之前每次 map_batches 都重建池、白付 actor 启动成本，现在常驻复用，吞吐从 3.82x 提到 6.01x。
- 每个 actor `setup()` 时加载一次模型权重（`_RayUDFActor`），后续 batch 复用（有状态契约）。
- GPU：`num_gpus>0` 时通过 `ResourceManager` 做准入（reserve/release），避免超订 GPU。

### 4.5 UDF 的 5 种形态

不止「一个 batch → 一个 batch」的 map。jude 支持：scalar（逐值）、vectorized（`VArrowScalar` 向量化）、table（生成器 UDF）、flat_map（一行→多行）、aggregate（分组聚合，借 DuckDB `list()` group-apply）。都走同一套后端分发。

### 4.6 map_batches 分布式版

`map_relation`（`ray.py:452`）：`_partition_tables` 切片 → 每片 `map_partition.remote(part, udf_payload)` → GPU 走资源准入，CPU 走计数窗口背压。这是「多模态批量推理」的主路径。

---

## 5. 资源与调度（保护集群不 OOM）

- **ResourceManager**（`src/dist/resource.rs`）：4 维资源向量（cpu/gpu/内存/object-store），reserve/release 准入。GPU 批推理靠它不超订。
- **ClusterScheduler**（`src/dist/cluster.rs`）：跨查询 worst-fit bin-pack，把任务塞进最合适的节点。
- 两者都是 Rust 纯决策 + Python 执行 reserve/launch/release。

---

## 6. 可观测性（怎么看到发生了什么）

- `src/observe.rs` `MetricsRegistry`：进程全局、Mutex 保护、GIL-free。记录 query 生命周期、stage 进度（rows/bytes/tasks/attempts）、UDF 池利用率、集群节点。
- `python/jude/observe.py`：单例 + `query()` 计时上下文 + `ray.nodes()` 轮询 + HTTP `/api/metrics` 端点。
- `frontend/`：React 仪表盘，1.5s 轮询 `/api/metrics`，展示节点/stage 进度条/query 链/UDF 池/活动流。

---

## 7. 常见问题速查

**Q：为什么调度非要在 Rust？** A：Vane 的分布式控制面是 ~2.8 万行 Python，受 GIL 拖累。jude 把决策移到 Rust，调度不占 GIL，且可单测。

**Q：collect() 怎么知道该走哪条路？** A：`ray.py:313` 先看 `plan_stages()`；嵌套 shuffle → `execute_dag`（通用流式执行器）；单 shuffle → 特化的 `distributed_aggregate`/`distributed_join_streaming` 等；无 shuffle → 并行分片扫描后拼接。

**Q：shuffle 数据会不会挤爆 driver？** A：不会。producer→reducer 走 object store，driver 只传 ObjectRef。

**Q：subprocess 和 ray_actor 怎么选？** A：单机 CPU 密集 → subprocess（最快）；需要 GPU / 有状态模型 / 多机 → ray_actor 常驻池。

**Q：加了新分布式算子要动哪些文件？** A：Rust 决策进 `src/dist/`；若涉及 plan 分解，改 `physical.rs` + `relation.rs:dist_step`；Python 执行进 `ray.py` + worker RPC 进 `_ray_shim.py`；写对拍测试（分布式结果==单机 ground truth）。

---

## 附：关键文件地图

| 文件 | 职责 |
|---|---|
| `src/plan.rs` | LogicalPlan IR + to_sql |
| `src/dist/worker_manager.rs` | 调度大脑（分区/分配/窗口/分桶决策） |
| `src/dist/stage.rs` | stage DAG 形状（shuffle 边界切分） |
| `src/dist/physical.rs` | 单步计划分解（peel 顶部 pw 区域 + 边界 + 子计划） |
| `src/dist/split_assigner.rs` | Arbitrary/Hash/Single split 分配器 |
| `src/dist/resource.rs` | 4 维资源准入 |
| `src/dist/cluster.rs` | 跨查询 bin-pack |
| `src/relation.rs` | Relation API + dist_step/aggregate_spec/join_spec |
| `src/udf/subprocess.rs` | Rust 子进程 UDF 池 |
| `src/observe.rs` | 指标注册表 |
| `python/jude/runners/ray.py` | RayRunner：分布式执行编排 + execute_dag |
| `python/jude/runners/_ray_shim.py` | _JudeWorker actor + Ray RPC 原语 |
| `python/jude/runners/_agg.py` | 两阶段聚合分解 |
| `python/jude/execution/udf_ray.py` | ray_task / ray_actor 执行器 |
| `python/jude/execution/_worker.py` | 子进程 UDF worker 主循环 |
| `python/jude/observe.py` | 可观测性 facade + HTTP 端点 |
| `frontend/` | React 仪表盘 |
