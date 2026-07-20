# jude 的分布式执行

## 赌注

jude 之所以存在，是押了一个注：Vane 的分布式层慢，慢在了错的地方。Vane fork 了 DuckDB，然后在外面
裹了一层约 2.87 万行的 Python 控制平面——driver、调度器、split 分配器、资源管理器、容错追踪器——而这些
组件全都跑在解释器里、受 GIL 约束、且位于每一次任务派发的热路径上。当你要在集群上调度成千上万个 split
时，调度器本身就是负载，而一个受 GIL 约束的调度器，恰恰把你想并行的那件事串行化了。

所以 jude 不 fork DuckDB，也不在 Python 里调度。它在**分区级别**基于原生 DuckDB 编排——和 Daft、Ray
Data 同一形态——并把编排放进 Rust。Python 只作为 Ray 的 RPC 边界存活：它持有 ObjectRef，调用
`.remote()`，在 `ray.get` 上阻塞。它不做任何决策。这个分工就是整个设计，而本文大部分篇幅都在为
"决策"与"转发"之间那条线辩护，因为性能主张就活在这条线上，也最容易在这里不小心作弊。

## 为什么选分区级，代价是什么

Fork DuckDB 并教它的优化器产出分布式 plan（Vane 的做法，也是 Presto、Spark 的做法）能换来流水线化、
算子融合的分布式执行：hash join 可以跨网络流式地跑 build 和 probe，无需物化。代价是你从此永远维护一个引擎
fork，而且它把调度器**拽进**了引擎里——变成 C++ 为了 UDF 回调 Python，正是 jude 要躲开的税。

分区级编排（jude 的选择）把 DuckDB 当作黑盒单机执行器，永远只递给它一个 Arrow 数据*分区*加一条 SQL。
分布式逻辑完全在引擎之外：切分输入、在每个分片上跑同一条本地查询、需要全局视图时 shuffle、最后 merge。
若拿一个*假想的全流水线引擎*（流式 Presto/Spark）作参照，代价是我们在 shuffle 边界物化——那种系统会融合并
流式跑的算子，在这里变成两个阶段中间夹一次 exchange。

这里必须说精确，因为这是本项目的核心对比、也最容易搞错：**这相对 Vane 并不是劣势。** Vane 跑的是
*容错执行*（FTE）引擎，而 FTE 按其构造就会物化 exchange——`vane/duckdb/runners/fte/fte_exchange.py`
把每个 exchange 分区按 attempt 落到文件系统路径，好让丢失的任务能靠重读落盘数据重试。Vane 同样**不**跨
shuffle 边界做流水线。所以在"shuffle 边界物化"这条轴上，jude 与 Vane 是同一形态；而在两条有差别的轴上，
jude 在速度上领先：jude 让 shuffle 分区走 **Ray 对象存储（内存）**，Vane 落**磁盘**；jude 围绕 shuffle 的
调度是 **Rust、无 GIL**，Vane 是 Python——而 shuffle 密集意味着任务多，正是受 GIL 约束的调度器吃亏之处。
Vane 落盘唯一换来、而 jude 暂时没有的，是 shuffle 的*容错*：jude 目前把 exchange 数据留在对象存储、无重试，
用容错换了 happy-path 速度。这是**功能**缺口（见"刻意还没做的"），不是更慢的 shuffle，且 `FteTaskAttemptId`
词汇就是为日后用可选落盘补上它而备。

分区级执行真正会输的地方，是 join 树很深、每分区计算量又小的查询——TPC-DS 那类分析查询——因为那里每阶段
的物化与调度开销盖过了实际计算。jude 瞄准的负载正相反（对多模态数据做扫描→解码→map→落地，加可分解聚合与
hash join）：shuffle 边界少、每分区计算量巨大，物化代价约等于零。我们明确暂不为深 join 树那种场景优化。

### 把 shuffle 边界画出来

设一个两阶段查询：扫描+局部聚合，然后**按分组键 shuffle**，再最终聚合。全部问题就在 shuffle 处发生什么。

```
模型 A — 流水线（流式 Presto/Spark；非容错）
Stage-1 的任务把行直接流给 Stage-2；Stage 2 在 Stage 1 结束前就开始跑。

  S1-p0 ─行┐
  S1-p1 ─行┼──▶（网络，实时）──▶ S2-p0, S2-p1   ← 已经在跑
  S1-p2 ─行┘
        无 barrier · 什么都不落 · 最快 · 一个任务死 = 全部重来

模型 B — 物化 exchange   ← Vane 和 jude 都是这个
每个 Stage-1 任务先跑完并**写下**输出；写完 Stage 2 才读。

  S1-p0 ─▶[写]┐
  S1-p1 ─▶[写]┤   ══ barrier ══▶   S2-p0 ◀─[读]
  S1-p2 ─▶[写]┘                    S2-p1 ◀─[读]
        先物化再下一阶段 · 比 A 慢 · 但一个任务死了只需重读/重跑
```

相对 Vane，jude 并没有选慢的那侧——两者都是模型 B。唯一区别是*写到哪*：

```
  Vane (FTE)：  S1 ─▶ 💾 磁盘文件        ─▶ S2 从磁盘读
                     持久 → 能重试丢失任务，但付磁盘 I/O
  jude：        S1 ─▶ 🧠 Ray 对象存储    ─▶ S2 从内存读
                     更快（无磁盘）；暂无重试 → 无容错
```

所以在这条查询上：流水线引擎 = 无 barrier（最快、脆弱）；Vane = barrier + 磁盘；jude = barrier + 内存
（happy-path 比 Vane 快，暂不容错）。深 join 树那条注脚，只是说 barrier 会叠多少个——jude 的目标流水线约
为 0，barrier 成本为零；TPC-DS 那类有很多个，那里模型 B（含 Vane）都会输给流水线引擎。

## 那条线：决策 vs 转发

一切构成*调度决策*的东西都在 Rust，在 `src/dist/`。一切碰 Ray 句柄的东西都在 Python，在
`python/jude/runners/_ray_shim.py`。我对每一行代码用的判据是：*如果我删掉 Ray、换一个执行底座（本地线程
池、别的 actor 框架），这行能活下来吗？* 能，它就是决策，归 Rust；要重写，它就是 RPC 胶水，归 shim。

具体说，决策是：把输入切成几个分区、这些分区的行边界落在哪、哪个 worker 跑分区 *i*、同时最多几个任务在飞、
以及——对 join——shuffle 成几个哈希桶、每个桶归哪个 worker。转发是：初始化 Ray、构造 actor、调
`.remote()`、跑收集结果的 `ray.wait` 循环。shim 里没有任何对数据大小的算术、没有配置常量、没有策略分支——
这由 CI 里的一条 grep 强制，而且不是玩笑：侵蚀性能主张最可能的方式，就是某个"小小的"sizing 调整因为
ObjectRef 恰好在 Python 而落在了 Python。当你需要这种调整时，它进 `WorkerManager`，然后把答案告诉 shim。

这条线只有一处确实会弯，值得诚实说明：那个把 *N* 个任务保持在飞的有界派发循环——预热窗口，然后
`ray.wait(num_returns=1)`，弹出完成的 ref，提交下一个——它物理上必须在 Python 跑，因为 Ray ObjectRef
进不了 Rust。我们考虑过让 Rust 驱动循环、每完成一个就回调 Python，否决了：那是把"每批结果一次 GIL"换成
"每完成一个一次 GIL"，严格更差且无收益。所以循环在 Python——但它循环到的*窗口大小*在 Rust 算
（`WorkerManager::dispatch_window`），循环本身完全不知道窗口为何是这个值。策略在 Rust，机制才在 Python。

## Rust 侧的解剖

Rust 编排自底向上由三层构成（已存在并编译通过），外加一个即将到来的规划器。

**词汇层**（`src/dist/fte.rs`）是每个分布式引擎都需要的名词集，从 Vane 的 `fte_types.py` 移植。`FteTaskId`
是一个工作单元的逻辑标识——`(query, fragment-execution, partition)`——`FteTaskAttemptId` 再套一个尝试号，
让重试和推测副本可寻址。`FteSplit` 是一份不可再分的输入：扫描 split（文件的一段、若干 parquet 路径）或
exchange split（某上游 shuffle 分区的输出）。它带 `size_bytes` 让分配器按字节而非按个数装箱，带
`addresses` 做局部性——已携带但尚未使用，我在此标明而非假装已接线。`TaskDescriptor` 是一个分区的可变、
增长的记录：split 随上游 fragment 产出而增量到达，所以 `append_splits` 按 sequence id 去重，并在每次真实
变化时递增 `descriptor_version`。那个版本号不是装饰——它让一个运行中的 worker 能收到*增量*（"再给三个
split，版本 7"）而非整包重发，也让 `seal_source` 能宣告某输入已耗尽从而 worker 知道可以收尾。这是增量、
容错调度的机件；今天大部分还处于潜伏态（因为可达算子尚未流式产出 split），但这是对的词汇，且已在 Rust 里，
而非等着日后在时间压力下再移植。

**分配器层**（`src/dist/split_assigner.rs`）把 split 流变成分区。三种策略，一个 trait。
`SingleSplitAssigner` 把一切汇入分区 0——退化情形，但对全局聚合是真实的。`HashSplitAssigner` 按
`source_partition_id % n` 路由，这是哈希 shuffle 让相同 key 共置的方式。有意思的是
`ArbitrarySplitAssigner`，它做基于字节的装箱并*自适应增长*：头几个分区故意小以便快出首结果（延迟），目标
分区大小几何增长——每装 64 个 split ×1.26，到上限为止——于是后面的分区大、吞吐高效。这是 Vane 调好的
启发式的直接移植（64 MiB 标准 split，每任务 2048 split 上限），做到*逐位一致*很重要，因为这是"jude 调度得
像 Vane、只是用 Rust"和"jude 调度得不一样、于是每次性能对比都被污染"的分野。它还处理广播（复制）源——标记
为 replicated 的 split 会扇出到每个分区，且晚创建的分区会追溯性地收到此前见过的复制 split，这是天真重写会
搞错的那个棘手正确性细节。

**大脑层**（`src/dist/worker_manager.rs`，暴露为 `jude.dist.WorkerManager`）是 runner 实际调用的 pyclass。
它持有配置——worker 数、size-grouping 开关、backlog 上限、open-cost 字节目标、最小分区下限——构造时从环境
读一次，并回答那五个调度问题。`target_partitions(nbytes, num_rows)` 是 DuckDB-Python 继承的 Spark 式
sizing 的逐行移植：下限取 `min_partition_num or num_workers` 让没有 worker 闲着，且 size-grouping 开时还要求
至少 `ceil(nbytes / open_cost_bytes)` 个任务、让单个任务不至于大得离谱。`partition_plan` 返回实际的
`(start, len)` 行切片——manager 决定切在*哪*，Python 只调 `table.slice`。`dispatch_window` 返回在飞上界。
`shuffle_bucket_count`/`shuffle_bucket_workers` 回答 join 的问题，后者实际跑一遍移植的
`HashSplitAssigner` 得到规范桶集再把桶轮询到 worker——于是连 join 的路由决策都*经过* Rust 分配器，而非手工
重算。九个单测把这些钉在旧 Python 产出的确切值上，这才让我能说重写是行为保持，而非仅仅看着像。

**规划器层**（`src/dist/stage.rs`，下一步）把今天埋在 `Relation::plan_json` 里的单层 `match` 推广开。
`LogicalPlan` 是棵树；分布式 plan 是把这棵树在需要全局视图的算子处切成阶段——聚合、join、distinct、排序、
集合运算、显式 repartition。规划器遍历树，在每个这样的边界产出一个阶段，携带其本地（非 shuffle）工作的
SQL、它 shuffle 所依据的分区键、以及上游依赖。这是通用 N 阶段流式执行器消费的产物；我交付的是那个*规划*，
并明确执行器（对任意阶段 DAG）是后续工作，因为今天真正可达的两个算子——聚合和 join——由专门的两阶段代码
处理（见下），假装不是这样，正是本设计力图避免的那种天真过度承诺。

## 一次查询到底怎么跑

先看常见情形：分区扫描或分布式 `map_batches`。relation 在 driver 上物化成一张 Arrow 表；
`WorkerManager.partition_plan` 把它切成分片；每个分片作为一次 `.remote()` 调用提交给 `worker_for(i)`，
结果是 ObjectRef；shim 的 `run_bounded` 按 manager 选的窗口、以提交顺序收集这些 ref，交回表。UDF（若有）
在 driver 上按值 cloudpickle、在 actor 内部解开，对着该 actor 自己的原生 DuckDB 连接运行。这条路径没有一处
在 Python 里算过调度。

两阶段聚合是分区级模型真正发力之处。`_agg.build_two_phase`（纯 SQL 字符串操作——它不决定任何放置，所以留在
Python）把 `GROUP BY … COUNT/SUM/MIN/MAX/AVG` 重写成一条*partial* 查询和一条*final* merge 查询。partial
在每个分区上并行跑（派发方式同上）；driver 拼接 partial——这里有个花了真金白银时间的真实 bug：
`concat_tables` 可能留下未对齐的 Arrow 缓冲，而 arrow-rs 的 C-stream 导入器遇到未对齐缓冲会 *panic*，所以
拼接后、跨回 Rust 前必须 `combine_chunks()`——然后在本地 DuckDB 上跑 final merge。`COUNT` 变成 count 的
`SUM`，`AVG` 分解成 `SUM/COUNT` 再重组，结果与单机答案逐位一致，测试直接断言相等而非近似。

hash join 是 jude 唯一做真正 shuffle 的地方。两侧都按 `hash(keys) % b` 分桶——*在 SQL 里*用 DuckDB 做，
因为哈希是执行而非调度——其中 `b = WorkerManager.shuffle_bucket_count`。相同 key 在两侧落进同一个桶，于是
左侧桶 *i* 和右侧桶 *i* 可以在一个 actor 上本地 join，由 `shuffle_bucket_workers[i]` 选定。join 投影是
`lhs.*, rhs.* EXCLUDE(keys)`，共享 key 列不重复。空结果情形有个代码处理的细节：若每个桶都空，你就丢了输出
schema，于是它专门重跑桶 0 来恢复列类型。

## 我刻意还没做的

没有容错：尝试号词汇已就位，但没有重试策略、没有推测执行、没有丢分区恢复。没有局部性感知放置：
`FteSplit.addresses` 已携带但被忽略，`worker_for` 是盲目轮询。没有通用阶段执行器：只有聚合和 join 有分布式
实现、都走两阶段路径，任意 N 阶段 DAG 是规划而未执行。这些被点名为缺口而非抹平，因为引擎的诚实状态是
"编排大脑真实且在 Rust，可达算子能跑且逐位精确，通用流式运行时是下一个里程碑"——而一份暗示不是如此的设计
文档，恰恰是对下一个读它的人毫无用处的那种天真。

## 里程碑

已完成：FTE 词汇与三个 split 分配器；`WorkerManager` 大脑暴露为 `jude.dist`、九个单测通过；把 `_JudeWorker`
actor 迁入的那个薄 Ray shim；改接线后的 `RayRunner`——每个决策都下放 Rust，且既有 Ray/scheduling 测试原样
通过。下一步：`StagePlanner` 做递归 shuffle 边界分阶段。之后大致按优先级：任意阶段 DAG 的流式执行器、局部性
感知分配、基于尝试号词汇的重试/容错。
