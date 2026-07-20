# Jude → Vane 复刻能力差距分析与实施计划

> 目标：逐步完美复刻 Vane（`/Users/zzywq/code/vane`）的能力，并在性能上超过它。
> 本文基于对两个代码库的完整源码分析（2026-07-18）。

---

## 实施进度（2026-07-18，持续更新）

已完成并提交（每阶段独立 commit；全套 179 测试绿）：

**架构地基（按用户修正重做）：**
- **Relation 逻辑计划树 IR** ✅ — `src/plan.rs` 的 `LogicalPlan` enum（Filter/Project/Aggregate/Join/SetOp/Order/Limit/Repartition/MapBatches/… + Arc 子节点）组成真正的算子 DAG，弃用 SQL 字符串拼接（旧方案扩展性差）。SQL 只是 `to_sql()` 一种 lowering；`rel.plan_tree()` 打印 IR。与 Vane 的原生算子 DAG 同构。
- **多模态类型系统** ✅ — `jude/types.py`：`TensorType`（Arrow fixed_shape_tensor，dtype+shape，存 DuckDB 降级 fixed_size_list，shape 可恢复）+ Image/Audio/Video/Document（binary）+ tensor/图像/音频编解码。**端到端验证：image bytes→model→embedding tensor，Ray actor 分布式，40×8 矩阵。**
- **多后端执行引擎** ✅ — `jude/execution/`：subprocess pool（Rust，GIL-free，6.5x）、`ray_task`（无状态）、`ray_actor`（有状态池，GPU pin）、流式 `imap`、call modes。对齐 Vane `duckdb/execution/`。（ref-bundle 块传递 shuffle 为后置优化）
- **cosmos-xenna 直接集成** ✅ — `jude.pipeline` 直接 re-export cosmos-xenna（Stage/PipelineSpec/run_pipeline/Resources），作为多模态 pipeline 引擎（不给 SQL 用）；无 cosmos 时本地 fallback。

**能力/引擎：**
- **Phase 0** ✅ 真 lazy Relation + 零拷贝 Arrow（C Stream Interface）+ 完整类型提取（Decimal/日期/时间戳/List）。
- **Phase 1** ✅ map_batches/flat_map/repartition + jude.runners；out-of-process UDF 引擎（6.5x）。
- **Phase 4** ✅ Ray 分布式运行器（partition 级，不 fork）。
- **Phase 5** ✅ DBAPI `execute().fetch*`、`create_function`、cursor/事务、**两阶段分布式聚合**、**分布式 hash join**、Polars/NumPy/Pandas 导出、**replacement scan**（pandas/polars/pyarrow 变量名自动解析）、Arrow 对齐修复。

**性能主张已验证**：Rust 编排 + GIL 旁路 = CPU UDF 6.5x。分布式聚合/join 位精确对齐单机。

**关键决策（用户锁定）**：worker 用 Ray（PyO3），**不 fork DuckDB**；能力全对齐；Phase 2/3（AI/vLLM）暂缓。

**仍待**：ref-bundle 块交换优化；更多 Vane 表达式类（`ConstantExpression(Value(...))`）；AI/vLLM（Phase 2/3）；跑 Vane `multimodal_inference_benchmarks` 真实对比。

---

## 0. 一句话结论

Vane 不是"一个 DuckDB 封装"，而是 **一个被 fork 改造的 DuckDB C++ 引擎 + 一整套分布式容错执行系统（Ray）+ 多模态/AI 推理层**，第一方代码约 **10.5 万行**（还不含被 fork 的 DuckDB 引擎本身，那是几十万行）。

Jude 目前是 **约 4200 行的 Rust + Python 薄封装**，基于**未修改的 stock `duckdb` crate**，其中：
- **能跑的**：`conn.sql()`、参数化 `execute`、标量 UDF、表达式构建、AI embed/classify/prompt（HTTP）、token 统计、env/config。
- **是假的/桩**：`Relation.filter/project/aggregate/join/order/distinct` 全部只存一个假字符串标记（如 `"__filter__..."`）**从不执行**；`Relation` 是 eager 立即物化（与 Vane 的 lazy plan 相反）；`from_arrow`/`to_table` 走 `/tmp` parquet 文件中转；`register()` 是空操作；Runner 是桩（没有任何 Ray）；`map_batches` 单线程进程内；无多模态、无 vLLM、无分布式、无流式、无反压。

**差距量级**：封装层约 25 倍代码差距；执行引擎层差距为"无穷"（Vane 有分布式引擎，Jude 没有）。这是一个多人月到多人年的工程。

**性能能否超过 Vane？能，但只有一条可行路径**（见 §4）：利用 Jude 是 Rust 的天然优势，用 **Rust 原生编排层** 替代 Vane 那 **~2.87 万行 Python 分布式控制面**。这是 Jude 唯一真实、可辩护的性能优势来源。

---

## 1. 根本架构差异（这是理解一切的前提）

| 层 | Vane | Jude |
|---|---|---|
| **SQL 引擎** | **Fork 了 DuckDB C++**（`AstroVela/duckdb` 子模块），在引擎内部加了分布式算子（`PhysicalRemoteExchangeSink`、exchange source、`PhysicalStreamingUDF`）、`TensorType` 多模态类型、`UDFExecutor` 工厂接口、`Repartition`/`LocalExchange` 算子 | **stock `duckdb` crate（bundled，未改）**——拿不到任何上述引擎内算子 |
| **绑定层** | pybind11 C++，`src/duckdb_py/` **~38.7k 行** | PyO3 Rust，`src/` **~3.5k 行** |
| **执行模型** | **Lazy**：每个 relation 方法构建 `shared_ptr<Relation>` 计划树，直到 fetch 才执行 | **Eager**：`sql()` 立即物化成内存 Arrow batches；relation 算子是桩 |
| **分布式控制面** | `duckdb/runners/` **~28.7k 行 Python**：Ray driver、FTE 调度器、worker actor、split 分配、推测执行、Flight shuffle | `runners/local.rs` **31 行**（原样返回 batches，无 Ray） |
| **UDF 执行引擎** | `duckdb/execution/` **~11.5k 行 Python**：子进程池 / Ray task / Ray actor / 流式 / ref-bundle，`udf_executor.cpp` 3781 行 C++↔Python 桥 | `expression_udf/registration.rs` **196 行**：进程内标量 UDF（1-4 参数），GIL 内逐行 |
| **AI/推理层** | `vane/ai/` ~6k 行 + vLLM C++ 桥 + prefix-aware bucketing | `src/ai/` ~1.5k 行 Rust（HTTP 到 3 家 + transformers），无 vLLM |
| **第一方代码总量** | **~105k 行**（不含 DuckDB fork） | **~4.2k 行** |

> 关键事实：Vane 的 C++ 绑定层里那些 `rel->Repartition(...)`、`rel->LocalExchange(...)`、`FunctionExpression("udf", ...)`、`TensorType`、`PlanRunner`、`WorkerManager`、`FteSplitQueue`、Flight shuffle **全部依赖被 fork 的 DuckDB 引擎**。这个 fork 不在工作副本里（子模块未 checkout），但它是整个系统的算法核心。

---

## 2. 分子系统能力差距明细

### 2.1 Relation / 关系代数
- **Vane**：完整 lazy 计划树。`filter/project/select/order/sort/aggregate/join/cross/distinct/union/except/intersect/limit/explode` 全部真实执行；外加分布式算子 `repartition`、`local_exchange`；海量聚合/窗口糖（`arg_max`、`quantile_cont`、`row_number`、`lag/lead` 等，用 SQL 串生成器实现）；`map_batches`/`flat_map`/`map` 流式算子。
- **Jude**：`select`/`limit`/`union` 对内存 batch 真实操作；`filter/project/aggregate/join/order/distinct/intersect/except` **全是桩**（存假字符串，不执行）。`Relation` 无连接引用，无法重新执行 SQL。
- **差距**：需要把 Relation 从 eager 改为 **lazy 计划构建器**（持有连接 + SQL/计划），所有算子转成真实 SQL 生成或 DuckDB relation API 调用。

### 2.2 Expression
- **Vane**：`DuckDBPyExpression` 包装 `ParsedExpression` AST，函数式构建；含 `case/when/else`、`lambda`、`coalesce`、`collate`、UDF 表达式工厂。
- **Jude**：`Expression` 已能生成 SQL 串（col/lit/算术/比较/逻辑/cast/between/in/isnull/asc/desc）——**这部分基本可用**，但缺 case/lambda/coalesce/collate 与 UDF 表达式。
- **差距**：中等，补齐 AST 节点即可。

### 2.3 UDF
- **Vane**：两条路径。(A) 经典进程内标量 UDF（stock 行为）。(B) **分布式流式 UDF**：把 pickle 的可调用对象打包成 payload STRUCT，经 `udf_executor.cpp` 的单后台线程 GIL 调度器分发到 **子进程池 / Ray task / Ray actor**；支持有状态 actor（`vane.cls`，`actor_number==1`）、批处理（`vane.cls.batch`）、GPU 分配、字节级动态批、流式 ref-bundle 交换、零拷贝 Arrow I/O。
- **Jude**：只有进程内标量 UDF（1-4 参数，GIL 内逐行）。`func`/`cls`/`cls.batch` 装饰器**只打标记属性，没有任何执行后端**。
- **差距**：巨大。这是 Vane 多模态批推理性能的核心，需要完整的 out-of-process UDF 执行引擎。

### 2.4 分布式执行（最大的差距）
- **Vane**：logical plan（client 序列化，不优化）→ driver 反序列化+优化+物理计划 → 切成 pipeline fragment DAG（以 exchange 为边界）→ `PlanRunner` 通过 `WorkerManager` 提交 `WorkerTask` → Ray actor 执行（GIL 释放，原生 DuckDB executor）→ 中间结果走 **Arrow Flight shuffle**，最终结果走 **Ray object store（`ray.put`）** → driver 异步轮询 result handle → 流式返回。含 **FTE 容错**（推测执行、重试、attempt 选择去重）、**动态 split 队列 + 反压**、**autoscaling**、**GIL 安全的跨线程销毁**。
- **Jude**：**完全没有**。`env.rs`/`config.rs` 里镜像了 Vane 的所有 `ray_*` 配置项（scan task grouping、max_task_backlog、open_cost_bytes 等）——**但没有任何实现**，纯占位。
- **差距**：无穷。这是要从零构建的最大子系统。

### 2.5 AI / 推理
- **Vane**：`embed_text`（含长文本分块+加权平均）、`embed`（表达式形式）、`classify_text`（零样本）、`prompt`（多模态 image_columns + 结构化输出 return_format + system_message）。Provider：OpenAI（embed/prompt/PDF/图像/结构化）、Anthropic（prompt/图像/tool_use 结构化）、Google（embed/prompt/图像）、**vLLM**（prompt/结构化，continuous batching + prefix-aware bucketing）、transformers（本地 embed/classify）。SQL 函数 `ai_prompt`/`ai_embed`。凭据安全（禁止内联 api_key）。
- **Jude**：`embed_text`/`classify_text`/`prompt`（tokio + semaphore 并发批），Provider：OpenAI/Anthropic/Google（HTTP）+ transformers（存疑，需核实是否真的加载模型）。**无 vLLM、无结构化输出、无 SQL 函数、多模态仅 prompt 的 image part**。
- **差距**：中等偏大。AI 层是 Jude 相对最接近的部分，但缺 vLLM（Vane 吞吐核心）和结构化输出。

### 2.6 多模态类型系统
- **Vane**：引擎内 `TensorType` 逻辑类型，UDF payload schema 里用 `kind="tensor"` + dtype + shape，Python 侧映射到 `pa.fixed_shape_tensor`。图像/音频/视频/文档统一表示。
- **Jude**：**无**。只有基础 Arrow 类型映射（且不全，很多类型 fallback 到 string）。
- **差距**：大，且依赖引擎层（TensorType 在 DuckDB fork 内）。

### 2.7 兼容层 / 生态
- **Vane**：完整 `duckdb` drop-in（re-export 所有符号）、DBAPI、Arrow/Polars/Pandas/NumPy 零拷贝、filesystem、replacement scan、`adbc_driver_duckdb`。
- **Jude**：Spark 兼容 shim（`experimental/spark`）——但 `filter`/`distinct` 用逐行 INSERT 实现（**极慢**）；Arrow 转换走逐值 Python 循环（慢）。
- **差距**：中等；很多可用 stock duckdb crate 能力补齐。

---

## 3. Jude 现状盘点（诚实版）

**真能用**：`connect` / `sql` / `execute(+params)` / `executemany` / 事务 / `read_csv|json|parquet` / 标量 UDF(1-4参) / Expression→SQL / AI embed·classify·prompt(HTTP) / token metrics / env / config。

**假的或严重低效**：
- Relation 关系代数（filter/project/aggregate/join/order/distinct/intersect/except）——**桩，不执行**。
- `Relation` eager 物化；无连接引用，算子无法链式真实执行。
- `from_arrow`/`to_table`/`to_csv`/`to_parquet` 走 `/tmp` parquet 文件或逐值 Python 循环。
- `register()` 空操作；`to_view`/`insert_into` 空操作。
- Runner 桩；`map_batches` 单线程进程内；`func/cls/cls.batch` 只打标记无后端。
- Spark `filter/distinct` 逐行 INSERT。

---

## 4. 性能策略：Jude 如何真的比 Vane 快

诚实前提：**当前 Jude 连 `.filter()` 都跑不了，谈性能为时过早。** 但一旦补齐功能，Jude 有一条**真实的、结构性的性能优势路径**：

### 4.1 核心论点：Rust 编排 vs Python 编排
Vane 的整个分布式控制面是 **~2.87 万行 Python**（driver、FTE 调度器、split 分配、result handle 轮询）。Python 控制面的固有成本：GIL 争用、调度延迟、序列化开销、`ray.put`/`ray.wait` 的 Python 往返。**Jude 已经是 Rust**——用 Rust 写编排层可以在以下方面结构性领先：
- 调度延迟与 fan-out 开销（无 GIL，真并行）
- 计划序列化/反序列化（Rust serde vs Python pickle）
- result handle 轮询与背压计费（Vane 用 C++ 后台线程 + `batch_wait_ready` 绕过 GIL，Jude 天然无此负担）
- UDF 分发（Vane 的 `udf_executor.cpp` 有一套"单线程独占 GIL"的复杂舞蹈，Rust 可以真并行分发）

### 4.2 关键架构决策：**是否 fork DuckDB？**
这是决定整个计划走向的岔路口：

**方案 A —— 也 fork DuckDB（照抄 Vane）**
- 优点：能拿到引擎内 exchange/streaming/TensorType 算子，行为与 Vane 一致。
- 缺点：维护地狱（跟上游 DuckDB rebase）、Rust 侧要重新绑定 fork、失去"用 stock crate"的简洁性、几乎不可能在性能上"更好"（因为核心就是同一个引擎）。

**方案 B（推荐）—— 围绕 stock DuckDB 做 Rust 原生分布式**
- 思路：**不改 SQL 引擎**，在 **partition 级别** 编排（像 Daft / Ray Data 那样）：Rust 把查询切成 per-partition 子查询，每个 partition 用 stock DuckDB 执行，shuffle 用 **Arrow Flight / 对象存储**，编排/调度/容错/背压全用 **Rust** 写。
- 优点：
  - 避开 fork 维护成本；
  - **Rust 控制面 vs Python 控制面 = 真实性能优势**；
  - stock DuckDB 引擎本身已是世界级向量化引擎，与 Vane 的 fork 基线相当；
  - 分布式 UDF/多模态批推理是"partition 级 map"，本就适合 partition 编排，不需要引擎内算子。
- 缺点：
  - 引擎内 exchange shuffle 要在编排层自己实现（可行，Daft 已证明）；
  - 复杂 SQL（跨 partition join/agg）的分布式化需要自己做 plan 切分（这是硬骨头）；
  - 无法 1:1 复刻依赖引擎内算子的行为。

> **建议**：走 **方案 B**。它是唯一能同时做到"复刻能力 + 性能更好"的路径。fork 引擎只会让你和 Vane 打平，无法超越。

### 4.3 其他可量化的性能抓手
- **零拷贝 Arrow FFI**：Jude 现在 `from_arrow` 走 `/tmp` parquet、`to_arrow` 逐值 Python 循环——改成 Arrow C Data Interface 零拷贝（`arrow` crate 的 `ffi`），单点就能有数量级提升。
- **UDF 无 GIL 分发**：Rust 侧 tokio + rayon 调度多个子进程/actor worker，避免 Vane 的单 GIL 线程瓶颈。
- **vLLM prefix bucketing**：这是纯算法（按公共前缀分桶提升 KV cache 命中），可在 Rust 里实现得比 Vane 的 C++ 算子更省——是可展示的 benchmark 亮点。
- **动态批 + 背压**：Rust 的 async + channel 背压比 Python asyncio semaphore 更低开销。

---

## 5. 分阶段实施计划

> 每个 Phase 结束都要有可跑的 pytest + benchmark，且 benchmark 与 Vane 同口径对比。

### Phase 0 —— 让核心真的能跑（地基，1-2 周）
**目标**：消灭所有桩，Relation 变 lazy 且真实执行。
1. `Relation` 改为持有 `Arc<Connection>` + 计划状态（SQL 串或 relation 引用），去掉 eager 物化。
2. 用 DuckDB relation API 或 SQL 生成实现真实的 `filter/project/select/aggregate/join/order/sort/distinct/union/except/intersect/limit`。
3. `from_arrow`/`to_arrow` 改用 **Arrow C Data Interface 零拷贝**，删除 `/tmp` parquet 中转。
4. `register()` 用 DuckDB 的 arrow/table 注册实现真实视图注册。
5. 修 Spark shim 的逐行 INSERT（改成 relation 算子）。
6. 补齐 Expression：`case/when/else`、`coalesce`、`collate`、`lambda`。
- **验收**：`test_jude.py` 全绿（现在很多是"跑而不断言"）；relation benchmark 与 Python duckdb 同口径。

### Phase 1 —— 本地 UDF 执行引擎（3-5 周）
**目标**：复刻 Vane 的 out-of-process UDF 后端（先做本地，不碰 Ray）。
1. UDF payload 格式（pickle callable + 输出 schema + backend + 批参数），对齐 Vane 的 STRUCT 契约。
2. **子进程池后端**（`subprocess_task` / `subprocess_actor`）：Rust 起 worker 进程，socket + shared memory + Arrow IPC 传输，Rust 侧无 GIL 调度。
3. `map_batches` / `flat_map` / `map` 四种 call mode。
4. 有状态 actor（`vane.cls`）、批 actor（`vane.cls.batch`）、字节级动态批 + 输出缓冲。
5. `func/cls/cls.batch` 装饰器接上真实后端。
- **验收**：图像处理类 batch UDF 能跑；与 Vane `subprocess_actor` 同口径 benchmark，**目标：Rust 调度开销低于 Vane 的 GIL 调度**。

### Phase 2 —— AI 推理层补齐（2-3 周）
**目标**：AI 能力对齐 Vane（除 vLLM 外先做全）。
1. 结构化输出（return_format / JSON schema），OpenAI/Anthropic/Google 三家。
2. `embed` 表达式形式、长文本分块+加权平均、L2 normalize。
3. SQL 函数 `ai_prompt`/`ai_embed`（经 UDF 算子 lower）。
4. 凭据安全（禁内联 api_key）。
5. 多模态 prompt（bytes/ndarray/PDF）对齐。
- **验收**：`multimodal_structured_outputs` 类示例可跑。

### Phase 3 —— vLLM 集成 + prefix bucketing（2-4 周）
**目标**：拿下 Vane 的吞吐核心，并做成性能亮点。
1. vLLM provider（本地 AsyncLLMEngine 后台线程）+ `prompt_batch` continuous batching。
2. **prefix-aware bucketing 用 Rust 实现**（按公共前缀分桶，`prefix_match_threshold`/`min_bucket_size`/`max_buffer_size` 等参数对齐）。
3. 结构化输出经 vLLM `structured_outputs`。
- **验收**：与 Vane 同模型同数据集跑 prefix-cache 命中率与吞吐，**目标：命中率≥、吞吐>**。

### Phase 4 —— Rust 原生分布式执行（核心，2-4 月）
**目标**：方案 B 的分布式编排，这是超越 Vane 的主战场。
1. **Plan 切分**：把 relation 计划切成 per-partition 子计划（先支持 scan→map→sink 的 embarrassingly-parallel 流水线，即多模态批推理的主场景）。
2. **Worker 抽象**：Ray actor（通过 PyO3 调 Ray）或纯 Rust worker 进程；GPU pinning（`CUDA_VISIBLE_DEVICES`）。
3. **Shuffle**：Arrow Flight（中间结果）+ 对象存储（最终结果），编排全在 Rust。
4. **动态 split 队列 + 背压**：对齐 `ray_max_task_backlog`、`scan_task_size_grouping` 等（这些 env 项已在 Jude 里占位）。
5. **调度器**：Rust async 调度，overlap CPU/GPU/IO。
6. **容错（FTE）**：先做重试，推测执行后置。
- **验收**：跑 Vane 的 `multimodal_inference_benchmarks`（audio/document/image/video 四类）同口径对比，**目标：吞吐 > Vane、Ray Data、Daft**。

### Phase 5 —— 复杂 SQL 分布式 + 生态补齐（长期）
1. 跨 partition 的分布式 join/aggregate（plan 切分 + repartition shuffle）。
2. autoscaling、多节点验证。
3. Polars/Pandas/NumPy 零拷贝、filesystem、replacement scan、DBAPI 完善。
4. 多模态 TensorType（若走方案 B，用 Arrow fixed_shape_tensor 在编排层实现）。

---

## 6. 里程碑与优先级建议

| 优先级 | 内容 | 理由 |
|---|---|---|
| **P0 立刻** | Phase 0（消桩 + lazy relation + 零拷贝 Arrow） | 现在核心是假的，一切建立在此之上 |
| **P1** | Phase 1（本地 UDF 引擎） | 多模态批推理的基础；Rust 无 GIL 是第一个可展示的性能优势 |
| **P1** | Phase 4 的"embarrassingly-parallel 流水线"子集 | Vane benchmark 的主场景就是这个，最快能出"比 Vane 快"的数字 |
| **P2** | Phase 2 + 3（AI + vLLM） | AI 层已最接近，补齐性价比高；vLLM 是吞吐亮点 |
| **P3** | Phase 5（复杂 SQL 分布式） | 最难，非 benchmark 主场景，后置 |

---

## 7. 需要决策的关键问题

1. **是否 fork DuckDB？** —— 强烈建议**否**（方案 B）。fork 只能打平，Rust 编排才能超越。
2. **分布式 worker 用 Ray（PyO3 调）还是纯 Rust？** —— 建议先用 Ray（对齐 Vane 生态 + benchmark），编排逻辑用 Rust。
3. **对齐目标是"API 兼容"还是"能力对等"？** —— 建议能力对等优先，API 尽量兼容 Vane 的 `vane.*` 与 `duckdb.*` 表面。
4. **benchmark 口径** —— 直接复用 Vane 的 `multimodal_inference_benchmarks/`，同数据同硬件，才有说服力。

---

*分析依据：Jude 全量源码 + Vane 的 `src/duckdb_py/`、`duckdb/runners/`、`duckdb/execution/`、`vane/ai/`、`vane/_expression_udf.py` 深度走读。Vane 的 DuckDB fork（`AstroVela/duckdb`）子模块未 checkout，其引擎内算子为接口边界推断。*
