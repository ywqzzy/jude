# jude 对比 Vane 的能力差距（不含 AI/大模型推理）

> 结论一句话：jude 把 Vane 分布式引擎的“数据结构”翻译成了 Rust，但没有翻译它的“运行时”。
> jude 自己的代码里就写着这句话（`src/dist/stage.rs:11-15`）：
> “这里只生成了执行计划；真正能跑任意 DAG 的流式执行器是后续工作，目前只有两阶段聚合、哈希 join 是专门写死的代码。”
>
> 而 Vane 有一整套 Trino 风格的“容错执行引擎（FTE）”：`duckdb/runners/fte/`（6674 行）+ `duckdb/runners/ray/`（18476 行）。
> jude 的 `src/dist/` 只有 1448 行，且大部分是结构体定义。
>
> **所以最值钱的差距，全都在“容错 / 分布式执行”这一块。**

> 本文由只读调研 agent 生成（2026-07-19），作为剩余工作的路线图基线。所有 `文件:行号` 证据均可核对。

---

## 一、分布式执行 / 容错（最大的坑，价值最高）

### 1a. Spooling Exchange（落盘交换）—— 价值最高
**是什么**：每个 shuffle 阶段的输出，Vane 会按“(输出分区, 尝试次数)”落盘到 spool 目录，配一个 `manifest.json` 清单，加 `committed`/`aborted` 标记文件，再用一个“选择器”挑出每个分区的“胜出尝试”，并清理掉没被选中的尝试。

- 证据：`duckdb/runners/fte/fte_exchange.py:347` 的 `SpoolingExchangeManager`；标记文件 `:348-350`；提交/中止逻辑 `:397-422`；清理未选中尝试 `:441`。
- 胜出尝试选择器：`FteExchangeSourceOutputSelector`（`:219`），`try_mark_final`（`:256`）。

**为什么重要**：这是容错的地基。有了它，下游 task 失败后可以直接“重新读上游落盘的结果”，而不用把整条 DAG 重算一遍。
jude 现在的哈希 join / 两阶段聚合是通过 Ray 的对象引用传中间数据，没有“可持久化、可重读、带尝试选择”的交换层。
jude 里 `grep spooling` = 0。

- **工作量**：大（是整个容错的骨架）。**价值**：最高。

### 1b. 基于“尝试次数”的 task 重试
- `fte_execution.py:115` 有 `max_attempts=4` / `remaining_attempts`；`task_failed(...)`（`:923`）会先中止当前 sink 尝试，然后返回 `ReadyTask(reason="retry")` 并起一个新尝试（`_maybe_create_attempt`）。
- 失败分类：`fte_failures.py:79-121` 会区分“用户错误 / 致命错误 / 内存溢出 / 可重试”，不可重试的错误直接快速失败。
- jude 只有结构体里一个 `attempt_id: u32` 字段（`src/dist/fte.rs:36`），没有重试循环。
- **工作量**：中（依赖 1a）。**价值**：高。

### 1c. 推测执行（Speculative Execution，治长尾/慢节点）
- 执行类别 STANDARD / SPECULATIVE / EAGER_SPECULATIVE：`fte_state.py:31-65`。
- 起推测尝试 + 撤销输掉的尝试：`fte_execution.py:972` 的 `revoke_speculative_attempts`，`RevokedAttempt`（`fte_attempts.py:48`）；调度器判断 `fte_fragment_scheduler.py:1641`。
- jude 运行时没有。
- **工作量**：中。**价值**：中高。

### 1d. 节点/Worker 挂掉后的恢复
- `fragment_worker_failures.py:19` 的 `mark_fte_worker_failed_for_event` → `scheduler.record_worker_failure(...)` → `_mark_fte_worker_failed`，会把死掉节点上跑过的所有 task 重新调度，还有 `arm_retry_delay` 做重试延迟。
- jude 里 `grep heartbeat` = 0，`grep worker_lost` = 0（完全没有）。
- **工作量**：中大。**价值**：高（大集群上“容错”这个卖点的核心）。

### 1e. 内存自适应重试（OOM 后加内存重来）
- `fte_execution.py:938` 专门处理 `_is_memory_failure(error)`；task 内存预留 `task_memory_bytes`（`:435/:482`）；`FteWorkerAdmissionConfig.task_memory_bytes`（`fte_config.py:37`）。OOM 的分区会被重新规划，而不是原样在同样内存下瞎重试。
- jude 有资源准入 + bin-pack，但没有“OOM 触发加内存重试”。
- **工作量**：小中。**价值**：中。

### 1f. 抗数据倾斜的哈希子分区
- Vane 的 `HashSplitAssigner` 能把一个源分区拆到多个 task 分区：`HashTaskPartition.sub_partition_count` / `split_by_source`（`fte_split_assigner.py:434-461`）。
- jude 的 `HashSplitAssigner` 只是简单取模路由，没有子分区（`src/dist/split_assigner.rs:104-143`）。
- **工作量**：小中。**价值**：中（热点 key 的 join/聚合）。

### 1g. 动态过滤下推（Dynamic Filter Pushdown）
- join 的构建端会算出 `dynamic_filter_domains`，运行时合并进探测端 scan 的请求：`fte_worker_runtime.py:761` 的 `_initial_dynamic_filter_domains`，合并 `:879-884`；传播 `duckdb/runners/ray/worker.py:656,898`。
- jude 里 `grep dynamic_filter` = 0。
- **工作量**：中。**价值**：中高（选择性强的 join 能大幅提速，且不依赖 FTE 也能做）。

### 1h. 运行中动态追加 split（增量输入）
- `dynamic_inputs.py` 会把 `scan_task:` / `exchange_source_task:` 的 split 边算边喂给正在跑的 task。
- jude 有描述符版本号（`fte.rs:107 descriptor_version`），但没有运行时动态喂 split 的管道。**部分实现。** 价值：中。

---

## 二、多模态（非模型部分）

### 2a. 通用流式 DataSource 接口 —— 价值高（同时也是 IO 的坑）
- `duckdb/datasource/__init__.py:38-85`：`DataSourceTask.execute()` 是个生成器，每次 yield 约 10MB 的 `RecordBatch`；`read_datasource`（`:206`）走 C++ 的 `datasource_scan` 表函数，每个 task 是独立的 ArrowArrayStream，由 pipeline 线程并行拉取，天然带背压。还支持 **tensor 类型的 schema** → `fixed_shape_tensor`（`:117-125`、`:180-183`）。
- jude 的 `jude/sources` 里 `FileSource.to_arrow` 是**一次性把所有字节读进一张 Arrow 表**（`python/jude/sources/__init__.py:133`）：不能流式、不能自定义源、没有 pipeline 并行、没有背压。jude `grep from_datasource/DataSourceTask` = 0。
- **工作量**：中（需要一个从 Python 生成器拉数据的 scan 表函数）。**价值**：高（支持无界/流式接入 + 用户自定义数据源）。

### 2b. 流式视频抽帧读取器
- `duckdb/datasource/video_reader.py` 的 `VideoFrameSource`：用 decord 解码，把帧**流式**输出成 fixed-shape tensor 批次，可配软块字节目标（128MiB）、resize 线程池、每分区字节预算。
- jude 只有批量的 `decode_video_batch`（PyAV），是把内存里一整列视频字节解码（`python/jude/multimodal/decoders.py:185`），没有“可横向扩展的流式帧源”。**部分实现。** 价值：中。

**不算差距**：jude 已经有 tensor 类型（`jude.types.tensor_array`）和 图片/音频/视频/文档 解码器（`multimodal/decoders.py:42/115/185/267`）。

---

## 三、UDF 引擎

### 3a. UDF Actor 的容错
- `duckdb/execution/udf_ray_config.py:12-13`：`MAX_ACTOR_RESTARTS=4`、`MAX_ACTOR_TASK_RETRIES=4`，作用在 UDF actor 池上。
- jude 的 actor 池（`python/jude/execution/udf_ray.py`）没有重启/任务重试参数。
- **工作量**：小。**价值**：中。

### 3b. Actor 预热（eager warm-up）
- `udf_ray_config.py:56` 的 `eager_actor_warm_up_enabled`（环境变量 `VANE_UDF_EAGER_WARM_UP` 或 payload `eager_warm_up`）。jude 有常驻池但没有预热开关。**价值**：低中。

### 3c. 流式 UDF 结果收集协议
- `udf_ray_config.py:70` 的 `stream_output_enabled`；专门的 `udf_stream_result_collector.py`（34KB）+ `ray_stream_adapter.py`。
- jude 已有基于 Ray 生成器的 sub-batch 流式输出，所以这里是**部分实现**——Vane 的收集协议更完整。价值：低中。

### 3d. UDF 级别的准入 + actor 线程策略
- `udf_admission.py`、`udf_task_admission.py`、`udf_threading.py`。jude 有跨查询 bin-pack/准入；显式的 actor **线程策略**（`udf_threading.py`）可能是差距点。**部分实现。**

**诚实说明**：async UDF 不是真差距——Vane 的 async 只在 AI/vLLM provider 里（已排除），两边都没有“通用异步标量 UDF”。

---

## 四、DataFrame / SQL / 关系型接口

### 4a. PySpark 兼容层 —— 价值中高，范围清晰
Vane 的 `duckdb/experimental/spark/sql/` 是接近完整的 PySpark 移植（共 10883 行）：
- `functions.py` = 6217 行（约 197 个函数）、`column.py` = 364 行（Column 表达式代数 + when/otherwise）、`group.py` = 425 行（`GroupedData.agg`/`pivot`）、`readwriter.py` = 435 行（读写器）、`types.py` = 1324 行、`session.py` = 297 行，还有 `catalog.py`、`conf.py`、`streaming.py`、`udf.py`、`type_utils.py`。
- jude 只有薄薄一层壳：`python/jude/experimental/spark/sql/dataframe.py`（166 行，约 35 个方法）+ `session.py`（136 行）。**缺**：整个 `functions` 模块、`Column` 类、`GroupedData`/`agg`/`pivot`、读写器、`types` 模块、`Catalog`、`conf`、`streaming`。
- **工作量**：大但机械。**价值**：中高（Spark 用户迁移故事）。

**不算差距**：jude 已有 sample（`LogicalPlan::Sample`）、union/intersect/except（`src/relation.rs:950-972`）、窗口函数（每个聚合都带 `window_spec`，`relation.rs:1007+`）。PIVOT/UNPIVOT：两边都没暴露 Python 方法，都靠 DuckDB SQL 的 `PIVOT`，算打平；唯一的 pivot 差距是 Spark 的 `GroupedData.pivot`（已并入 4a）。

---

## 五、IO / 存储 / 格式

- **5a. 流式 DataSource** —— 同 2a，是 IO 的主要坑。价值高。
- **5b. ADBC 驱动** —— `adbc_driver_duckdb/dbapi.py`，让外部 Arrow/BI 工具通过 ADBC 连进来。jude 没有。价值：低（小众）。
- **5c. Polars 惰性 IO 桥** —— `duckdb/polars_io.py`（311 行）扫描/流式桥。jude 只有 `relation.rs`/`arrow_ffi.rs` 里零散的 polars 转换。**部分实现。** 价值：低中。

**不算差距**：两边都基于 DuckDB，原生 parquet/csv/json/httpfs 是打平的；jude 已有 Lance/Iceberg/Hive/Daft。

---

## 六、可观测性 / 运维 / 配置

### 6a. 算子级 / pipeline 级进度上报
- `duckdb/runners/progress.py`（640+ 行）：`build_progress_snapshot`（`:407`）、`format_progress_snapshot`（`:639`）、`LocalProgressSnapshotStore`（`:607`）、pipeline 拓扑校验（`:54`），带每 pipeline/每算子的行数+字节吞吐、耗时、task 计数，能渲染成仪表盘。
- jude 基本没有查询执行的进度子系统（它的 `progress` 命中主要是 AI 指标 + 一个 pipeline 文件）。
- **工作量**：中。**价值**：中（用户体验/运维）。

### 6b. 配置/调参面
- Vane 约有 155 个环境变量开关，jude 约 42 个。这主要是上面 FTE/UDF 机制的“副产品”，不是独立特性。价值：低。

---

## 按价值排序（目标：除 AI 推理外全面超过 Vane）

1. **Spooling exchange + 尝试重试（1a + 1b）** —— 容错地基，解锁其它所有容错能力。*工作量大，价值最高。*
2. **节点丢失恢复（1d）** —— 容错这个卖点的核心。*中大 / 高。*
3. **通用流式 DataSource（2a / 5a）** —— 无界接入 + 自定义源 + 流式视频的基础。*中 / 高。*
4. **动态过滤下推（1g）** —— join 大幅提速，且不依赖 FTE。*中 / 中高。*
5. **推测执行（1c）** —— 治慢节点长尾。*中 / 中高。*
6. **PySpark 兼容层（4a）** —— 面大、迁移故事好，但机械。*大 / 中高。*
7. **进度/可观测性（6a）** —— 体验/运维。*中 / 中。*
8. **较小项**：内存自适应重试（1e）、抗倾斜子分区（1f）、UDF actor 重试/预热（3a/3b）、流式视频读取器（2b）、ADBC/Polars 惰性（5b/5c）。*各自小到中。*

---

## 明确“不是差距”的（jude 已有 / 两边都没 / 属于 AI）

分布式 scan/map/filter/聚合/join/排序/去重/top-k、自动路由的 `collect()`、Ray 生成器 sub-batch 流式、资源准入 + 跨查询 bin-pack、常驻 actor 池 + 子进程 UDF 池、五种 UDF 形态、每 UDF 的 GPU 分配（`src/relation.rs:490,2024`）、tensor 类型 + 四种解码器、Lance/Iceberg/Hive/Daft、sample/集合运算/窗口函数，以及通用 async UDF（两边都没有；Vane 的 async 只在 AI 里）。
