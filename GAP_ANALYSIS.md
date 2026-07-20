# Jude vs Vane — 能力差距诚实盘点 + 实施计划

> 回答"为什么 jude 代码少这么多，到底缺什么"。基于对 Vane 源码的完整走读（distributed_plan_bindings.cpp、fte/*、ray/*、多模态四条 benchmark 管线）。

## 代码量对比（第一方代码）

| | Jude | Vane |
|---|---|---|
| Rust/C++ | 5.2k (Rust) | 38.7k (C++ pybind) |
| Python | 2.3k | 56.2k (`duckdb/` 40k + `vane/` 6k + 其他) |
| runners/调度 | **588 行** | **~28.7k 行** (`duckdb/runners/`) |

**核心真相**：Vane 大出来的量，**几乎全在分布式容错执行（FTE）+ 调度 + 资源管理**这一层，以及它 fork 了整个 DuckDB C++ 引擎。jude 的 588 行 runner 只是"分区 + round-robin dispatch"，而 Vane 的 2.87 万行是一个**真正的分布式查询引擎控制面**。

## 具体缺失的能力（诚实清单）

### A. 分布式执行引擎（最大缺口，Vane ~15k 行 vs jude ~590 行）
jude 现在：`_partition_tables` 切片 + round-robin + bounded backlog。**没有**：

1. **Stage DAG / fragment 图** — Vane 把查询切成以 shuffle 为边界的 stage DAG（`{node_id, input_node_ids, num_partitions, is_sink}`），jude 只有"单层 map"，**不能表达多阶段查询**（scan→join→agg→sink 的真正跨阶段拓扑）。
2. **Shuffle / exchange** — Vane 有 Arrow Flight 流式 shuffle + spooling 两阶段提交 shuffle（`fte_exchange.py` 482行：ExchangeSinkHandle/SourceHandle/OutputSelector）。jude 的 join/agg 是"全量物化到单机再算"，没有真正的分布式 shuffle。
3. **Split assigner** — Vane `fte_split_assigner.py`(630行)：64MiB 目标的自适应装箱、adaptive growth（前小后大）、locality 分组、broadcast 复制、hash 分区。jude 只有均匀切片。
4. **TaskDescriptor / FteSplit / attempt** — Vane 有可增量更新的任务描述符（`fte_descriptor.py` 685行）、split 去重、多 attempt（重试+推测执行）。jude 的 task 就是 `(table, sql)`。
5. **调度器** — Vane `fte_fragment_scheduler.py`(2797) + `driver.py`(3447)：pipelined dispatch（partition 一 ready 就发，不等整个 fragment）、三级优先级（EAGER_SPECULATIVE/STANDARD/SPECULATIVE）、worker placement + reservation。jude 无。
6. **资源管理** — Vane `query_resource_manager.py`(2772) + `cluster_resource_coordinator.py`(690)：DRF 公平调度、多查询并发、autoscaling、CPU/GPU/内存/object-store 四维资源向量。jude 无。
7. **容错** — 重试、推测执行、attempt 选择、worker 失败恢复、两阶段提交 shuffle。jude 无。
8. **流式结果 + 背压** — Vane partition 粒度流式返回 + object-store lease 背压。jude `run_iter_tables` 是 eager collect。

### B. 引擎内算子（Vane fork DuckDB 得到，jude 方案B用 SQL/Arrow 边界替代）
- 原生 `PhysicalRemoteExchangeSink/Source`、scan-split 注入、动态 filter 下推、`TensorType` 逻辑类型。jude 用 Arrow `fixed_shape_tensor` + SQL 边界等价替代（不 fork）。

### C. 多模态（jude 有类型系统，缺 DataSource 流式摄取 + 真实解码管线）
- Vane：`DataSource`/`DataSourceTask` 流式摄取（generator→Arrow RecordBatch→C stream），双层背压（内存水位 + 解码信号量），字节目标批大小；`VideoFrameSource` 完整视频帧解码器。
- jude：有 `TensorType`/`ImageType`/... 类型 + `tensor_array`/`decode_image`，**缺** DataSource 摄取抽象、视频/音频流式读取器、字节预算批处理。

## 为什么之前"看起来对齐了"其实没有
jude 的 relation/UDF/AI/兼容层**接口**对齐得不错，测试也绿。但**分布式引擎是空心的**：`ray.py` 把整表拉到内存、均匀切片、每片跑一遍——这对"embarrassingly parallel 的 map"够用（多模态批推理正好是这种），但**不是一个能跑分布式 join/agg/多阶段查询的引擎**。Vane 的 2.87 万行正是这个引擎。

---

## 实施计划

### Plan 1：真正的分布式执行引擎（对齐 Vane FTE 架构，Python 实现，不 fork）

按 Vane 的 Python 侧架构逐件镜像（C++ native 部分用 SQL/Arrow 边界替代）：

**P1.1 Stage DAG 规划器** — 把 relation 的 LogicalPlan 切成 stage DAG：以 shuffle 边界（join/aggregate/order/distinct/repartition）切分，每个 stage = 独立 SQL + 输入规格。产出 `{query_id, stages:[{stage_id, input_stage_ids, num_partitions, is_sink, sql}], terminal}`。（对应 Vane `collect_execution_stages`）

**P1.2 FteSplit / TaskDescriptor 数据结构** — 直接照抄 Vane 的 dataclass（`fte_types.py`/`fte_descriptor.py`）：FteSplit(source_node_id/sequence_id/kind/data/size_bytes/addresses)、TaskDescriptor(task_id/fragment_id/splits/no_more_splits/...)。

**P1.3 ArbitrarySplitAssigner** — 几乎逐行移植 Vane 的装箱算法：64MiB 目标、adaptive growth(1.26×/64分区)、max_task_split_count、locality 分组、broadcast 复制。加 HashSplitAssigner（hash shuffle）。

**P1.4 WorkerManager + 执行契约** — 一个 driver 类管理 Ray worker actor 池：`submit_tasks`/`query_status`/`wait`/`drop_query`；worker actor `execute_fragment(stage_sql, inputs)`：把输入 Arrow 表/parquet 注册为视图 → 跑 stage SQL → 返回 Arrow（sink stage 写 shuffle 输出）。

**P1.5 Shuffle** — `ray.put` Arrow 表按 (stage, output_partition) 键；hash-partition producer 输出成 N 份；consumer i 读所有 producer 的第 i 份，concat → 注册视图 → 下游 SQL。（对应 flight/spooling exchange 的简化版）

**P1.6 调度 pump** — pipelined dispatch：partition 一 ready 就发；bounded in-flight（背压）；worker placement（locality + 负载）。

**P1.7 流式结果** — driver 按 partition 粒度 yield ObjectRef，lease 背压。

**P1.8 容错（后置）** — attempt 重试、推测执行、exchange attempt 选择。

### Plan 2：多模态数据处理（完备实现）

**P2.1 DataSource / DataSourceTask 摄取抽象** — ABC：`DataSource.schema`(dict) + `.get_tasks()`；`DataSourceTask.execute()` generator yield ~10MB Arrow RecordBatch。`read_datasource(source)` → jude Relation。

**P2.2 内置解码 DataSource** — `VideoFrameSource`（decord/pyav 解帧→fixed_shape_tensor UINT8 (H,W,3)，双层背压+字节批），`ImageFileSource`、`AudioFileSource`（soundfile）、`DocumentSource`（pymupdf 分页）。

**P2.3 多模态类型收口** — 已有 TensorType/Image/Audio/Video/Document；补 array_type(embedding FLOAT[N])、STRUCT[]（检测框）、三处 DuckDB↔Arrow 映射统一到一个 canonical 表。

**P2.4 字节预算批处理** — UDF 输入/输出按字节目标切批（含 tensor 元素大小），对齐 Vane `_payload_output_row_budget_bytes`。已有 max_batch_bytes，补 tensor-aware 估算。

**P2.5 四条参考管线** — 移植 Vane 的 image_classification / audio_transcription / video_object_detection / document_embedding 作为 examples + 端到端测试（用小模型或 stub）。

**P2.6 多模态 prompt（VLM）** — image_columns → base64/magic-byte MIME → provider content parts（OpenAI/Anthropic），return_format 结构化输出。（Phase 2/3 AI 层，之前暂缓）

---

## 优先级
1. **Plan 1（分布式引擎）**——这是最大、最真实的差距，也是"代码量少"的根本原因。先做 P1.1-P1.5（stage DAG + split assigner + shuffle + worker），让 jude 能真正分布式执行多阶段查询。
2. **Plan 2（多模态）**——P2.1-P2.2（DataSource + 视频/图像/音频读取器）是多模态摄取的地基。
3. 容错、资源管理、VLM 后置。
