# jude vs Vane：功能与系统能力差距分析 + 优化计划

状态：分析。目的是把 jude 和 Vane 的差距讲清楚——哪些是真差距、哪些其实我们已经领先、哪些是
"看起来差但代价不值得追"。所有结论都对着两边的**实际代码**核过，不是拍脑袋；Vane 的 C++ 引擎核心
（`external/duckdb` 子模块）在它的 checkout 里没拉下来，涉及引擎内部算子的部分我标注为"据 pybind 绑定
与 Python 控制面推断"，不当成实锤。

先把最根本的一条摆前面，因为后面所有差距都从它派生：

> **Vane 是 DuckDB 的 fork（改了 C++ 引擎），jude 是 stock DuckDB + Rust 外壳（不 fork）。**

Vane 把分布式、vLLM 算子、tensor 逻辑类型直接焊进了引擎；jude 坚持不 fork，靠"物化边界 + 分区级
编排"在引擎外面拼这些能力。这决定了：Vane 能做我们做不了的引擎内算子（流式 Flight shuffle、引擎内
两阶段聚合、prefix-cache 路由的 vLLM 算子）；而 jude 白拿 DuckDB 的每一次版本升级、不用维护一个
会持续 diverge 的 fork。差距要在这个前提下看，不是"我们缺功能"这么简单。

---

## 一、系统能力差距（这是真差距，也是 Vane 的护城河）

Vane 的分布式控制面是它投入最重的地方：`duckdb/runners/` 约 **28.7K 行 Python** + `src/duckdb_py/ray/`
约 **11.2K 行 C++ 绑定** + 没拉下来的 C++ 引擎核心。它有**两条**执行路径共用同一套物理计划/exchange 底座：
一条是 native 流式（Arrow Flight 管线化 shuffle），一条是 FTE（Trino 式容错，落盘物化）。

jude 这边：Rust 的 `WorkerManager`（`src/dist/worker_manager.rs`）+ `StagePlanner`
（`src/dist/stage.rs`）+ 三种 split assigner（`src/dist/split_assigner.rs`），Python 只是薄薄的 Ray RPC
转发（`python/jude/runners/ray.py` + `_ray_shim.py`）。能力覆盖到：分区级 scan→map→sink、两阶段聚合
（`distributed_aggregate`）、hash-shuffle join（`distributed_join`）、locality 感知分配、size 分组、有界
backlog 背压。**这是"简单 80%"，Vane 的重投入全在剩下的 20%。**

具体缺什么，按"值不值得补"排：

**1. 容错执行（FTE）—— 最大的真差距。** Vane 有一整套 Trino 式容错：`FteTaskAttemptId`（query.fragment.
partition.attempt 四级标识）、每分区有界重试（默认 4 次）+ 指数退避（10s→60s，×2）、失败分类
（USER/INTERNAL/EXTERNAL，用户错和致命错不重试）、`SpoolingExchangeManager` 把每次 attempt 的输出落盘成
Arrow 文件 + `manifest.json` + committed/aborted 标记、下游只读被选中的已提交 attempt、单 worker 挂了会
按 `host#index` 扩散成整机恢复、还有 speculative execution（STANDARD/SPECULATIVE/EAGER_SPECULATIVE）。
jude 现在是"某个分区挂了就整个查询失败重来"。**评估**：这是 Vane 真正的护城河，也是最难补的——补齐需要
落盘 exchange + attempt 生命周期 + 调度器改造，是数千行的活。我们已经把词汇（`FteSplit`、`TaskDescriptor`、
`ArbitrarySplitAssigner`/`HashSplitAssigner`）ported 进 Rust（task #20/#21 已完成），骨架在，但**没接进
执行路径**。

**2. 流式多 fragment 运行时 + Arrow Flight shuffle。** Vane 的 native 路径是真流式：`PhysicalRemoteExchange
Sink/Source` 通过 Flight 边算边传，fragment 之间管线化，还能在 fragment 运行中通过 `FteSplitQueue` 喂
动态输入。jude 是 Model B 物化——每个 stage 跑完、结果落 Ray object store、下一个 stage 再读。**评估**：
happy-path 上物化模型反而更简单、够快（多模态批推理这类 scan→map→sink 本来就没有跨 stage 流水线可言）；
流式的收益主要在深 join/聚合链路和低延迟场景。**优先级中等**，不是多模态负载的瓶颈。

**3. 资源准入控制。** Vane 的 `QueryResourceManager`（2772 行）按 4 维 `ResourceVector`（cpu/gpu/heap/
object_store）做准入、output-block 租约背压、下游预留、软硬限。jude 只有"N 个并发任务 + 有界 backlog"。
**评估**：GPU 批推理场景下，按 GPU 显存/object store 容量做准入是有实际价值的（避免 OOM、避免 object
store 撑爆），**优先级中高**，且可以增量做（先加 GPU/显存维度）。

**4. 跨查询集群调度。** Vane 的 `cluster_resource_coordinator.py` 读 Ray 节点容量、把 actor/task bundle
bin-pack 到节点上。jude 没有跨查询协调（依赖 Ray 自己的调度）。**评估**：单查询场景用不上，多租户才需要，
**优先级低**。

**5. 运行时动态过滤（join 下推到 scan）。** Vane 把 join 的 dynamic filter domain 下推到 scan task。jude
靠把 SQL filter 尽量下推进 `iceberg_scan`/DuckDB，但没有分布式 join 的运行时过滤。**优先级低**（DuckDB
单机内部已有运行时过滤，跨 worker 的收益有限）。

---

## 二、功能层面差距

### UDF —— Vane 最成熟的面，但 jude 差距没想象中大

Vane 的 UDF 是它除 AI 外最厚的子系统：scalar/vectorized(Arrow-batch)/batch/flat_map/table(generator)/
class-stateful 六种 flavor，subprocess + Ray actor/task 四种 out-of-process 后端（GIL 绕过），外加一堆细粒度
batching/背压/GPU 资源旋钮（cpus/gpus/memory_bytes/actor_number）。

jude 已经有的（核过代码）：
- scalar + vectorized 标量 UDF（`create_function`，`VScalar`/`VArrowScalar`，null/exception/side_effects
  旋钮，`src/expression_udf/registration.rs`）——task #28/#30 已完成；
- `map_batches`（`src/relation.rs`，Arrow-batch）+ 字节级动态 batching + scalar map call mode
  （最近提交 c6e7f46）；
- **class-based / stateful UDF**：`jude.cls` / `jude.cls.batch`（`python/jude/expression_udf.py`），带
  `actor_number`、`gpus` 旋钮——这块我们其实**已经有了**；
- subprocess 池 GIL 绕过（conftest 里有跨测试拆池，提交 3062029）。

真正还缺的：
- **table / generator UDF**（一行进、多行出的用户表函数）；
- **flat_map**（Spark 式 explode 型 UDF）——注意多模态那边我们有 `explode_multimodal`，但不是通用 flat_map；
- **aggregate UDF**（用户自定义聚合）——两边其实都没有通用 Python 聚合 UDF；
- Ray **actor 池**后端（jude 现在是 subprocess 池 + Ray task，缺常驻 actor 池复用模型/GPU 上下文）。

**评估**：table UDF 和 flat_map 是**中优先级**、可增量补的；Ray actor 池对"每个 worker 常驻一个大模型"
的批推理场景**中高优先级**（现在每个 task 重新加载模型代价高）。

### AI / LLM —— Vane 领先，但 jude 有 Rust 原生底座

Vane：5 个 provider（OpenAI/Anthropic/Google/HuggingFace/vLLM）、把 `ai_prompt`/`ai_embed`/`vllm` 注册成
**原生 SQL 函数**、prefix-cache 感知的 vLLM 批推理路由（`PrefixRouterActor`、bucket）、结构化输出、token
计量、多模态 prompt（图片/PDF 进 prompt）。vLLM 算子本身是引擎内 C++ 算子。

jude（核过 `src/ai/`）：4 个 Rust 原生 provider（openai/anthropic/google/transformers，reqwest 直连）+
vLLM options（`VllmProviderOptions`/`VllmPromptOptions`，但**没有独立 vllm provider 实现**，目前是 options
占位）、`prompt`/`embed`/`embed_text`/`classify_text`/token 计量（`jude.ai`）、retry。

差距：
- **vLLM 深度集成**：Vane 有 prefix 路由 + 连续批处理 + 引擎内算子；jude 只有 options 壳。**这是 AI 面最大
  的真差距**，但也最难——引擎内 vLLM 算子我们不 fork 就做不了，只能在编排层做 prefix 分桶路由（可行、
  中高优先级）。
- **SQL 级 AI 函数**：Vane 的 `ai_prompt`/`ai_embed` 是原生 SQL 标量函数；jude 的 AI 走 Python API。可以用
  jude 已有的 scalar UDF 机制把 `ai_prompt`/`ai_embed` 注册成 SQL 函数——**低成本、中优先级**。
- 多模态 prompt（图片进 prompt）：Vane 的 OpenAI provider 支持；jude 待补——**中优先级**，且和下面多模态
  领先面能连起来。

### 多模态 —— **jude 领先**

反直觉但核过代码属实：**Vane 没有表达式命名空间式的多模态 API**（没有 `.image.decode()`/`.url.download()`
——那是 Daft 的 API，只出现在 Vane 的对比 benchmark 里）。Vane 的多模态是"在 `map_batches` 里自己调 PIL/
torchvision/decord/pymupdf"，加一个 decord 视频帧 DataSource + tensor 类型。

jude 这边有完整的表达式层：`.image`（decode/resize/crop/to_mode/to_tensor/encode）、`.url`（download）、
`.audio` 命名空间（`python/jude/_mm_expr.py`），Rust 侧 `MmOp` 用 `image` crate 做 kernel
（`src/multimodal/mod.rs`）。**这块是我们的相对优势**，应该继续做厚（补 audio/video 解码 kernel、把多模态
和 AI prompt 打通），而不是去追 Vane 其实没有的东西。

### 存储 / 表格式 —— **jude 领先**

Vane：只有 Parquet/CSV 读写，**没有** Iceberg/Delta/Paimon/Hudi/Lance，没有 catalog 集成。

jude：Iceberg 读（`iceberg_scan` + 快照/时间旅行）+ 单机写（Rust `COPY TO` 分区写 + pyiceberg 提交）+
**分布式写**（`RayRunner.distributed_write_iceberg`，worker 并行写数据文件、driver 一次提交），见
`docs/storage_design.zh.md`。**这是 jude 明确领先且符合"分布式写入引擎"定位的地方。** Paimon 因本机
`pypaimon` 依赖冲突（要 pyarrow<20，我们用 25）暂时 blocked，诚实标注。

### 索引 / 向量检索 / 倒排 —— 两边都没有（打平）

Vane：**完全没有**向量检索/ANN/全文倒排/二级索引（核过，grep hnsw/faiss/ivf/fts 无命中）。jude：有设计文档
（`docs/index_design.zh.md`）但未实现。**结论**：这不是"和 Vane 的差距"，是共同空白。要不要做取决于 jude 自己
的产品定位（做 AI 数据引擎的话，向量检索比追 Vane 的 FTE 更有差异化价值）。

### DataFrame / SQL 保真度 —— 收尾中

Vane 继承 DuckDB 全套关系代数 + window 函数 + Spark 兼容 DataFrame API。jude 已对齐了绝大部分
（聚合 + window 全套，提交 20facec）。剩余 41 个 ported 测试 gap 集中在深保真度尾巴：`test_relation`（15，
relation 对象的 view/materialize/update/serialize/close 语义）、`test_all_types`（14，主要是 pandas/numpy
roundtrip 的 enum/union/masked-array 保真）、`test_replacement_scan`（5）、`test_read_csv`（4，sniffing/
encoding 选项）、`test_rapi_aggregations`（3，value_counts/string_agg/list 结果形状）。**都是长尾保真度，
不是能力缺失**，逐个啃即可。本 session 已把 gap 从 69 降到 41。

---

## 三、优化计划（按"价值/代价"排序）

原则：**先做我们已经领先、能拉开差距的（多模态 + 存储 + AI 编排），再补对多模态批推理负载真正有痛感的
系统能力（GPU 资源准入 + actor 池），FTE 这种重投入放最后且分期做。** 不去追 Vane 其实也没有的东西
（向量检索按 jude 自身定位单独决策，不算"追平 Vane"）。

**P0 — 巩固领先面 + 低垂果实（本周期）**
1. 把 `ai_prompt` / `ai_embed` 用现有 scalar UDF 机制注册成 SQL 函数（低成本，抹平 Vane 的 SQL 级 AI 差距）。
2. 多模态 prompt 打通：图片列进 `jude.ai.prompt`（把领先的多模态层和 AI 层连起来）。
3. 继续啃 ported 测试长尾（41→目标 <25）：优先 `test_read_csv`（选项对齐，性价比高）和
   `test_rapi_aggregations`（结果形状）。

**P1 — 对批推理负载有痛感的系统能力（下个周期）**
4. **Ray actor 池后端**：worker 常驻 actor 复用模型/GPU 上下文，避免每个 task 重载模型。对"大模型批推理"
   场景收益最大。接到现有 `WorkerManager` 调度里。
5. **GPU / 显存维度的资源准入**：给 `WorkerManager` 加 GPU/object-store 容量维度的准入（增量做，不必一步到
   位做 Vane 的 4 维 ResourceVector），防 OOM / object store 撑爆。
6. **table UDF + flat_map**：补齐通用一进多出 UDF（多模态 explode 已有，泛化成通用能力）。

**P2 — vLLM 编排层集成（较难，分期）**
7. 在**编排层**（不 fork 引擎）实现 prefix-cache 感知的批分桶路由，配合 vLLM endpoint；先做真正的 vllm
   provider 实现（替换现在的 options 占位）。

**P3 — FTE 容错执行（最重，分期，先落盘 exchange）**
8. 复用已 ported 的 `FteSplit`/assigner 骨架，先做**落盘物化 exchange + attempt 生命周期**（committed/
   aborted manifest），让"单分区失败只重跑该分区"先跑通；speculative execution / 整机恢复 / 资源租约
   放更后面。这是数千行的活，价值高但不紧急（多模态批推理的容错可先靠 Ray task 级重试兜底）。

**不做 / 单独决策**
- 跨查询集群调度（P-低，多租户才需要）。
- 向量检索 / 倒排：不算"追 Vane"（它也没有）；若做，是 jude 自身差异化，按产品定位单独立项。
- Paimon 写：本机依赖 blocked，待 pyarrow 版本约束解决或有可测环境再做。

---

## 一句话总结

jude 不是"处处落后 Vane"：**多模态表达式 API 和 Iceberg 分布式写我们领先，索引两边打平，SQL 保真度在
收尾。** Vane 真正甩开我们的是**分布式系统能力**（FTE 容错、流式 Flight shuffle、资源准入）和 **vLLM 引擎内
深度集成**——前者难但可分期在编排层补，后者受限于我们不 fork 的原则、只能在编排层逼近。计划因此把资源
优先投在"巩固领先 + 补批推理痛点"，把 Vane 的重护城河（FTE）放到分期、增量的轨道上，而不是一头扎进去
和一个 fork 了引擎的对手拼引擎内能力。
