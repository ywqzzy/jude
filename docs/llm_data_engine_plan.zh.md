# jude 作为「大模型数据处理引擎」的能力规划与设计文档

> 状态：规划文档（2026-07-19）。基于对 Vane（`/Users/zzywq/code/vane`）与行业专用框架（NeMo Curator / DataTrove / Dolma / Daft / Mosaic Streaming）的调研。
> 定位：本文覆盖「给大模型做数据准备/处理」的能力，**不含 LLM 推理本身**（推理是另一条线）。
> 配套文档：能力差距（对标 Vane 全量）见 [`gap_vane_vs_jude.zh.md`](gap_vane_vs_jude.zh.md)。

---

## 0. 一个必须先纠正的认知

调研（走读 Vane 全量 `vane/` `duckdb/` `src/duckdb_py/`）得到一个关键结论：

**Vane 里那些「大模型数据处理」能力——MinHash 去重、语言过滤、句子分块、Common Crawl/WARC 处理——全部是 `examples/` 脚本，不是引擎内建算子。**

- `examples/minhash_dedupe.py` 头注明写 "adapts Daft's Common Crawl MinHash tutorial"，MinHash/LSH/UnionFind 是写在 example 里的纯 Python `map_batches` UDF。
- `examples/common_crawl.py`、`examples/llms_red_pajamas.py` 同理——语言过滤是 example 里的 SQL `where`，语义匹配是手写 numpy。
- 对 `vane/`+`duckdb/`（排除 test/example）grep `dedup|minhash|quality_filter|language|pii|gopher` **0 命中**。

**推论**：在「LLM 数据治理算子」这个细分赛道，**Vane 也没做成引擎能力**。jude 若把去重/质量过滤/分块做成**引擎内建、分布式、Rust 加速的一等算子**，会在这个赛道**直接超过 Vane**。所以本规划对标的主要是**行业专用框架**，而非 Vane。

---

## 1. jude 已具备（不重复建设）

| 能力 | 证据 |
|---|---|
| 文本 embedding（批） | `src/ai/functions.rs` `embed_text` + `EmbedTextBatch` |
| 零样本分类 / LLM prompt（批） | `src/ai/functions.rs` `classify_text` / `prompt_relation`；`relation.prompt()` |
| 向量索引 + ANN 检索 | `python/jude/_lance.py` `create_vector_index`(IVF_PQ/HNSW)、`vector_search`、scalar index |
| 多模态解码 | `multimodal/decoders.py`：image/audio/video(逐帧)/document(逐页) |
| 多模态摄取源 | `python/jude/sources/`：Image/Audio/VideoFrame/Document Source |
| 存储 | Lance/Iceberg 分布式读写 + git-like 版本、Hive、Daft bridge、catalog |
| 关系算子 | sample / order / limit / distinct / union|intersect|except / window |
| 分布式 | scan/map/filter/agg/join/sort/distinct/topk + **通用流式 stage-DAG 执行器** + 资源准入 + bin-pack |
| 可观测性 | Rust MetricsRegistry + HTTP endpoint |
| token 用量统计 | `src/ai/metrics.rs` |

**对标 Vane 反而领先**：Iceberg/Lance 分布式写、多模态表达式命名空间、向量索引、无引擎 fork。

---

## 2. 能力差距清单（按对「大模型数据处理」的价值排序）

### 🔴 第一梯队 —— LLM 训练数据准备的地基

#### C1. 模糊去重 Fuzzy Dedup（MinHash + LSH，分布式）
- **解决什么**：训练语料 30–70% 是近重复，去重直接决定模型质量与训练效率，是 LLM 数据处理**最核心**一步。
- **Vane**：❌ 仅 example（单机 Python UnionFind，不可扩展）。**行业**：NeMo Curator（GPU MinHash-LSH + 连通分量）、DataTrove、Dolma、Daft `minhash`。
- **jude 现状**：❌ 无。但有分布式 hash-shuffle + join + Rust，**天然适合**做分布式 MinHash 签名 + LSH banding 分桶 + 分布式连通分量。
- **难度**：中。**价值**：极高。

#### C2. 精确去重 Exact Dedup（文档级 / URL 级 / 行级）
- **解决什么**：完全重复文档/URL/行删除，去重第一道也最便宜的一道。
- **Vane**：❌（可用 SQL distinct，但无面向文档 hash 的原语）。**行业**：全框架标配。
- **jude 现状**：🟡 有分布式 `distinct`，缺「内容规范化 + hash + 跨分片全局去重」封装。
- **难度**：小（`sha256(normalize(text))` 列 + 分布式 distinct，基建已有）。**价值**：高。

#### C3. 质量过滤 Quality Filtering（启发式规则集 + 分类器）
- **解决什么**：过滤垃圾文本（符号比、平均词长、重复行/n-gram、停用词占比、困惑度）。Gopher/C4/FineWeb 规则集是行业共识。
- **Vane**：❌（example 里只有简单 SQL where）。**行业**：NeMo Curator `heuristic_filter`+fastText、DataTrove `GopherQualityFilter`/`C4QualityFilter`、Dolma taggers。
- **jude 现状**：❌ 无。
- **难度**：中（一批 Rust/SQL 标量函数 + 可选分类器 UDF）。**价值**：极高（与去重并列两大支柱）。

#### C4. 语言识别 Language ID
- **解决什么**：按语言拆分/过滤（多语或纯英数据集必需）。
- **Vane**：❌。**行业**：fastText lid.176 事实标准。
- **jude 现状**：❌。
- **难度**：小（一个 fastText/lingua 的 vectorized UDF，UDF 框架现成）。**价值**：高。

#### C5. 文本分块 Chunking（字符 / 递归 / token 级）
- **解决什么**：RAG 索引、长文本 embedding 的前置。jude 连 Vane 那种基础字符分块都没有。
- **Vane**：🟡 有 `chunk_text`（字符级+overlap）+ `embed_text` 内置分块聚合。**行业**：LangChain/LlamaIndex recursive/semantic、tiktoken token 分块。
- **jude 现状**：❌ `embed_text` 签名无 `max_chunk_chars`，无 `chunk_text` API。
- **难度**：小（字符/递归）→中（语义/token）。**价值**：高（RAG/embedding 管线）。

#### C6. 结构化输出 Structured Output（JSON schema / Pydantic guided decoding）
- **解决什么**：让 LLM 抽取/标注返回严格结构（数据标注、合成数据、信息抽取必需）。
- **Vane**：✅ `prompt(..., return_format=PydanticModel)` + vLLM guided decoding。
- **jude 现状**：❌ `prompt` 无 `return_format` 参数。**这是 jude 对标 Vane 真正落后的 2 项之一。**
- **难度**：中（API 层 + provider response_format 透传 + 解析校验）。**价值**：高。

### 🟠 第二梯队 —— 专用数据引擎差异化

#### C7. 语义去重 Semantic Dedup（embedding + 聚类，SemDeDup）
- **解决什么**：删「意思重复但字面不同」的样本，比 MinHash 更狠，SOTA 数据集在用。
- **Vane**：❌。**行业**：NeMo Curator `SemanticDeduplication`（embed→kmeans→簇内剔除）。
- **jude 现状**：🟡 **零件齐全没组装**：embed_text + Lance 向量索引/检索 + 分布式聚合。缺编排。**jude 因有 Lance 向量栈，做这个比 Vane 更顺——潜在护城河。**
- **难度**：中。**价值**：高。

#### C8. 训练格式写出（WebDataset .tar / Mosaic MDS / 尺寸对齐 sharded parquet）
- **解决什么**：训练框架（PyTorch/Mosaic/Megatron）要分片、可流式、大小对齐的数据集。这是「处理→训练」的最后一公里。
- **Vane**：❌（只有 Parquet/CSV）。**行业**：Mosaic MDS、WebDataset、HF datasets、Lance。
- **jude 现状**：🟡 有 parquet/分区 parquet/Lance 写，缺 WebDataset/MDS/尺寸对齐分片。**jude 已有分布式写引擎，补这个是自然延伸——差异化优势。**
- **难度**：中。**价值**：高。

#### C9. 数据集混合与全局 shuffle（Blending / global shuffle / 分层 & 蓄水池采样）
- **解决什么**：多来源按权重混合（50% web+30% code+20% books）、全局打散、分层采样——训练数据配比核心。
- **Vane**：❌（有 sample/order，无加权混合、无全局 shuffle）。**行业**：NeMo Curator `blend_datasets`/`shuffle`、Ray Data `random_shuffle`。
- **jude 现状**：🟡 有 sample/order，缺加权混合/分布式全局 shuffle/分层采样。有 shuffle 基建可复用。
- **难度**：小（加权混合）→中（分布式全局 shuffle）。**价值**：高。

#### C10. PII 检测与脱敏
- **解决什么**：邮箱/电话/SSN/密钥识别与替换，合规刚需。
- **Vane**：❌。**行业**：NeMo Curator `PiiModifier`(Presidio)、DataTrove `PIIFormatter`。
- **jude 现状**：❌。
- **难度**：小（正则/Presidio UDF）→中（NER）。**价值**：中→高。

#### C11. 基准去污染 Task Decontamination
- **解决什么**：从训练集删除与评测基准（MMLU/GSM8K）重叠的样本，避免评测泄漏。发布合规数据集必备。
- **Vane**：❌。**行业**：NeMo Curator `TaskDecontamination`、n-gram 去污染。
- **jude 现状**：❌。
- **难度**：中（n-gram 索引 + 匹配删除，复用去重基建）。**价值**：中。

### 🟡 第三梯队 —— 生态对接

- **C12. vLLM 原生集成 + 前缀感知分桶路由**：Vane 招牌（`duckdb/execution/vllm.py` `PrefixRouter` + C++ 分桶），jude 只有 options 壳。**难度大、价值极高**，但更偏「推理」，本线暂缓、单列。
- **C13. 训练数据加载桥**（torch IterableDataset / 流式喂 trainer）：难度中、价值中。
- **C14. HF Datasets / WARC reader**：难度小→中、价值中。
- **C15. Tokenization / token 计数 / sequence packing**：难度小→中、价值中。
- **C16. 数据血缘 / 溯源**：难度中→大、价值中（研究/合规级）。
- **C17. 图像/多模态去重与质量过滤**（pHash / CLIP-score / NSFW）：难度中、价值中→高（结合 jude 多模态定位，潜在护城河）。

---

## 3. 架构原则（与 jude 既有设计一致）

1. **算子在 Rust，编排在 Rust，Python 只做薄 RPC**（延续「尽可能少 py」）。计算密集的核（MinHash 签名、n-gram、规则统计）用 Rust/PyO3 向量化。
2. **一等算子，非 example**：以 `Relation` 方法 + SQL 函数暴露，如 `rel.fuzzy_dedup(...)` / `rel.quality_filter(...)` / `rel.chunk_text(...)`，可组合进 SQL/关系代数。
3. **分布式复用现有基建**：去重的 banding 分桶复用 hash-shuffle；全局 shuffle 复用 shuffle exchange；语义去重复用 Lance 向量索引 + 分布式聚合。
4. **流式优先**：分块、质量过滤、PII 等 row-wise 算子走 sub-batch 流式；去重/混合走 stage-DAG 执行器。
5. **可观测**：每个数据治理算子上报 MetricsRegistry（输入/保留/删除行数、各规则命中数），前端可视化「数据漏斗」。

---

## 4. 落地顺序（价值/难度比）

### 阶段一：小而高价值（让 jude 立刻具备「数据治理」能力）
- **C2 精确去重**（分布式 distinct 封装 + 内容规范化 hash）
- **C4 语言识别**（fastText vectorized UDF）
- **C5 字符/递归分块**（Rust 分块 kernel + `chunk_text` / `embed_text(max_chunk_chars=)`）
- **C9 前半：加权混合采样**

### 阶段二：中难度支柱
- **C3 质量过滤规则集**（Gopher/C4 规则的 Rust 标量函数库）
- **C1 模糊去重 MinHash-LSH**（Rust 签名 + shuffle banding + 分布式连通分量）
- **C6 结构化输出**（`prompt(return_format=)`）
- **C7 语义去重**（复用 embedding + Lance + 聚合，编排成 SemDeDup）
- **C8 训练格式 writer**（WebDataset / MDS + 尺寸对齐分片）
- **C10 PII**、**C17 多模态过滤**、**C9 后半：分布式全局 shuffle**

### 阶段三：啃硬骨头
- **C12 vLLM 前缀分桶**（连续批处理 + 前缀路由 + actor 池）
- **C11 去污染**、**C13 torch 桥**、**C14 WARC/HF reader**、**C15 sequence packing**、**C16 血缘**

---

## 5. 里程碑验收

每个算子的验收标准：
1. Rust 核 `cargo test --lib` 覆盖（纯逻辑单测，无 PyO3 符号依赖）。
2. Python 端 pytest：正确性对拍（分布式结果 == 单机 ground truth）+ 边界（空表/单分片/倾斜）。
3. 分布式：在多分片下运行，去重/混合类要验证**跨分片全局**正确（不是 per-shard）。
4. 可观测：算子上报输入/输出/删除计数到 MetricsRegistry。
5. 文档：本文对应条目从「规划」更新为「已实现 + 证据文件:行号」。

---

## 附：关键证据（调研留痕）

- Vane 去重/清洗仅 example：`vane/examples/minhash_dedupe.py`、`common_crawl.py`、`llms_red_pajamas.py`
- Vane 分块/结构化输出：`vane/ai/functions.py`（`chunk_text` L260、`return_format` L530/L838）、`vane/ai/providers/vllm.py` L56-88
- Vane vLLM 前缀路由：`vane/duckdb/execution/vllm.py`（`PrefixRouter` L785）+ `vane/src/duckdb_py/vllm_executor.cpp` L301-304
- jude AI：`src/ai/functions.rs`（embed 无 chunk、prompt 无 return_format）、`src/ai/options.rs` L169-193（vLLM 仅 options，无 provider）
- jude 向量栈：`python/jude/_lance.py`（`create_vector_index` L53、`vector_search` L83）
- jude 写出：`src/relation.rs`（`to_parquet` L1417、分区 parquet L1530、`sample` L1395）
