# jude 大模型数据引擎 —— 预训练/后训练痛点整合 · 设计与执行计划

> 视角：先后以 **LLM 后训练工程师** 和 **LLM 预训练工程师** 的身份审视 jude，把两类真实
> 工作流的痛点合并、去重、按"能否解锁真实使用"排序，给出一份可落地的设计文档 + 分阶段执行
> 计划。本文承接 `pain_points_audit.zh.md`（引擎内部 bug 已基本清完）与 `llm_data_engine_plan.zh.md`
> （早期能力规划），聚焦"要变成研发/生产真敢用的工具，还差什么"。

---

## 0. 定位与一条必须澄清的原则

jude 是**数据引擎**：负责数据的 IO、清洗、去重、混合、分词打包、编排与可观测；**不实现模型的
前向/训练本身**。但这不等于"数据管线里没有模型"——现代 curation 最大的质量杠杆恰恰是模型
（FineWeb-edu / DCLM 的质量分类器、LLM-as-judge、合成数据）。因此：

- **「把模型当一个 stage 来编排」是一等公民能力**，不是附属。引擎负责 **batching + 资源调度 +
  背压 + 重试**；模型由用户提供。
- **没有 GPU 也能做**。stage 调用的模型可以是：
  - **CPU 分类器**：fastText 质量分/语言识别、ONNX Runtime 上的小 BERT —— FineWeb-edu 的质量
    过滤器本身就是 fastText，纯 CPU 能跑；
  - **远程推理端点**：vLLM/TGI/OpenAI 兼容 API（引擎只发 HTTP，GPU 在别处或没有）；
  - **mock actor**：开发/测试期用假模型验证调度+背压+重试正确性。
  - GPU 只是 stage 的一个**资源注解**（`Resources(gpus=1)`），cosmos 已支持；没有 GPU 时注解为
    0，用 CPU/远程后端。**所以模型-stage 的全部管道逻辑现在就能建、就能测。**
- **S3 用本地 MinIO 测试**：面向 `s3fs`/`fsspec` 抽象，测试套件起一个 MinIO 容器验证读写。

**明确 out-of-scope（引擎永不做）**：模型权重、训练循环、推理 kernel、GPU 算子。这些是用户带进
来的 stage 内容。

---

## 1. 痛点整合清单（两视角去重后）

标注：`[pre]` 预训练 · `[post]` 后训练 · `[both]` 两者 · 状态 `✅已做` / `🔧部分` / `⬜新增`。

### L0 —— IO & 存储（"根本进不来/出不去"层）
| 编号 | 痛点 | 视角 | 状态 |
|---|---|---|---|
| L0.1 | 只支持本地 FS,无 S3/GCS/OSS(数据都在对象存储,PB 级) | both | 🔧(jude.storage fsspec 层:memory/file 已测,s3://MinIO 同一代码路径,配 endpoint_url+s3fs) |
| L0.2 | 无 WARC/WAT/WET 读（CommonCrawl 是 WARC，预训练管线的入口） | pre | ⬜ |
| L0.3 | 输出只 `collect()` 成一张表，无**流式写 sink**（PB 输出攒不进内存） | both | ⬜ |
| L0.4 | 无压缩分片流式读（`.jsonl.gz`/`.warc.gz` 千万分片） | pre | 🔧(单机流有) |

### L1 —— 抽取 & 规范化
| L1.1 | HTML→正文抽取（trafilatura/resiliparse 式 boilerplate 去除） | pre | ⬜ |
| L1.2 | 编码/乱码修复（ftfy 式 mojibake fix）、Unicode NFC 规范化 | both | ⬜ |
| L1.3 | 语言识别（heuristic 已做；缺 fastText lid.176 = CPU 模型 stage） | both | 🔧 |

### L2 —— 去重（预训练最硬的问题）
| L2.1 | 全局模糊去重（MinHash-LSH 跨**整语料**万亿文档） | pre | 🔧(全局 CC 用**流式 union-find**,driver 内存 = 标签数组+单桶边,不再物化全量边;超 driver 内存的标签数组需分布式 label-prop,已记为后续) |
| L2.2 | 全局精确子串去重（`substring_dedup` 现单机顺序、全局 hash set 在一进程） | pre | 📝(单机版可用;分布式按 hash 窗口 shuffle 为后续) |
| L2.3 | 增量去重（新 CC dump 对**已存在语料**去重，不重跑全量）—— 需持久化 dedup 索引(bloom/MinHash store) | pre | ⬜ |
| L2.4 | 行级/文档级/URL 级精确去重 | both | ✅ |
| L2.5 | 语义去重 SemDeDup（贪心非传递 + 聚类 scale） | post | ✅ |

### L3 —— 质量 & 安全
| L3.1 | 启发式质量（Gopher/C4：停用词门/重复 n-gram/digit 门/行清洗） | both | ✅ |
| L3.2 | **模型质量分类器作为一等 stage**（FineWeb-edu/DCLM：fastText/小 BERT，CPU 可跑） | both | 🔧(引擎侧 model_score/filter + ModelScorer stage 已做,GPU-free) |
| L3.3 | perplexity 过滤（KenLM 或小 LM stage） | pre | ⬜ |
| L3.4 | PII 脱敏（Luhn 已加）/毒性/安全过滤（毒性需模型 stage） | both | 🔧 |
| L3.5 | 基准去污染（抗稀释覆盖度已做） | both | ✅ |

### L4 —— 分词 / Packing / 分片（"我真正要的输出"）
| L4.1 | tokenizer 接入（HF tokenizers / tiktoken），token 计数 | both | 🔧(byte 内置+可插拔已做) |
| L4.2 | sequence packing（拼接+按 seq_len 切、doc 边界、EOS、attention mask） | both | ✅ |
| L4.3 | token-shard 写出 + 索引（Megatron `.bin/.idx` 或 Lance token 列）| both | 🔧(Lance+flat .bin 已做) |
| L4.4 | token 级全局 shuffle（PB token 跨 shard 洗牌，落盘/对象存储，非内存） | both | 🔧(write_shuffled_shards 落盘分片 已做) |
| L4.5 | tokenizer-aware 长度阈值（质量门用 token 数而非词数，C15） | both | ⬜ |

### L5 —— 混合 / 血缘 / 可复现
| L5.1 | token 比例配比 + 上/下采样 + 域权重（DoReMi 式）+ epoch 控制 | pre | 🔧(blend_by_tokens 已做) |
| L5.2 | provenance 列（每条 token 来自哪个源、过了哪些 filter/版本） | both | 🔧(add_provenance/blend 打 _source) |
| L5.3 | **pipeline 配置 hash → 数据版本 → checkpoint** 血缘绑定（Lance 版本化是地基） | both | 🔧(jude.lineage sidecar 已做) |
| L5.4 | 语料 profiling（HLL 基数、长度/语言/dup 率 sketch，PB 级近似统计） | both | ✅ |

### Cross-cutting —— 健壮性 & 规模
| X.1 | 长作业容错：map 路径**直接用 cosmos-xenna 内建 FT**（jude 现在一个旋钮都没设）；shuffle 路径用 Ray lineage + actor 重启（`_JudeWorker` 现 `max_restarts=0`，没开） | both | 🔧(机制在,没配) |
| X.2 | shuffle spill（预训练全局 shuffle/dedup **必须** spill，之前按要求跳过——需重启） | pre | ⬜ |
| X.3 | 交互式采样 + 被丢文档探查（`sample_dropped`，调阈值救命） | both | ✅ |
| X.4 | 规模真凭实据（当前 bench 单机/模拟多机，PB/百节点无实证） | both | ⬜ |

---

## 2. 目标架构（数据管线分层）

```
        ┌────────────────────────────────────────────────────────────────┐
Layer 0 │ Source: s3fs/fsspec (MinIO test) · WARC/WET reader · gz shards   │
        │ Sink:   streaming shard writer → Lance/parquet on object store  │
        ├────────────────────────────────────────────────────────────────┤
Layer 1 │ Extract: HTML→text · encoding/mojibake fix · unicode NFC · lid  │
        ├────────────────────────────────────────────────────────────────┤
Layer 2 │ Dedup:  exact(line/doc/url) · global fuzzy(MinHash, spill+UF)    │
        │         · global exact-substring(distributed) · incremental idx │
        ├────────────────────────────────────────────────────────────────┤
Layer 3 │ Quality: heuristic(Gopher/C4) · MODEL STAGE(classifier/ppl)      │←── 用户带模型
        │          · PII/toxicity · decontamination                       │    (CPU/远程/GPU)
        ├────────────────────────────────────────────────────────────────┤
Layer 4 │ Tokenize → pack(seq_len,EOS,mask) → token-shard write + index   │
        │          → token-level global shuffle (spill-backed)            │
        ├────────────────────────────────────────────────────────────────┤
Layer 5 │ Mix(token-ratio,domain weight) · provenance · lineage(cfg→ver)  │
        │          · corpus profiling (HLL/sketch)                        │
        └────────────────────────────────────────────────────────────────┘
Cross:  fault-tolerant staged executor (checkpoint/resume) · spill ·
        interactive sample/inspect · Rust WorkerManager (size-aware sched)
```

引擎不变的原则（与既有一致）：调度决策在 Rust（`WorkerManager`）；Python 是薄 Ray RPC + 控制流；
compute-heavy kernel 在 Rust（`curate`）；模型只作为 stage 内容由用户注入。

---

## 3. 执行计划（分阶段）

排序原则：**先打通管线两端**（进得来 L0 / 出得去 L4），因为这是"能不能用"的开关；再啃**规模去重**
（L2，预训练核心）与**容错**（X.1，长作业存在性）；模型-stage（L3.2）穿插进来（无 GPU 用 CPU/远程测）；
最后是**血缘/混合/profiling**（用得爽不爽）。每阶段列 in-scope、Python/Rust、测试策略、验收。

### 阶段 P1 —— 打通两端（解锁"能用"） 【最高优先】
- **L0.1 对象存储读写**：`storage.open(url)` 走 `fsspec`（`s3://`/`gs://`/`file://`），凭证从 env。
  scan 源 + Lance/parquet sink 都接。*测试：docker MinIO + `s3fs`，读写往返 + 分片 glob。*
- **L0.3 流式写 sink**：`pipeline.write_lance(path, shard_rows=)` / `write_parquet_shards` —— 每个输出
  shard 产出即写对象存储，不 `collect()`。配合 D3 的深度优先流式，做到输入输出都有界。
- **L4.1+L4.2+L4.3 分词打包写出**（北极星）：
  - `curate.tokenize(col, tokenizer="…")` → HF tokenizers（Rust `tokenizers` crate 或 Python HF），加 `n_tokens` 列；
  - `curate.pack_sequences(seq_len, eos_id, add_doc_mask=)` → 拼接 + 切 seq_len，产 `input_ids`/`doc_ids`/`lengths`；
  - `write_token_shards(path, format="megatron"|"lance")` → `.bin/.idx` 或 Lance `list<int32>` 列 + manifest。
  - **L4.5**：`quality_filter(min_tokens=…, tokenizer=…)` token-aware 长度门。
  - *测试：小语料 tokenize→pack→写→读回，验证 token 数守恒、doc 边界、EOS 位置、seq_len 对齐；
    Megatron `.idx` 结构校验。*
- 归属：**纯 Python + 可选 Rust tokenizers**。无 GPU、无大规模需求，立刻可测。

### 阶段 P2 —— 抽取前半（解锁预训练入口）
- **L0.2 WARC/WET reader**：`datasource.WarcSource`（`warcio`/`fastwarc`）→ 流式产 (url, html/text) 记录；接进 L0 流式读。
- **L1.1 HTML→正文**：`extract_text(col, engine="resiliparse"|"trafilatura")` stage（可选依赖，缺则报明确错）。
- **L1.2 编码修复 + NFC**：`fix_encoding` / `normalize_unicode`（`ftfy` 可选 + 内置 NFC，纯 Python/Rust）。
- *测试：小 WARC fixture 端到端抽正文；mojibake 样例修复；NFC 幂等。*

### 阶段 P3 —— 模型作为一等 stage（无 GPU 也能做） 【与 P2 并行】
- **L3.2 分类器 stage 框架**：`pipeline.model_filter(fn|endpoint, batch_size=, resources=Resources(gpus=0..N), threshold=)`
  —— 引擎负责 micro-batch、背压（`max_in_flight`）、重试；stage 体是用户函数：
  - CPU 后端：加载 fastText/ONNX 分类器（`model_stage_fasttext(path)`）——**FineWeb-edu 质量过滤 = 这条**；
  - 远程后端：`model_stage_http(endpoint, template)` 发 OpenAI 兼容请求（GPU 在别处/或无）；
  - mock 后端：测试期确定性假打分，验证调度语义。
- **L3.3 perplexity**：`model_stage_kenlm(model)`（KenLM CPU）或复用远程 LM stage。
- *测试（GPU-free）：mock 分类器验证 batching/背压/重试正确性 + 顺序保持；CPU fastText 小模型 fixture
  跑质量分过滤端到端；HTTP stage 打到本地 mock server。*
- 归属：cosmos stage + RayRunner；把"模型 stage"从附属提为文档首页示例。

### 阶段 P4 —— 规模去重 + 容错（预训练核心，最难） 【啃硬骨头】
- **X.2 shuffle spill**（重启）：reducer 侧当 bucket 超阈值落盘（Arrow IPC 分块），`concat` 改为分块合并；
  全局 shuffle/dedup 的前置。
- **L2.1 全局模糊去重 scale**：driver 侧全量 CC → **分布式 union-find / 迭代 label-propagation**（边按 rid 再 shuffle，
  多轮收敛），(rid,sig) shuffle 走 spill。*测试：模拟多节点跨桶簇 == 单机；注入节点失败验证收敛。*
- **L2.2 全局精确子串去重**：窗口 hash 走分布式 shuffle（同 hash 窗口共桶），首次出现全局定序。
- **L2.3 增量去重**：持久化 MinHash/bloom 索引（Lance 存 sig 列），新数据只对索引查。
- **X.1 长作业容错 —— 两套互补机制，都不自建 checkpoint 系统**：

  **(a) pipeline/map 路径 → 直接用 cosmos-xenna 内建容错。** 预训练里绝大部分长跑、易挂节点的活是
  **embarrassingly-parallel 的 map**（每 shard：抽正文/过滤/分词/模型打分），这正好是 cosmos 的 stage 模型。
  cosmos `PipelineConfig` 已自带生产级 FT 旋钮，jude 现在**一个都没设**（`_run_cosmos` 只给了 execution_mode +
  return_last_stage_outputs）。要做的只是**透传/默认打开**：
  `num_run_attempts_python`(task 重试)、`num_setup_attempts_python`+`reset_workers_on_failure`(actor 起不来/挂了重建)、
  `max_setup_failure_percentage`(容忍一部分坏节点)、`ignore_failures`/`failures_return_nones`(坏 record 跳过不 abort)、
  `worker_restart_interval_m`/`worker_max_lifetime_m`、`enable_work_stealing`(负载再均衡)。
  → **FT-critical 的长 map 作业跑 cosmos 引擎，容错几乎白得**（NVIDIA 就是拿它做 PB 级多模态 curation 的）。

  **(b) shuffle 路径（jude 自己的 RayRunner：全局 dedup / join / group-by）→ Ray actor 重启（不盲目重试）。**
  这些是自定义 all-to-all shuffle，不是 cosmos stage。`_JudeWorker` 配 **`max_restarts`**（死了重建，保住整个池——
  状态只有可重建缓存，永远安全）；**`max_task_retries` 默认 0**（不自动重跑在途 task）——因为 Ray 的 task 重试是
  actor-wide 的，纯查询/读 task 幂等（对不可变输入的 SELECT/agg 重跑结果一致）可重试，但**写 task（write_lance_fragment/
  write_parquet_file）有副作用不能盲目重跑**；只读管线可 `JUDE_ACTOR_MAX_TASK_RETRIES>0` 显式开。
  （注：Ray 只在 actor/节点死时重试，application error 从不重试。）源读 durable + 可选 shuffle 边界 `write_lance` 粗物化。

  **流式/pipelined 生成器路径不做 FT**（中途死了丢位置）——FT-critical 走 (a) cosmos 或 (b) stage-based。
  *测试：cosmos 路径注入 stage 异常验证 `ignore_failures`/重试；shuffle 路径 `ray.cluster_utils` 杀 worker 验证 actor 重启 + lineage 重算，结果与无故障一致。*
- 归属：Rust（spill 编解码、UF）+ Python（多轮 shuffle 编排）。

### 阶段 P5 —— 混合 / 血缘 / 探查（用得爽）
- **L5.1 token 配比混合**：`blend(sources, token_ratios=, upsample=, domain_weights=)` 按 token 数而非行数。
- **L5.2/L5.3 provenance + 血缘**：每条加 `_source`/`_pipeline_ver`；`run()` 把 **config hash + 输入数据版本 + 输出 Lance 版本**
  写进 `observe` 审计，形成"配置→数据版本→checkpoint"可查链。
- **L5.4 语料 profiling**：`profile(col)` → HLL 近似基数、长度/语言/dup 率直方图（cheap，PB 可算）。
- **X.3 交互式探查**：`CurationFlow.sample_dropped(op, n=)` / dashboard "被某过滤器丢掉的 N 条 + 原因"。
- **L4.4 token 级落盘全局 shuffle**：复用 P4 的 spill shuffle，输出 shard 全局随机。
- 归属：纯 Python（profiling 的 HLL 可 Rust）。

### 阶段 P6 —— 规模实证（可信度）
- **X.4**：在真实多节点（或大 spot 集群）跑一次 100GB→1TB CommonCrawl 子集端到端
  （WARC→抽取→dedup→质量→分词→写 shard），出吞吐/成本/容错报告。补真实（非模拟）多机数。

---

## 4. 测试策略（无 GPU、本地可跑）

- **S3**：`docker run minio/minio` + `s3fs`，测读写往返、分片 glob、流式写 sink。CI 用 MinIO service 容器。
- **模型 stage**：mock actor（确定性假打分）测调度/背压/重试；CPU fastText/ONNX 小模型 fixture 测端到端；
  HTTP stage 打本地 mock server。**全程零 GPU。**
- **规模去重/容错**：`ray.cluster_utils.Cluster` 多节点 + 主动杀 worker，验证 parity + 恢复。
- **分词打包**：token 守恒、doc 边界、seq_len 对齐、Megatron `.idx` 结构、读回一致。
- 每个算子延续本仓风格：小、确定性、与单机 parity 对拍。

---

## 5. 里程碑验收

- **M1（P1）**：一条 `s3://…/*.jsonl` → 清洗 → `write_token_shards(s3://…, megatron)` 的端到端跑通（MinIO 上），
  产出能被 Megatron/nanotron loader 读。→ **jude 从"文本工作台"变成"出训练就绪 token"**。
- **M2（P2+P3）**：`s3 WARC → 抽正文 → lid → CPU fastText 质量分过滤 → dedup → token shard`，全 CPU、MinIO。
  → **能碰真实 CommonCrawl 形态的数据 + 模型质量过滤（无 GPU）**。
- **M3（P4）**：全局模糊去重在模拟多节点上与单机 parity，且杀 worker 能恢复；spill 让 shuffle 不 OOM。
  → **敢在真实大 run 里用**。
- **M4（P5+P6）**：血缘可查（配置→数据版本→checkpoint）、profiling 出报告、真实多机吞吐实证。
  → **天天用、可复现、可信**。

---

## 6. 与既有工作的衔接（不重复建设）

本会话已完成的地基（`pain_points_audit.zh.md`）：分布式全局 fuzzy dedup 正确性（A3）、贪心 SemDeDup（A5）、
LSH 校准（C3）、Gopher/C4 质量信号 + Luhn + 抗稀释去污染（C2/C6）、精确子串去重单机版（C1）、
size-aware split 分配在 Rust（E3）、加盐倾斜连接（B4）、分布式 GROUP BY 归并（B2）、分布式全局 IDF BM25（A2）、
`from_datasource` 真流式（D3）、缓存失效（A1）、原生 id 类型（A4）。

本计划是在这些**正确的单机/小规模地基**之上，补**规模化（spill/UF/容错）**、**管线两端（IO/分词打包）**、
**模型 stage 一等化** 与 **血缘/混合/profiling**，把 jude 从"正确的原型"推到"预训练/后训练工程师真敢用"。
