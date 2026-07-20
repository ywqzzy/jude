# jude LLM 数据处理算子：实现原理与用途

> 面向"想搞懂 jude 的 LLM 数据处理算子怎么实现的、每个有什么用"的读者。
> 定位：jude 是**大模型数据处理引擎**——给 LLM 训练/RAG/推理准备数据。这些算子是它的核心竞争力。
> 配套：能力规划见 [`llm_data_engine_plan.zh.md`](llm_data_engine_plan.zh.md)。最后更新：2026-07-19。

---

## 0. 总览：为什么需要这些算子

原始语料（网页、PDF、文档）**不能直接喂给大模型**。业界共识的数据准备流水线大致是：

```
原始数据 → 分块(chunk) → 质量过滤(quality) → 精确去重(exact dedup)
        → 模糊去重(fuzzy dedup) → [语义去重] → 混合/打散 → 写训练格式
```

每一步都直接影响模型质量:
- **去重**决定训练效率和过拟合风险(重复数据让模型"背答案")——训练语料 30-70% 是近重复。
- **质量过滤**剔除垃圾文本(乱码、模板、导航栏),否则模型学到噪声。
- **分块**是 RAG 索引和长文本 embedding 的前提。

jude 把这些做成**引擎内建、Rust 加速、可组成 cosmos 多阶段 batch pipeline** 的一等算子——这是它区别于"通用数据引擎 + 手写 example"(Vane/Daft 的做法)的地方。

**架构分层(每个算子都一样):**
```
Rust 计算核 (src/curate.rs)         ← 纯函数、单测、无 GIL,热路径在这
   ↓ PyO3 (src/curate_py.rs)         ← 向量化批接口 (list[str] -> list)
Python facade (jude/curate.py)       ← Arrow 表级算子 + cosmos Stage
   ↓
用三种方式用: ①直接对 Relation/Table ②cosmos pipeline Stage ③map_batches UDF
```

---

## 1. 文本分块 Chunking (C5)

### 用途
把长文档切成小块。**RAG** 要把文档切块后分别 embedding 存进向量库检索;**长文本 embedding** 因为模型有最大长度,必须先切;**训练**时也常按块组织。

### 实现
`src/curate.rs` 两个核:

**① `chunk_chars(text, chunk_chars, overlap)` — 硬字符切分**
- 按 Unicode **字符**(不是字节)切,多字节文本(中文/emoji)安全。
- `overlap` 让相邻块重叠 N 个字符——避免把一句话从中间切断导致语义丢失。
- 步长 `step = chunk - overlap`,滑窗前进,末块对齐到结尾。

**② `chunk_recursive(text, chunk_chars, overlap, separators)` — 递归分隔符切分(LangChain 风格)**
- 按分隔符优先级尝试:`["\n\n", "\n", ". ", " "]`——先按段落分,段落太大再按行,再按句子,最后按词。
- 核心思想:**尽量在自然边界(段落>句子>词)切开**,而不是粗暴地按字符数硬切。
- 切出的碎片再**贪心合并**到接近 `chunk_chars`,块之间带 `overlap` 字符。

### 关键代码
```rust
// 递归:能整块放下就整块;否则按当前分隔符拆,拆完的碎片对不够大的再降级分隔符
fn split_recursive(text, chunk, seps, depth) {
    if text.len() <= chunk { return vec![text]; }
    if depth >= seps.len() { return chunk_chars(text, chunk, 0); }  // 兜底硬切
    let sep = seps[depth];
    for part in text.split(sep) {
        if part.len() > chunk { out.extend(split_recursive(part, chunk, seps, depth+1)); }
        else { out.push(part); }
    }
}
```

### 用法
```python
from jude import curate
# 1 行 -> 多行(其余列复制),加 chunk + chunk_index 列
out = curate.chunk_text(table, column="text", chunk_chars=1024, overlap=128, recursive=True)
# 或作为 cosmos 阶段
pipe.chunk(chunk_chars=1024, overlap=128)
```

---

## 2. 精确去重 Exact Dedup (C2)

### 用途
删掉**完全重复**(或只差大小写/空白)的文档。这是去重最便宜的第一道——很多语料里同一篇文章被抓了几十次。

### 实现
`src/curate.rs`:

**① `normalize_text(text)` — 归一化**
- 小写化 + 把所有连续空白(空格/换行/tab)压成单个空格 + 去首尾空白。
- 效果:`"Hello   WORLD\n"` 和 `"hello world"` 归一化后相同 → hash 相同 → 判为重复。

**② `content_hash(text, normalize)` — SHA-256 指纹**
- 对(归一化后的)文本算 SHA-256,输出十六进制字符串。相同内容 → 相同 hash。
- 为什么用 SHA-256:碰撞概率可忽略,可作为去重 key 直接 `DISTINCT`。

### 去重逻辑
```python
# 加一列 content_hash,然后按它去重(保留首次出现,稳定)
out = curate.exact_dedup(table, column="text", normalize=True)
```
分布式:算完 hash 列后,直接复用 jude 现有的**分布式 DISTINCT**(hash-shuffle 让相同 hash 落同一 worker),所以精确去重天然可扩展到多机。

---

## 3. 质量过滤 Quality Filtering (C3)

### 用途
剔除垃圾文本:乱码、符号堆砌、导航栏/页脚模板、极短或极长文档、大量重复行。**Gopher / C4 / RefinedWeb** 论文的启发式规则集是行业共识。低质数据直接拖垃圾进模型。

### 实现
`src/curate.rs` 一次遍历算出 `QualitySignals`:

| 信号 | 含义 | 抓什么问题 |
|---|---|---|
| `word_count` | 词数 | 太短(没信息)/太长(可能是拼接垃圾) |
| `mean_word_len` | 平均词长 | 太短=乱码/符号,太长=没空格的垃圾串 |
| `symbol_ratio` | 非字母数字非空白字符占比 | 符号堆砌(`!@#$%^`)|
| `alpha_word_ratio` | 含字母的词占比 | 大量纯数字/符号"词" |
| `dup_line_ratio` | 重复行占比 | 模板/导航栏(同一行出现很多次)|
| `top_word_ratio` | 最高频词占比 | 单词疯狂重复(spam)|

**判定** `quality_reject_reason(signals, thresholds)`:按 Gopher 风格阈值逐条检查,返回第一个不通过的原因(如 `"too_few_words:12<50"`),全过返回 `None`。默认阈值:词数 50~10万、平均词长 3~10、符号占比<0.3、含字母词占比>0.6、重复行<0.3、最高频词<0.3。

### 用法
```python
# 直接过滤(丢弃不合格行)
out = curate.quality_filter(table, min_words=50, max_symbol_ratio=0.3)
# 或标注模式(不丢,加一列 reject 原因,便于审计/调阈值)
out = curate.quality_filter(table, reason_column="reject")
# 或只加信号列自己分析
out = curate.quality_signals(table)   # 加 q_word_count / q_symbol_ratio / ...
# cosmos 阶段
pipe.quality_filter(min_words=50)
```

---

## 4. 模糊去重 Fuzzy Dedup — MinHash + LSH (C1) ⭐

### 用途
删掉**"意思几乎一样、字面略有差异"**的近重复文档——模板页、转载改一两个词、样板合同。这是 LLM 数据处理**最核心也最难**的一步,精确 hash 抓不到(差一个字符 hash 就完全不同),需要**相似度**去重。SOTA 数据集(FineWeb/RefinedWeb)都靠它。

### 原理(三步)
**难点**:N 篇文档两两比相似度是 O(N²),百万文档跑不动。MinHash+LSH 把它降到接近 O(N)。

**① MinHash 签名** — 把每篇文档压成一个定长(如 128)的整数向量,使得**两个签名相等位的比例 ≈ 两篇文档的 Jaccard 相似度**。
- 先把文档切成**词 n-gram shingle**(如 2-gram:`"the quick"`, `"quick brown"`…),得到一个 shingle 集合。
- 用 128 个不同的哈希函数 `h_i(x) = (a_i·x + b_i) mod p`(p 是大质数 2⁶¹-1),对每篇文档,签名第 i 位 = 所有 shingle 过 `h_i` 的**最小值**。
- 数学性质:`P(minhash_i(A) == minhash_i(B)) = Jaccard(A,B)`。所以签名相等位的比例就是 Jaccard 估计。
- **确定性**:哈希系数 `(a_i,b_i)` 由 `seed` 经 splitmix64 生成——不同 worker 用同一 seed 得到一致签名,才能分布式。

**② LSH 分桶(banding)** — 避免两两比较。
- 把 128 位签名切成 `bands` 个 band(如 16 band × 8 行)。
- 每个 band 的 8 行一起 hash 出一个 band key(前缀 band 序号防跨 band 碰撞)。
- **两篇文档只要有任意一个 band key 相同,就是候选近重复**。相似度越高,越可能在某个 band 上完全一致 → 命中同一桶。
- 这样只需比较**落进同一桶的文档对**,而不是全体两两比。

**③ 连通分量聚类** — 候选对可能成链(A~B, B~C)。
- 用 **Union-Find(并查集)** 把候选对合并成"近重复簇"。
- 每簇保留一个代表(最小行号,确定性),其余删除。

### 关键代码
```rust
// MinHash: 每个哈希函数对所有 shingle 取最小
for &x in &shingle_hashes {
    for (i, &(a, b)) in coeffs.iter().enumerate() {
        let hv = (a as u128 * x + b) % P;
        if hv < sig[i] { sig[i] = hv; }   // 取 min
    }
}
// LSH: 每个 band 的行 hash 成一个 key
for b in 0..bands {
    let slice = &sig[b*rows .. b*rows+rows];
    keys.push(format!("{b}:{hash(slice):016x}"));
}
// 聚类: 候选对 union,取每簇最小 id 为代表
uf.union(a, b);  reps = uf.representatives();
```

### 用法
```python
# MinHash -> LSH 分桶 -> 验证 Jaccard>=threshold -> 并查集 -> 每簇留一个
out = curate.fuzzy_dedup(table, column="text", threshold=0.7,
                         num_hashes=128, bands=16, ngram=2)
# 或标注每行所属簇代表(不删)
out = curate.fuzzy_dedup(table, threshold=0.7, keep_cluster=True)  # 加 dup_cluster 列
```

**分布式化**(设计就绪):LSH band key 就是 shuffle key——按 band key 做 hash-shuffle,把候选文档路由到同一 worker,每 worker 本地算候选对,driver 端汇总做全局连通分量。复用 jude 现有的 hash-shuffle 基建。

### 参数怎么调
- `threshold` 高(0.8+)= 只删很像的,保守;低(0.5)= 激进去重。
- `bands` 多 = 召回高(更容易判为候选)但假阳多;`bands` 少 = 精确但可能漏。经验:`bands × rows = num_hashes`,band 数控制 LSH 的 S 曲线拐点在 `threshold` 附近。
- `ngram` 大 = 对词序更敏感;2~3 常用。

---

## 5. 怎么组成一条完整的 curation pipeline

用 cosmos 多阶段 batch(每阶段独立分配资源、独立并行):
```python
import jude.pipeline as jp
from jude import curate

result = (
    jp.RelationPipeline.from_table(raw_docs, rows_per_shard=10000, engine="cosmos")
      .quality_filter(min_words=50)        # C3: 先扔垃圾(最便宜)
      .chunk(chunk_chars=1024, overlap=128) # C5: 切块
      .content_hash()                       # C2: 加去重 key
      .run()
)
# 精确/模糊去重是全局操作(要跨分片),在 pipeline 后用 Relation 级算子:
result = curate.exact_dedup(result)
result = curate.fuzzy_dedup(result, threshold=0.7)
```

**为什么是 batch 不是 streaming**:数据 curation 是**多阶段批处理**——每个阶段处理整批、可独立扩缩、去重需要全局视角。这和"流式摄取"是两回事(用户明确指出这点)。

---

## 6. 现状与后续

**全部已实现**(文本 + 多模态,单机 + 分布式,测试全绿):

| 能力 | 文本算子 | 多模态 | 分布式 |
|---|---|---|---|
| C5 分块 | `chunk_text` | — | `curate_dist.dist_chunk_text` |
| C2 精确去重 | `exact_dedup` | — | `dist_exact_dedup`(hash shuffle) |
| C1 模糊去重 | `fuzzy_dedup`(MinHash-LSH) | `curate_mm.image_dedup`(pHash) | `dist_fuzzy_dedup`(band-key shuffle) |
| C7 语义去重 | `semantic_dedup`(embedding+聚类) | (复用) | (复用 embedding) |
| C3 质量过滤 | `quality_filter`(Gopher/C4) | `image_quality_filter` | `dist_quality_filter` |
| C4 语言识别 | `detect_language`/`language_filter` | — | `dist_detect_language` |
| C9 混合/shuffle | `blend_datasets`/`global_shuffle` | — | — |
| C8 训练格式 | `training_format`(WebDataset/MDS/parquet) | ✅ | — |
| C10 PII | `redact_pii`/`detect_pii` | — | (map-style 可分布) |
| C11 去污染 | `decontaminate`(n-gram) | — | — |
| C6 结构化输出 | `structured.extract`(JSON schema/Pydantic) | — | — |
| 向量检索 | `vector.knn`/`distributed_knn` + VSS HNSW；Lance ANN + FTS/混合RAG | — | ✅ |

**分布式两种形态**:map-style(分区并行,`curate_dist.dist_map`)+ dedup shuffle(按 dedup key 洗牌让同键共位,`dist_exact_dedup`/`dist_fuzzy_dedup`)。

**证据文件**:`src/curate.rs`/`src/curate_mm.rs`(计算核 + 单测)、`src/curate_py.rs`(PyO3 批接口)、`python/jude/curate.py`/`curate_mm.py`/`curate_dist.py`/`structured.py`/`training_format.py`/`vector.py`/`lance.py`、`tests/test_curate*.py`/`test_vector.py`/`test_lance_ops.py`/`test_training_format.py`。
