<p align="center">
  <img src="assets/jude-logo.svg" alt="jude" width="440">
</p>

<h3 align="center">Take a sad song and make it better.</h3>

<p align="center">
  <b>jude</b> 是面向 <b>LLM 训练与 RAG</b> 的分布式数据引擎 —— 治理、索引、检索网络规模的
  多模态数据,全部跑在原版 <b>DuckDB</b> 上,内核用 <b>Rust</b>。
</p>

<p align="center">
  <a href="https://github.com/ywqzzy/jude/actions/workflows/ci.yml"><img src="https://github.com/ywqzzy/jude/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License: Apache-2.0">
  <img src="https://img.shields.io/badge/rust-1.85%2B-orange.svg" alt="Rust 1.85+">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/built%20on-DuckDB-black.svg" alt="Built on DuckDB">
  <img src="https://img.shields.io/badge/scales%20on-Ray-028cf0.svg" alt="Scales on Ray">
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="#能力">能力</a> ·
  <a href="#快速上手">快速上手</a> ·
  <a href="docs/vector_retrieval_report.zh.md">性能报告</a>
</p>

---

> 名字取自 Beatles 的《Hey Jude》——*"take a sad song and make it better"*(把一首悲伤的歌变得更好)。
> 这正是它的工作:把原始、杂乱、网络规模的数据变得**更好** —— 更干净、去重、更高质量,可以拿去训练。

### 为什么用 jude

- **一个引擎,打通整条管线。** SQL 分析、LLM 数据治理、向量/全文检索,在*同一个*分布式运行时里 —— 不用把三套系统胶水拼起来。
- **该快的地方快。** 1M×768 维向量检索单机 **182 QPS / 99.9% 召回**;治理和检索都随 worker 近线性扩展。
- **Rust 内核,不 fork DuckDB。** 引擎与分布式调度用 Rust;DuckDB 用原版;Python 只是薄壳。没有 GIL 瓶颈,不锁死厂商。
- **原生多模态。** 图像 / 音频 / 视频 / 文档都是一等列,配流式的解码-变换 API。
- **模型自备。** jude 止步于推理边界 —— 它是你模型**周围**的数据引擎,不是又一个推理服务。

---

## 能力

### 🗄️ SQL 与关系核心
DuckDB 兼容的 `Connection` / `Relation` API,底层是真正的 `LogicalPlan` 算子 DAG(不是 SQL 字符串拼接 —— SQL 只是其中一种下降)。

```python
import jude
con = jude.connect()

# 完整 DuckDB SQL:join、窗口函数、CTE、聚合、集合运算
con.sql("""
    SELECT category, count(*) AS n, avg(score) AS mean
    FROM read_parquet('docs/*.parquet')
    WHERE year >= 2020
    GROUP BY category HAVING n > 100
    ORDER BY mean DESC
""").show()

# 或 DataFrame 风格的关系 API(惰性 LogicalPlan;SQL 是它的下降)
r = (con.from_parquet("docs/*.parquet")
        .filter("year >= 2020")
        .aggregate("category", "count(*) AS n, avg(score) AS mean"))
r.to_arrow()
```

- 数据源:CSV / Parquet / JSON / Arrow / Lance / Hive / Iceberg。
- 流式执行:结果以 Arrow `RecordBatch` 流式返回(内存有界)。
- 进程外 **UDF** —— Rust 子进程池释放 GIL(`Python::detach`)获得真正的 CPU 并行(标量、表/生成器、flat-map、聚合 UDF)。

### 🌐 分布式查询引擎(Rust 编排,跑在 Ray 上)
同一套 SQL / 关系程序在**原版** DuckDB 上分布式执行 —— 不 fork 绑定。调度**大脑**在 Rust(`WorkerManager`、split 分配器、资源准入控制、跨查询装箱);Python 只转发 Ray RPC。

```python
from jude.runners.ray import RayRunner
runner = RayRunner(num_workers=8)
result = runner.collect(relation)          # 分区、并行、归并
```

- **流式 stage-DAG 执行器** —— 在 shuffle 边界(Aggregate / Join / Distinct / Order / SetOp)把计划切成流水线阶段;map 与 reduce 重叠。
- **分布式 join 与聚合** —— hash-shuffle exchange 通过 Ray object store 流转 Arrow(`distributed_join`、流式变体、两阶段聚合);确定性 FNV-1a shuffle 路由。
- **健壮性** —— 查询级重试:worker 故障时整条分布式读**重跑**(`RAY_MAX_QUERY_RETRIES`),而非 actor 级恢复。
- **资源感知调度** —— GPU / 内存 / object-store 准入控制 + 跨并发查询的 worst-fit 装箱,都在 Rust。
- **分布式数据源** —— 生成器驱动的流式扫描、分布式 Hive 读、分布式 Lance 写 + 分布式向量索引构建。

### 🔎 向量与全文检索 —— 规模化 RAG
**单机**(1M×768 维,top-100,实测 —— 见报告):

| 方法 | 召回 | QPS |
|---|---|---|
| 精确暴力(1 核) | 100% | 1.3 |
| `knn_rerank`(IVF + 精确重排) | 99.9% | 137 |
| `knn_ann_resident`(IVF 取 id + 内存重排) | 99.9% | **182** |

- **全部 Lance 索引类型**:IVF_FLAT / IVF_SQ / IVF_PQ / IVF_HNSW_{FLAT,SQ,PQ},支持索引常驻内存(`set_index_cache_size`)。
- **场景算子**:带过滤 ANN(元数据谓词下推进索引扫描)、范围/阈值检索(去重/实体消歧)、MMR 多样性重排。
- **全文**:Lance 倒排上的 BM25;**混合检索**(向量 + BM25,RRF 融合)。

**分布式检索**(扇出到分片,driver 归并):

| 方法 | 说明 |
|---|---|
| `distributed_knn_resident` | 精确,数据常驻 worker,100% 召回 |
| `distributed_knn_resident_batch` | 批量 —— 聚合 QPS 随 worker 近线性扩展 |
| `distributed_ann_knn` | 分片 ANN(每片自带索引)—— 十亿级 |
| `distributed_ann_knn_routed` | **簇路由** —— 查询只碰最近质心的那几个分片 |
| `distributed_fts` | 分片 BM25 全文 |
| `distributed_hybrid` | 分布式向量 + 全文,RRF 融合 |

批量吞吐近线性扩展(1→8 worker ≈ 5.3×);带过滤 ANN(`where=`)在分布式路径同样可用。完整基准矩阵(方法 × 索引 × worker)见 [`docs/vector_retrieval_report.zh.md`](docs/vector_retrieval_report.zh.md)。

### 🖼️ 多模态
原生 `TensorType`(Arrow `fixed_shape_tensor`)+ Image / Audio / Video / Document 列,配 Daft 风格的流式表达式 API 与批量解码器。

```python
import jude
rel = con.from_arrow(table)

# 流式多模态表达式 —— decode/resize/crop/encode 都在计划里
rel = rel.with_column("img", jude.mm("bytes").image.decode().image.resize(224, 224))
rel = rel.with_column("wav", jude.mm("audio").audio.decode(sample_rate=16000, mono=True))
```

- **图像**:`.image.decode().resize(w,h).crop(...).encode("PNG")` → 张量。
- **音频**:`.audio.decode(sample_rate=, mono=)`,带重采样。
- **视频**:抽帧 / 流式解码;**文档**:PDF → 逐页。
- **摄取源**覆盖 图像/音频/视频/文档,批量解码器(`decode_image_batch`、`decode_audio_batch`、`decode_video_batch`、`decode_document_batch`)—— 都可在 cosmos-xenna 上分布式跑。

### 🧹 LLM 数据治理 —— 核心差异点
决定训练集形态的算子,单机**和**分布式(`jude.curate` / `jude.curate_dist`):

- **去重** —— 精确(SHA-256)、模糊(MinHash-LSH,bands 按 Jaccard 阈值自动校准;分布式形态用全局连通分量做到与单机召回逐行一致)、语义(贪心非传递 SemDeDup + 分布式 k-means)、**精确子串**(Lee et al. 滚动哈希:剥离跨不同文档的共享段落/样板)。
- **网页语料清洗** —— C4 式行级过滤、跨文档行去重、mojibake 修复 + Unicode NFC 规范化。
- **质量过滤** —— Gopher/C4 启发式(停用词门、重复 n-gram、数字/符号比)、**模型质量分类器作为批处理 stage**(`jude.model_stage`:自带 CPU fastText/ONNX 模型、远程 vLLM/API 端点、或任意 callable —— GPU 可选)、token 感知长度门、语种识别。
- **PII** —— 检测/脱敏(邮箱、URL、IPv4、SSN、Luhn 校验的信用卡)。
- **去污染** —— 抗稀释的 benchmark 覆盖度重叠检测。
- **切块** —— 定长 / 递归,用于分词与 RAG。
- **训练就绪输出** —— `jude.tokenize`:分词(可插拔 byte / tiktoken / HF)、打包成定长序列(文档边界 + EOS)、写 token 分片(Lance 或可 memmap 的 `.bin`/`.idx.json`)。
- **画像** —— `jude.profile`:一遍扫出语料统计,HyperLogLog 近似基数(去重率)、长度分位、语种分布 —— 大规模下 O(1) 内存。

Rust 热点循环 + cosmos-xenna 阶段流水线;每个算子都有分布式形态。

### 💾 存储与版本
Lance(读 + 单机与分布式写、类 git 的 **branch/tag** 数据版本、向量 + 全文索引),以及 Hive、Iceberg 读。

### 📊 可观测性
无 GIL 的 Rust 指标注册表(按 查询/阶段/UDF/集群)、持久化审计日志(redb)、React 仪表盘。`python -m jude.observe` 提供服务并挂到 Ray 集群。端点:`/api/metrics`(完整 JSON 快照)、`/api/summary`(汇总:延迟 p50/p95/p99、rows/sec、任务进度、**治理保留率**)、`/api/prometheus`(Prometheus 文本 → Grafana)。**数据质量可观测性**:用 `observe.curate(op, rows_in)` 包住治理算子,即可按算子追踪 rows in→out / 去除数 / 保留率。

---

## 快速上手

```python
import jude

con = jude.connect()
con.sql("SELECT 42 AS answer").show()

# 数据治理:对语料做模糊去重
from jude import curate
clean = curate.fuzzy_dedup(corpus, threshold=0.7, bands=16)

# 向量检索:建索引,用内存重排查询
from jude import vector
jude.connect().create_lance_vector_index("emb.lance", "v",
                                          index_type="IVF_SQ", metric="cosine")
hits = vector.knn_ann_resident("emb.lance", "v", query, k=100, nprobes=16)
```

### 跑分布式(Ray)

把 jude 指到 Ray 集群(或让它起本地集群),**同一套**关系程序就并行执行 —— jude 负责分区、调度(在 Rust 里)、归并。**最简形式就是一条 SQL 字符串**:`runner.collect()` 会读这条查询的 stage DAG(来自 Rust 规划器)自动路由(`ORDER BY`→分布式排序、`DISTINCT`→分布式去重、scan/filter/project→并行分区扫描),不用你手挑分布式算子。

```python
import ray; ray.init(address="auto")          # 挂到你的集群(本地可省略)
from jude.runners.ray import RayRunner
runner = RayRunner(num_workers=8)              # GPU 阶段用 num_gpus_per_worker=1

# 一条 SQL 字符串,分布式执行(由 Rust stage 规划器自动路由)
table = runner.collect(con.sql("""
    SELECT category, count(*) AS n
    FROM read_parquet('s3://bucket/docs/*.parquet')
    WHERE year >= 2020
    GROUP BY category ORDER BY n DESC
"""))

# 关系 API 同理
rel = con.from_parquet("s3://bucket/docs/*.parquet").filter("year >= 2020")
table = runner.collect(rel)

# 也可直接驱动某个分布式算子
runner.distributed_sort(rel, ["score DESC"])
runner.distributed_top_k(rel, ["score DESC"], 100)
runner.distributed_write_lance(rel, "out.lance", mode="overwrite",
                               vector_index={"column": "v", "index_type": "IVF_SQ"})

# 分布式治理 + 向量检索用同一个 runner
from jude import curate_dist, vector
curate_dist.dist_fuzzy_dedup(table, runner=runner, threshold=0.7, bands=16)
vector.distributed_ann_knn(shard_paths, "v", query, k=100, runner=runner)

# 流式数据源也能分布式读
rel = jude.datasource.read(my_source, distributed=True)
```

也可以设一次进程级全局 runner,之后所有算子自动用它:
`jude.runners.set_runner_local(num_workers=8)`(或赋一个 `RayRunner`)。

### 多模态流水线(cosmos-xenna 阶段)

把 解码 + 治理 + 你的模型 串成**阶段(stage)**;本地开发的流水线(`engine="local"`,保序)可原样分布式跑在 cosmos-xenna 上(`engine="cosmos"`),每个阶段单独配 CPU/GPU/batch:

```python
from jude.pipeline import RelationPipeline
from jude.sources import ImageFileSource

out = (RelationPipeline.from_source(ImageFileSource("imgs/*.png"), engine="cosmos")
         .load_files()                              # 读字节,自成一个可扩展阶段
         .decode("image")                           # 字节 → 张量
         .quality_filter(min_words=8)               # 一个治理阶段
         .map_batches(embed_fn, gpus=1, batch_size=64)  # 你的 GPU 模型阶段
         .to_relation(con))                         # → 可查询的 jude Relation

out.filter("score > 0.9").to_arrow()               # 继续用 SQL/关系操作
```

开箱阶段:`load_files`、`decode("image"|"audio"|"video"|"document")`、`chunk`、`quality_filter`、`content_hash`,以及 `map_batches(fn)` 承接任意自定义(embedding / LLM 调用等 —— 模型自备)。

### 实时观测

启动指标 API + React 仪表盘 + Ray 仪表盘:

```bash
python -m jude.observe
```

---

## 构建与测试

环境:Rust(stable)+ Cargo,Python 3.14(venv 在 `.venv`),[`maturin`](https://www.maturin.rs/)。

```bash
source .venv/bin/activate
maturin develop --release        # 把原生扩展编进 venv
cargo check --lib                # 快速内循环类型检查(不做 Python 链接)
python -m pytest tests/ -q -m "not slow and not benchmark"
```

我们还把 **Vane 的测试套件原样**跑在 jude 上(`tests/vane_ported/`,框架把 `import duckdb` 别名到 `jude`);架构上无法对齐的用例记录在 `known_gaps.txt` 并自动 `xfail`,套件保持绿色,新失败即真回归。

基准在 `benchmarking/`(挂到常驻的 `python -m jude.observe` Ray,运行会出现在仪表盘)。

## 文档

- [`docs/vector_retrieval_report.zh.md`](docs/vector_retrieval_report.zh.md) —— 向量/全文检索:方法 × 索引 × worker,召回/延迟/QPS。
- [`docs/llm_data_operators.zh.md`](docs/llm_data_operators.zh.md) —— 每个治理算子怎么实现、有什么用。
- [`docs/distributed_and_udf_internals.zh.md`](docs/distributed_and_udf_internals.zh.md) —— Rust/Python 分界、`WorkerManager`、shuffle、UDF 池。
- [`docs/observability_and_ray_dashboard.zh.md`](docs/observability_and_ray_dashboard.zh.md) —— 指标、审计、Ray + jude 仪表盘。
- [`docs/billion_scale_vector_search.zh.md`](docs/billion_scale_vector_search.zh.md) · [`docs/vector_high_recall.zh.md`](docs/vector_high_recall.zh.md) —— 规模与召回深入。
- [`docs/cosmos_pipeline.md`](docs/cosmos_pipeline.md) —— cosmos-xenna 多阶段多模态流水线。

## 定位

jude 是**数据引擎**,**不是**推理服务:刻意止步于 LLM/embedding 推理边界(模型自备)。推理之前与周边的一切 —— 摄取、SQL 分析、治理、索引、检索、分布式 —— 都在范围内。核心判断:"LLM 数据"里大部分工作是数据工程,而这类工作需要一个把 SQL、治理算子、检索融合进同一个分布式运行时的引擎。
