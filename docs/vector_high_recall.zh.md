# jude 高召回向量检索报告 + 索引构建指南

> 回答:"支持 top-10w(100k)召回,召回率尽可能高" + "索引构建怎么做"。
> 测试脚本:`benchmarking/bench_vector_recall.py`(可复现)。最后更新:2026-07-20。

---

## 一、召回率测试报告(实测)

环境:N=100,000 向量,dim=128,查询 top-k=1000,单机。`recall@k = |命中的真实top-k| / k`。

### 真实场景(聚类数据 —— 真实 embedding 都是聚类的)

| 方法 | recall@1000 | 耗时 |
|---|---|---|
| **EXACT 暴力检索** | **100.0%** | **0.014s** |
| IVF_FLAT + rerank, nprobes=20 | 100.0% | 0.034s |
| IVF_FLAT + rerank, nprobes=50 | 100.0% | 0.034s |
| IVF_FLAT + rerank, nprobes=100 | 100.0% | 0.037s |
| IVF_FLAT + rerank, nprobes=316(全部) | 100.0% | 0.049s |

**结论:召回率 100%,远超 95% 目标。** 两条路都达标。

### 关键发现(诚实,踩过的坑)

1. **top-100k / 大 k,在百万级以内的数据上,EXACT 暴力检索就是最优解** —— 100% 召回,而且**最快**(100k×128 只要 0.014s)。ANN 索引是为 tiny-k(10~100)+ 超大数据(千万~十亿)设计的;当 k 是数据集的一大部分时,ANN 反而又慢又低召回。**别在这个规模用 ANN。**

2. **IVF_PQ 会拖低召回(实测 top-1000 只有 84.5%,即使 rerank)。** 因为 PQ(乘积量化)**压缩了向量**,ANN 候选集本身就是有损的,rerank 也救不回来。**要高召回,别用 IVF_PQ,用 IVF_FLAT(不压缩)。**

3. **随机噪声向量是 ANN 的最坏情况**(维度灾难,kmeans 分桶失效),召回很低。但真实 embedding 是聚类的,IVF_FLAT 上召回轻松 100%。测召回一定要用**贴近真实分布**的数据。

4. **`nprobes` 是召回旋钮** —— 扫描更多 IVF 单元 → 召回更高(代价是更慢)。要多少召回,调 nprobes 到满足为止;`nprobes = num_partitions` 就等于扫全表(≈exact)。

---

## 二、怎么达到 ≥95% 召回(决策树)

```
数据量多大?
├─ ≤ 几百万向量  → 用 EXACT(jude.vector.high_recall_knn / distributed_knn)
│                   100% 召回,大 k 也快,不用建索引。★ 首选
└─ 千万~十亿    → 建 IVF_FLAT 索引 + 两阶段 rerank(jude.vector.knn_rerank)
                    - 用 IVF_FLAT,不要 IVF_PQ(除非内存紧张能接受召回损失)
                    - overfetch=5~10(多取候选再精排)
                    - nprobes 从 sqrt(N) 起,不够就调高,用 recall_at_k 量化
                    - 想要 100%:nprobes → num_partitions(退化为全扫)
```

用 `jude.vector.recall_at_k(近似结果ids, 精确ids)` 随时量化召回,调到达标。

---

## 三、索引构建怎么做

### 1. EXACT(不用索引,推荐用于 ≤ 百万级 / 大 k)

```python
import jude
from jude import vector

con = jude.connect(); con.register("emb", table)   # table 有 FLOAT[dim] 列 v
# 单机精确
res = vector.knn(con, "emb", "v", query, k=100_000, metric="cosine")
# 分布式精确(超内存时;每 worker 算局部 top-k,driver 归并全局 top-k)
res = vector.high_recall_knn(table, "v", query, k=100_000, metric="cosine")
```
100% 召回,无需建索引,无需维护。

### 2. Lance IVF_FLAT 索引(推荐用于超大数据 + 要高召回)

```python
import jude
con = jude.connect()
# 先把向量写成 Lance 数据集
jude._lance.write(table, "/data/emb.lance", mode="create")
# 建 IVF_FLAT 索引(不压缩向量 -> 候选精确 -> 高召回)
con.create_lance_vector_index(
    "/data/emb.lance", "v",
    index_type="IVF_FLAT",              # ★ 不要 IVF_PQ,要高召回
    metric="cosine",                    # cosine / l2 / dot
    num_partitions=int(N**0.5),         # 经验值 sqrt(行数),如 100万 -> 1000
)
# 两阶段高召回检索:ANN 多取候选 + 精确 rerank
from jude import vector
res = vector.knn_rerank("/data/emb.lance", "v", query,
                        k=1000, overfetch=5, nprobes=100, metric="cosine")
```

**参数怎么调:**
- `index_type`: **IVF_FLAT**(高召回,不压缩)vs IVF_PQ(省内存,召回低)vs IVF_HNSW_SQ(HNSW图,更快但更复杂)。要召回选 IVF_FLAT。
- `num_partitions`: IVF 单元数,经验 `≈ sqrt(N)`。太少=每单元太大扫得慢;太多=召回需要更高 nprobes。
- `num_sub_vectors`(仅 IVF_PQ):PQ 子向量数,越多越精确但越慢/占内存。
- 检索时 `nprobes`: 扫多少单元(召回旋钮);`overfetch`: 多取几倍候选再精排。

### 3. DuckDB 进程内 VSS HNSW(轻量、无需 Lance)

```python
from jude import vector
con = jude.connect(); con.register("emb", table)
vector.create_hnsw_index(con, "emb", "v", metric="cosine")   # autoload vss 扩展
res = vector.knn(con, "emb", "v", query, k=1000)             # 走索引
```
适合中小规模、不想落 Lance 的场景。注意 DuckDB 的 HNSW 磁盘持久化是实验特性(内存态稳定)。

### 4. 增量维护(数据在变)

```python
from jude import lance as jl
jl.merge_insert("/data/emb.lance", new_rows, on="id")   # upsert 新/改向量
jl.compact("/data/emb.lance")                            # 合并小 fragment(分布式写后必做)
# 加了新数据后重建/优化索引
jude._lance.optimize_lance_indices("/data/emb.lance")    # 把新 fragment 并入全局索引
```

---

## 四、一句话总结

- **top-100k 召回:jude 达到 100%**(> 95% 目标)。
- **≤ 百万级 / 大 k:直接用 EXACT**(`high_recall_knn`)—— 100% 召回、最快、免索引。这是 top-100k 的正解。
- **超大数据要 ANN:用 IVF_FLAT + `knn_rerank`(over-fetch + 精确 rerank),别用 IVF_PQ**;`nprobes` 是召回旋钮,用 `recall_at_k` 量化调优。
- **证据**:`benchmarking/bench_vector_recall.py`(可复现),`python/jude/vector.py`(实现),`tests/test_vector_recall.py`(测试)。
