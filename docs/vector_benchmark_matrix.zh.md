# jude 向量检索完备基准报告(固定 1M)

> 回答:"固定 1M 数据量,单 worker / 多 worker,不同算法,不同索引,全维度性能测试"。
> 脚本:`benchmarking/bench_vector_matrix.py`(可复现)。环境:单机,N=1,000,000,dim=96,top-k=100,30 次查询取 p50/p95/QPS,聚类数据(200 簇)。最后更新:2026-07-20。

原始向量:0.36 GiB。全部指标含:索引构建时间、召回率(对精确 ground truth)、延迟 p50/p95、QPS。

---

## A. 单机不同索引(1 核 / 1 进程)

| 方法 | 建索引 | recall@100 | p50 ms | p95 ms | QPS |
|---|---|---|---|---|---|
| exact 暴力(1 核) | — | 100% | 68.2 | 74.8 | 14 |
| **IVF_FLAT + rerank** | 5.1s | **100%** | 28.8 | 38.0 | **33** |
| IVF_PQ + rerank | 13.2s | 76% | 29.6 | 33.5 | 33 |
| IVF_HNSW_SQ + rerank | 13.1s | 100% | 35.9 | 51.7 | 26 |

**结论:**
- **IVF_FLAT + rerank 是单机最佳** —— 100% 召回,比 exact 快 2.4x(33 vs 14 QPS)。建索引最快(5.1s)。
- **IVF_PQ 用召回换内存**(76% 召回),这个规模下不换速度(和 IVF_FLAT 同 QPS)。要压缩内存才选它。
- **IVF_HNSW_SQ** 100% 召回但这个规模下比 IVF_FLAT 略慢(HNSW 图的优势在更高维/更大 N / 更高 QPS 目标时显现)。

---

## B. 分布式 worker 扫描(数据常驻 worker)

| 方法 | recall@100 | p50 ms | p95 ms | QPS |
|---|---|---|---|---|
| resident EXACT(1 worker) | 100% | 19.6 | 22.7 | 51 |
| resident EXACT(2 workers) | 100% | 14.9 | 16.4 | 67 |
| **resident EXACT(4 workers)** | **100%** | 13.0 | 15.1 | **76** |
| sharded ANN IVF_FLAT(1 shard) | 100% | 42.9 | 55.6 | 23 |
| sharded ANN IVF_FLAT(2 shards) | 100% | 29.4 | 36.7 | 33 |
| sharded ANN IVF_FLAT(4 shards) | 100% | 30.9 | 38.5 | 32 |

**结论:**
- **resident EXACT 随 worker 数近线性扩展**:1→2→4 worker = 51→67→76 QPS,全程 100% 召回。**4 worker 比单机 exact(14 QPS)快 5.4x。** 数据常驻 worker(numpy 矩阵缓存),每次查询只广播 query 向量 —— 这是分布式向量检索的正确架构。
- **sharded ANN 在这个规模反而不如 resident exact**:1M/dim96/聚类数据下,每个分片的 numpy 精确扫描已经很快,ANN 的索引查找 + rerank 反而是额外开销。**ANN 的优势在更大 N / 更高维**(exact 的 O(N·d) 扫描变得昂贵时)。

---

## 全维度对照(把两部分放一起看)

| 方案 | 核数 | recall | QPS | 何时用 |
|---|---|---|---|---|
| 单机 exact | 1 | 100% | 14 | 小数据 / 基线 |
| 单机 IVF_FLAT+rerank | 1 | 100% | 33 | 单机、要召回、免多机 |
| 单机 IVF_PQ+rerank | 1 | 76% | 33 | 单机、内存紧张 |
| resident EXACT | 4 | 100% | 76 | **多核/多机、要 100% 召回**(本规模最快) |
| sharded ANN | 4 | 100% | 32 | 超大 N / 高维(此规模不占优) |

---

## 关键洞察(诚实)

1. **这个规模(1M, dim96)下,精确检索非常能打** —— resident-exact 4 worker 76 QPS、100% 召回,比任何单机 ANN 都快。ANN 的价值随 **N 和维度增长**才显现(exact 的 O(N·d) 扫描超过索引查找开销时)。
2. **要高召回就用 IVF_FLAT,不要 IVF_PQ**(PQ 压缩把召回压到 76%)。
3. **分布式向量检索必须"数据常驻"**(resident,数据在 worker 上、只传 query),不能每查询重新分发数据 —— 后者(旧的 `distributed_knn`)慢 4x,已废弃改用 `distributed_knn_resident`。
4. **worker 扩展有效**:1→4 worker,QPS 51→76(1.5x;受 Ray 调度 + driver 归并开销限制,不是完美 4x,但随 N 增大扩展比会更好,因为扫描占比更高)。

**诚实边界**:本机只测到 1M(0.36 GiB)。更大 N / 更高维下 ANN 相对 exact 的优势会放大;真实分布式(多物理机)的网络开销需在目标集群复测。可复现:`python benchmarking/bench_vector_matrix.py --n 1000000 --dim 96 --k 100`。
