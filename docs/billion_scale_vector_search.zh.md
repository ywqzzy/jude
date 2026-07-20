# 十亿级向量检索设计:1B 向量召回 top-100w

> 回答:"10 亿数据召回 100 万,用什么算法" —— 沉淀为代码(`jude.vector.distributed_ann_knn`)+ 测试(`tests/test_vector_sharded.py`)+ 本报告。
> 最后更新:2026-07-20。

---

## 一、为什么不能暴力检索

10 亿(1B)向量,dim=768,float32:
- **存储**:1e9 × 768 × 4 B ≈ **2.86 TB**。单机内存放不下。
- **暴力检索每次查询**:算 1B 次距离 ≈ 数十秒~分钟级,且要扫 2.86 TB。就算分布式暴力(每 worker 扫自己分片),100 台机器每台还要扫 28 GB/查询 —— 对高 QPS 不可行。
- 单个整体索引也放不下一台机器。

**结论:必须"分片 + 每片本地 ANN 索引 + 分布式归并"。**

---

## 二、算法:分布式分片 ANN(distributed sharded ANN)

这是 Milvus / Vespa / 各大厂十亿级向量库的通用架构,jude 用 `distributed_ann_knn` 实现:

```
                          query q, k=100万
                               │  广播
        ┌──────────────┬───────┴───────┬──────────────┐
        ▼              ▼               ▼              ▼
   shard_0        shard_1         shard_2   ...   shard_S     (各在一台机器)
   本地IVF索引     本地IVF索引       本地IVF索引        本地IVF索引
   local top-k    local top-k     local top-k     local top-k  (两阶段:ANN多取+精排)
        └──────────────┴───────┬───────┴──────────────┘
                               ▼ driver 归并
                    global top-k (ORDER BY dist LIMIT k)
```

**三步:**
1. **建库(一次性)**:把 1B 向量分成 S 片(如 S=1000,每片 100 万),每片写成一个 Lance 数据集并建**本地 IVF 索引**(每片索引能放进一台机器内存)。
2. **查询 map**:query 广播到所有 S 片,每片用自己的索引做**两阶段 ANN**(over-fetch 多取候选 + 精确 rerank),返回**本片 local top-k**。
3. **归并 reduce**:driver 收集 S 个 local top-k,concat,全局 `ORDER BY distance LIMIT k`,得 global top-k。

**代码:**
```python
from jude import vector
hits = vector.distributed_ann_knn(
    shard_paths,          # [每片一个已建索引的 Lance 路径] × S
    column="v", query=q, k=1_000_000,
    overfetch=3,          # 每片多取 k*overfetch 候选再精排(调召回)
    nprobes=1024,         # 每片扫多少 IVF 单元(调召回)
    metric="cosine",
)
```

**正确性/召回**:若每片 ANN 能召回它自己真实的 local top-k,归并就是精确的全局 top-k。ANN 是近似的,所以靠 `overfetch`/`nprobes` 把每片召回抬上去;整体召回 ≈ 每片召回(实测聚类数据 IVF_FLAT ≥ 90%,可调更高)。

---

## 三、top-100w 这个"大 k"的特殊处理

k=100 万很大,归并阶段会收到 `S × k` 个候选(S=1000 → 10 亿候选,又爆了)。两种应对:

1. **每片只返回 `k/S × 安全系数` 而非完整 k**:若数据在片间均匀分布,全局 top-100w 里每片平均贡献 `100w/S`。每片返回 `ceil(k/S × overfetch_merge)`(如 S=1000 → 每片返回 2000~5000),归并 S×5000=500 万候选 → 全局 top-100w。**大幅减小归并量,轻微牺牲召回**(某片贡献超均值时会漏)。
2. **分层归并(tree merge)**:S 片先两两/分组归并成中间 top-k,再归并,避免 driver 单点收 S×k。

jude 现在的 `distributed_ann_knn` 用方案 1 的简化版(每片 top-k,driver 归并)——适合 k 不是极端大、或 S 不是极端多的场景。极端 1B→100w 时,把每片的 `k` 设成 `k/S` 的几倍即可(见参数调优)。

---

## 四、参数怎么调(1B → 100w 场景)

| 参数 | 含义 | 1B→100w 建议 |
|---|---|---|
| **S(分片数)** | 向量分成几片 | 每片 ≤ 一台机器内存能放的索引,如 100万~500万/片 → S=200~1000 |
| **num_partitions**(每片 IVF) | 每片索引的 IVF 单元数 | ≈ √(每片行数),100万/片 → ~1000 |
| **index_type** | 每片索引类型 | **IVF_FLAT**(要召回) / IVF_PQ(省内存、召回低) / IVF_HNSW_SQ(更快) |
| **每片返回 k'** | 每片 local top-k' | `ceil(k/S) × 3~5`(均匀分布时);数据倾斜就调大 |
| **nprobes** | 每片扫多少单元 | 从 √(num_partitions) 起,不够就加;越大召回越高越慢 |
| **overfetch** | 每片 ANN 多取候选再精排 | 3~10,越大召回越高 |

**调优流程**:用 `vector.recall_at_k(结果, 小规模精确ground_truth)` 在一个可暴力的子集上量化召回,把 nprobes/overfetch/每片k' 调到达标,再上全量。

---

## 五、为什么 jude 适合做这个

- **分片调度在 Rust WorkerManager**:片→worker 的分配、并发窗口、归并都走现成的分布式基建,不占 GIL。
- **每片索引用 Lance**:Lance 的 IVF 索引是 Rust 实现,建/查都快;支持 append + compact + 版本,库可增量维护。
- **两阶段 rerank 现成**:`knn_rerank`(over-fetch + 精确 rerank)已实现并测过高召回。
- **fan-out/merge 现成**:`distributed_ann_knn` 复用 RayRunner 的 worker 池 + object-store 归并。

---

## 六、实测(可运行规模,同一算法)

无法在本机跑 1B,但**同一套代码**在可运行规模验证了正确性与召回:

| 项 | 值 |
|---|---|
| 规模 | 4 片 × 5 万 = 20 万向量,dim=32,聚类数据 |
| 每片索引 | IVF_FLAT,num_partitions=√(5万)≈224 |
| 查询 | top-k=500,overfetch=5,nprobes=224 |
| 结果 | **召回 ≥ 90%**(vs 全局精确 ground truth),结果按距离升序 |
| 证据 | `tests/test_vector_sharded.py`(绿) |

架构完全一致:把 4 片换成 1000 片、每片 5 万换成 100 万、分布到多机,就是 1B→100w。分片是独立的,水平扩展。

---

## 七、一句话总结

**1B → top-100w = 分布式分片 ANN**:S 个分片各建本地 Lance IVF 索引 → 查询广播 → 每片两阶段 ANN 出 local top-k' → driver 归并全局 top-k。jude 的 `vector.distributed_ann_knn` 已实现;召回靠每片的 `overfetch`/`nprobes`/`k'` 调,用 `recall_at_k` 量化。要 100% 召回且数据 ≤ 百万级就别用 ANN、直接 `high_recall_knn` 暴力;十亿级必须走这条分片 ANN。

---

## 八、科学基准测试(可复现)

脚本:`benchmarking/bench_billion_scale.py`。方法:**40 次查询**取均值(非单次),报 recall@k(对精确 ground truth)、延迟 p50/p95、QPS,扫 index_type × nprobes × overfetch。环境:N=500,000,dim=96,top-k=100,聚类数据,单机。

**精确基线**:recall=100%(定义),p50=10.6ms,QPS=93。

**IVF_FLAT(num_partitions=707,建索引 2.5s):**

| nprobes | overfetch | recall@100 | p50 ms | p95 ms | QPS |
|---|---|---|---|---|---|
| 44 | 1 | **100.0%** | 9.2 | 19.4 | 94 |
| 44 | 5 | 100.0% | 21.1 | 35.9 | 44 |
| 176 | 1 | 100.0% | 23.0 | 66.3 | 33 |
| 707(全部) | 1 | 100.0% | 87.7 | 147 | 11 |

**IVF_PQ(num_partitions=707,建索引 12.0s):**

| nprobes | overfetch | recall@100 | p50 ms | p95 ms | QPS |
|---|---|---|---|---|---|
| 44 | 1 | 41.5% | 6.6 | 7.2 | 150 |
| 44 | 5 | **86.5%** | 16.8 | 32.2 | 55 |
| 176 | 5 | 86.5% | 24.4 | 44.7 | 37 |
| 707 | 5 | 86.5% | 58.0 | 77.7 | 17 |

**科学结论:**
1. **IVF_FLAT 在所有 nprobes 下 recall=100%** —— 因为它存**未压缩**向量,候选质量高,over-fetch+精确 rerank 稳定命中真 top-k。且 nprobes=44 时 p50 仅 9.2ms、QPS 94(与精确相当),而它是 O(nprobes×单元大小)、随 N 增长远慢于精确的 O(N) —— **N 越大,IVF_FLAT 相对精确的优势越大**。这是十亿级要它的原因。
2. **IVF_PQ recall 封顶 86.5%**(即使 overfetch=5) —— PQ 压缩向量,候选有损,rerank 也救不回。换来更省内存(压缩)+ overfetch=1 时更高 QPS(150)。**要召回选 IVF_FLAT;要极致省内存/吞吐、能接受 ~85% 召回选 IVF_PQ。**
3. **recall-latency 权衡清晰**:overfetch/nprobes 上调 → recall 上升、QPS 下降。用 `recall_at_k` 在这条曲线上选点。
4. **1B 外推**:dim=768 时 1B = 2.79 TiB,单机放不下 → 必须分片(S=1000×100万 或 200×500万,每片 3~14 GiB 可进单机),每片按上表调 IVF_FLAT,fan-out + 归并。

**诚实说明**:本机只能到 50 万级(0.18 GiB);同一套代码与参数直接外推到分片后的 1B。真实 1B 数还依赖网络/磁盘/embedding 分布,需在目标集群复测。这里给的是**算法正确性 + 单片 recall/latency 曲线 + 分片外推**,不是 1B 实测数(诚实标注)。
