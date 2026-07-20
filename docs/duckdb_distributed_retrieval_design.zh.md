# 设计:DuckDB 上的分布式全文/向量检索 + 混合分析查询

> 状态:**设计,未实现**(按用户要求先设计)。本文给出把检索(FTS + 向量)与分析(AP:join/聚合/过滤/窗口)融合进**同一个 DuckDB 查询计划**、并分布式化的方案。**不 fork DuckDB** —— 全部用 DuckDB 的公开扩展 API + Arrow C-Stream,在 jude 自己的 Rust 里实现。

## 1. 为什么走 DuckDB 这条路(比 Lance-only 强在哪)

现在 jude 的检索(`jude.vector` / `jude.lance`)是**纯检索**:索引出 top-k,返回一张表。要再做"按 category 过滤、按 author 聚合、和用户表 join"就得在 Python/另一个引擎里接着算。

**DuckDB 路线的价值:检索结果直接进 SQL 计划**,和 join/聚合/窗口/过滤在**一个查询优化器**里融合 —— 这是"混合分析检索"(hybrid analytical retrieval):

```sql
-- 检索 + 分析一次完成:相似文档里,按作者聚合、只看近三年、关联用户表
SELECT u.org, count(*) n, avg(d._score) mean_rel
FROM   jude_search('docs', 'v', :query_vec, k => 500) d      -- 检索(见 §3)
JOIN   users u ON u.id = d.author_id
WHERE  d.year >= 2023
GROUP  BY u.org ORDER BY mean_rel DESC LIMIT 20;
```

Lance 做不到把这步揉进一个计划;DuckDB 可以。这就是"更强"的含义。

## 2. 基石:`lance_scan()` table function + 下推(不 fork)

jude 已有 `src/mat_scan.rs` —— 一个**自己写的 DuckDB table function**(over Arrow batches),用的是 DuckDB 公开扩展 API。把它扩成 `lance_scan(path)`:

- DuckDB 通过 `bind`/`init` 回调把**投影下推**(要哪些列)和**过滤下推**(`WHERE year>2020`)交给我们的 Rust table function;
- 我们翻译成 Lance `scanner(columns=..., filter=...)`,让 **Lance 只读需要的列、只返回匹配行**;
- 结果以 RecordBatch 流回 DuckDB。

于是 `SELECT ... FROM lance_scan('emb.lance') WHERE ... JOIN ...` 里,**DuckDB 跑 AP,Lance 负责 IO + 列裁剪 + 谓词下推**,零 fork。这是所有后续能力的地基。

> 注意:向量/FTS **索引**(IVF/倒排)DuckDB 的 scan 用不上(它不懂 Lance 索引)。所以检索要走**两段式**(§3),不能指望纯 scan 走索引。

## 3. 检索算子:两段式(索引出候选 → SQL 分析)

由于 DuckDB 看不到 Lance 索引,检索用**两段式**,包装成 DuckDB table function `jude_search(...)` / `jude_fts(...)`:

1. **阶段 1(索引,Rust 侧):** 调 Lance 的 ANN(IVF)或 FTS(倒排)出候选 `id + _score/_distance`(复用 `jude.vector.knn_*` / `_lance.full_text_search`,只取 id + 分数,不搬向量)。
2. **阶段 2(DuckDB):** 把候选作为一张表,交给 DuckDB 与其他表 join / 聚合 / 过滤。

两种落地:
- **A. table-function 形态**:`jude_search('docs','v',:q,k=>500)` 内部执行阶段 1,产出候选行,DuckDB 接着算。用户写一条 SQL。
- **B. 显式两步**:Python 先 `vector.knn_ann_resident(...)` 得候选,`con.register("cand", cand)`,再 `con.sql("SELECT ... FROM cand JOIN ...")`。今天就能用(无需新代码),是 A 的手动版 / MVP。

DuckDB 原生的 `vss`(HNSW)/`fts` 扩展也可用于**纯 DuckDB 内**的小规模检索(数据已在 DuckDB 表里),作为不依赖 Lance 的另一条腿;但持久化 HNSW 在 DuckDB 仍是实验特性,大规模仍走 Lance 索引 + 两段式。

## 4. 分布式:复用 jude 现有的 Rust 编排

分布式不新造轮子 —— 复用 `RayRunner` + Rust `WorkerManager` + stage-DAG:

**分布式混合查询的执行形态(scatter-gather + AP):**
```
        ┌── worker: jude_search 本地分片 → 候选(id+score) + 本地谓词下推 ──┐
query ──┤   ...（每分片一个,Rust 调度路由;向量可簇路由,只碰相关分片）    ├── driver:
        └── worker: ...                                                     ┘   concat 候选
                                                                                → DuckDB:
                                                                                  JOIN 维表
                                                                                  + GROUP BY
                                                                                  + 全局 top-k
```

- **检索分片**:复用 `distributed_ann_knn` / `distributed_ann_knn_routed` / `distributed_fts`(已实现)出全局候选。
- **分析在 driver**:候选量小(k×shards),在 driver 的**单机 DuckDB** 里做 join/聚合/过滤 —— 大多数 RAG-分析场景够用。
- **分析也需要分布式时**(候选很大 / 维表很大):把阶段 2 的 SQL 交给现有 **stage-DAG 执行器**(shuffle-join / 两阶段聚合),候选表作为其中一个 shuffle 输入。这一步是把"检索候选"接到"分布式 SQL"的管道上,`StagePlanner` 已能切分。

**谓词下推到分片**:`where=` 已在 `distributed_ann_knn(where=...)` 打通(Lance prefilter),所以 "相似 AND category=x" 的过滤在分片内、进索引扫描时就完成,不必等到 driver。

## 5. Rust / Python 分界(遵循"编排在 Rust")

| 组件 | 位置 |
|---|---|
| `lance_scan` table function + 投影/谓词下推翻译 | **Rust**(`src/mat_scan.rs` 扩展,DuckDB 公开 API) |
| `jude_search`/`jude_fts` table function(两段式包装) | **Rust** 壳 + 调用现有检索(Lance 侧) |
| 分片路由 / scatter-gather 调度 | **Rust**(`WorkerManager`,已存在) |
| 候选归并 + 阶段 2 SQL 规划 | Rust `StagePlanner`(已存在)+ DuckDB 优化器 |
| Python | 只转发 RPC + 提供 `con.sql(...)` 入口 |

## 6. 分期

- **P0(✅ 已实现,`jude.retrieval`)**:形态 B —— `retrieval.search_then_sql(con, sql, candidates=...)` 把检索结果(`vector.*` / `distributed_*` / FTS)注册为命名关系,再跑引用它的 SQL;`retrieval.hybrid_analytical(con, path, sql, vector_query=/text_query=, ...)` 是常见 RAG-分析场景的便捷封装(单机走 `knn_rerank` 带 payload 列,分布式走分片检索)。检索与 join/聚合/过滤在一个 DuckDB 计划里融合。
- **P1**:`lance_scan(path)` table function + 投影/谓词下推(把 `read_lance` 从"全量物化"升级为"流式 + 下推")。
- **P2**:`jude_search`/`jude_fts` table function(一条 SQL 完成两段式)。
- **P3**:分布式混合查询把阶段 2 接入 stage-DAG(候选表作为 shuffle 输入),支持大维表分布式 join/聚合。

## 7. 诚实的边界

- **索引不经 scan**:DuckDB scan 用不上 Lance 的 IVF/倒排,检索必须两段式;`lance_scan` 只加速"扫描 + 过滤 + 列裁剪",不是"带索引的检索"。
- **FTS 的 IDF 是分片本地的**(分布式 BM25 的固有近似),精确全局 IDF 需要一次 term-stat 预扫;RAG 召回场景通常可接受。
- **DuckDB 原生 HNSW 持久化仍实验**;生产大规模走 Lance 索引。
- **P0 的候选归并在 driver 单机**;候选/维表大到单机吃不下才需要 P3 的分布式阶段 2。

## 8. 一句话

用 **`lance_scan` + 两段式 `jude_search`**,把 jude 的检索结果变成 DuckDB 计划里的一张表,让 **join/聚合/过滤/窗口和检索在一个优化器里融合**;分布式复用现有 `WorkerManager` + stage-DAG。全程不 fork DuckDB。这比 Lance-only 的纯检索多了"分析"这一维,正是"更强"的地方。
