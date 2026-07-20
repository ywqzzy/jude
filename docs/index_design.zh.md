# jude 的向量检索与倒排索引（设计）

状态：设计。jude 今天**没有**向量索引、没有 ANN 检索、没有倒排/全文索引——一次相似度查询目前是全表暴力
扫描。本文设计如何补上这块，且遵循 jude 的一贯路线：先榨干原生 DuckDB（`vss` / `fts` 扩展在钉的版本上
都能 `INSTALL`+`LOAD`，已实测），DuckDB 覆盖不到的分布式与 Rust 原生部分再自己建。文风与
`distributed_design.md` 对齐——讲清取舍、诚实标注硬骨头，不堆表格。

## 先说清楚：免费能拿到什么

我在钉的 DuckDB 版本上实测过，三件事开箱可用：

- **`vss` 扩展**：`CREATE INDEX ... USING HNSW(embedding)` + `array_distance(e, q)`。实测
  `SELECT id FROM v ORDER BY array_distance(e, q) LIMIT k` 在建了 HNSW 索引后能出 top-k。**限制**：HNSW
  的持久化要开 `hnsw_enable_experimental_persistence=true`（顾名思义，实验性），且它是**单节点、单表**的
  索引——一个 DuckDB 连接内的一张表。
- **`fts` 扩展**：`PRAGMA create_fts_index(table, id, body)` 建倒排索引，`match_bm25(id, 'query')` 出
  BM25 分数。实测能对 'fox' 命中并给分。**限制**：它是基于宏展开的倒排索引，`create_fts_index` 是一次性
  快照（新数据要重建），同样单节点单表。
- **暴力基线**：`array_distance` / `list_cosine_similarity` 本身不需要索引——`ORDER BY array_distance
  LIMIT k` 就是精确的暴力 KNN，正确但 O(N)。

所以 jude 的策略分层：**暴力（永远正确的基线）→ DuckDB 单节点索引（vss/fts，快、免费）→ jude 分布式/
Rust 原生索引（DuckDB 覆盖不到时）**。

## 向量检索

### API

挂在 relation 上，与多模态表达式同一风格：

```python
rel.vector_search(column, query, k=10, metric="l2"|"cosine", filter=None)
```

它返回按距离升序的前 k 行（外加一个 `_distance` 列），能继续用 SQL 组合。`column` 是一个
`FLOAT[d]` / 张量列——正是多模态表达式层（`embed_image` 等）产出的东西，于是"图片/文本 → 嵌入列 → 建索引
→ 检索"端到端成立。

### 三条执行路径

1. **暴力（基线，先实现）**：直接下降为
   `SELECT *, array_distance(col, q) AS _distance FROM (下层) ORDER BY _distance LIMIT k`。纯 SQL，
   跑在原生 DuckDB 上，无新机制，永远正确。这是 `vector_search` 的默认与正确性对照。
2. **DuckDB HNSW（单节点加速）**：当用户在列上 `rel.create_vector_index(column, metric)` 后，jude 在
   底层 DuckDB 连接上建 `USING HNSW`，`vector_search` 改走索引。这把 O(N) 降到近似对数，代价是近似
   （召回<100%）与索引构建/内存。适合数据能放进单节点的情形。
3. **jude 分布式索引（DuckDB 覆盖不到时）**：数据跨分区分布时，没有一个全局 DuckDB 索引。做法是
   **每分区建索引、查询 scatter-gather**：在每个 Ray worker 上对本分区建索引（DuckDB HNSW，或一个
   Rust 原生 HNSW/IVF over Arrow 嵌入），查询时把 query 广播到所有分区、各出本地 top-k、driver 归并成
   全局 top-k。归并是"k 路有序合并取前 k"，是纯决策，放 Rust（复用 `WorkerManager` 的分区/派发）；每
   分区的本地检索是执行，跑在 worker 上。这与两阶段聚合是同一形态（partial→merge）。

### ANN + 过滤这个硬问题（必须诚实）

带过滤的向量检索有个经典陷阱：`WHERE category='cat' ORDER BY distance LIMIT k`。
- **先 ANN 后过滤（post-filter）**：先从 HNSW 取 top-k，再过滤 category——可能 k 个里没几个是 'cat'，
  召回崩掉。
- **先过滤后 ANN（pre-filter）**：先按 category 筛，再在子集上检索——但 HNSW 索引是在全量上建的，没法
  只在子集上走图。
DuckDB vss 目前对带过滤的 ANN 支持有限，这块 jude 要么退回暴力（过滤后子集小时最简单且正确），要么建
**带属性的 IVF**（按 category 分桶）——本文先明确：带过滤时默认退回"过滤后暴力"，纯 ANN 走索引，带过滤的
索引加速是后续（且这正是 Rust 原生 IVF 的动机）。不假装 HNSW+任意过滤已解决。

## 倒排索引 / 全文检索

### API

```python
rel.create_fts_index(column)                       # 建倒排索引
rel.full_text_search(column, query, k=None)        # 按 BM25 排序返回命中行 + _score
```

### 执行路径

1. **DuckDB fts（单节点，先实现）**：`create_fts_index` 下降为 `PRAGMA create_fts_index(...)`，
   `full_text_search` 下降为 `... match_bm25(id, 'query') AS _score WHERE _score IS NOT NULL ORDER BY
   _score DESC`。分词、BM25 打分全由扩展做。**诚实限制**：一次性快照索引（增量数据要重建）、单表。
2. **分布式**：与向量检索同构——每分区建 fts 索引、查询 scatter、各出本地 BM25 top-k、driver 归并。BM25
   分数跨分区可比（同一打分函数），归并即按分数取全局前 k。
3. **jude Rust 原生倒排（DuckDB fts 不够时）**：若要增量更新、或与向量做混合检索，则在 Rust 里建
   term→postings 的倒排结构 over Arrow 文本列。这是较大工程，列为后续，只在 DuckDB fts 的一次性快照
   限制成为瓶颈时才做。

## 混合检索（向量 + 全文），为什么值得

多模态 + RAG 的真实需求是**混合**：既要语义相近（向量）又要关键词命中（BM25）。有了上面两条，混合就是在
关系层融合两个 top-k：各取 top-k，用 RRF（reciprocal rank fusion）或加权分合并。因为两者都产出
`(id, score)` 列、都在关系层落地，融合是一次普通的 join+排序，不需要新机制——这正是"两者都建在物化边界/
关系层之上"的红利。

## 与多模态嵌入路径的衔接

端到端：`ImageFileSource → .image.decode() → embed(model)（多模态 UDF，产出 FLOAT[d] 列）→
create_vector_index → vector_search(query_embedding, k)`。向量检索消费的正是多模态表达式层产出的嵌入列，
两条线在"嵌入列"处对接，无缝。

## Rust 与 Python 的线

- **决策在 Rust**：分布式检索的 scatter-gather 归并（k 路合并取 top-k）、分区/派发（复用 WorkerManager）、
  以及（后续）Rust 原生 HNSW/IVF/倒排结构本身。
- **执行/胶水**：建 DuckDB 索引与 `array_distance`/`match_bm25` 查询是 SQL（DuckDB 执行）；扩展的
  `INSTALL`/`LOAD` 是 Python/连接层薄封装。

## 分阶段计划

- **P1 暴力向量检索**：`vector_search` 下降为 `array_distance ORDER BY LIMIT`，正确性基线，纯 SQL。
- **P2 DuckDB 单节点索引**：`create_vector_index`→HNSW、`create_fts_index`/`full_text_search`→fts；带
  过滤默认退回过滤后暴力。
- **P3 分布式 scatter-gather**：每分区索引 + Rust 归并 top-k（向量与全文同构），复用 WorkerManager。
- **P4 混合检索**：关系层 RRF/加权融合。
- **P5 Rust 原生索引**：HNSW/IVF + 增量倒排 over Arrow，解决 DuckDB 的持久化/增量/带过滤限制——最大工程，
  最后做，且只在前面几层的限制成为真实瓶颈时。

## 测试

小规模合成向量（几百条已知最近邻）验证暴力与 HNSW 的召回/正确性；小文档集验证 BM25 命中与排序；分布式路径
断言 scatter-gather 结果与单节点暴力一致（top-k 集合相同）。门禁：`cargo test` + `pytest`，全套 0 失败。
