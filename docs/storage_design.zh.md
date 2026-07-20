# jude 的表格式存储：Iceberg 与 Paimon（设计）

状态：设计。用户的定位是：**jude 不只是计算引擎，还是分布式写入引擎**。所以本文既覆盖**读**（把 Iceberg /
Paimon 表扫进 jude relation），也覆盖**分布式写**（jude 跨 Ray worker 并行写出 Iceberg / Paimon 表，并完成
表格式的提交协议）。路线仍是先榨干原生 DuckDB、覆盖不到的自己建；文风与 `distributed_design.md` 对齐。

## 先说清楚：免费能拿到什么（实测）

我在钉的 DuckDB 版本上实测过：

- **Iceberg 读**：`INSTALL iceberg; LOAD iceberg` 可用，提供 `iceberg_scan`、`iceberg_snapshots`、
  `iceberg_metadata`、`iceberg_column_stats`、`iceberg_partition_stats` 等——都是**读侧**函数。所以
  "扫一张 Iceberg 表"几乎白拿。
- **Iceberg 写**：DuckDB 的 iceberg 扩展目前**没有**写提交（没有 `COPY TO iceberg`）。但 DuckDB 能写
  Parquet 数据文件（实测 `COPY (…) TO 'x.parquet' (FORMAT parquet)` OK）。也就是说：**数据文件 jude 能自己
  写，但 Iceberg 的提交协议（manifest + metadata.json + 原子切换）DuckDB 不做，jude 得做。**
- **Paimon**：DuckDB 没有 Paimon 扩展。读要么直接读 Paimon 底层的 Parquet/ORC + 解析它的 manifest/snapshot，
  要么走 pypaimon。写同理要自己实现 Paimon 的 LSM bucket + snapshot 提交。

结论：**读侧 Iceberg 基本免费、Paimon 要自己解析；写侧两者的数据文件都能用 DuckDB/Arrow 写，但提交协议都要
jude 自己实现**——而这正是"分布式写入引擎"的核心，也是最有价值的部分。

## 读路径

### Iceberg
`jude.read_iceberg(path_or_metadata, snapshot=None)` 下降为
`SELECT * FROM iceberg_scan('…')`，返回一个普通 jude relation，之后 filter/join/聚合照常在 DuckDB 上跑。
- **谓词/分区下推**：`iceberg_scan` 会用 Iceberg 的分区与列统计裁剪文件；jude 把 relation 上的 filter
  尽量下推进这条 SQL，让裁剪生效。
- **快照 / 时间旅行**：`iceberg_snapshots('…')` 列出快照，`iceberg_scan(..., snapshot_id=…)` 读历史版本；
  `read_iceberg(snapshot=…)` 暴露它。

### Paimon
没有扩展，两条路：
1. **直读底层文件（Rust 优先）**：Paimon 表目录里是 Parquet/ORC 数据文件 + `snapshot/` + `manifest/`。
   解析最新 snapshot → 得到本次要读的 manifest → 得到数据文件清单 → 用 DuckDB 的 `read_parquet` 扫。
   manifest 解析（Avro/JSON）放 Rust 或薄 Python，数据扫描交 DuckDB。
2. **pypaimon 兜底**：复杂特性（LSM 合并读、changelog）先用 pypaimon 拿到文件清单/schema，jude 再扫。
诚实地说，Paimon 读的第一版只保证"读某个 snapshot 的全量 Parquet 数据文件"，LSM/changelog 语义是后续。

## 分布式写路径（重点）

这是 jude 作为写入引擎的核心。写一张表格式表 = **并行写数据文件 + 一次原子提交**，天然套进 jude 的
分区级模型（和 Model B 物化 exchange 同构：worker 各自产出、driver 收尾）：

```
   rel.write_iceberg(path, partition_by=[...], mode="append"|"overwrite")

   ① 分区：WorkerManager 决定把 relation 切成 N 个分区（决策在 Rust）
   ② 并行写数据文件（在 Ray worker 上）：
        worker i ── 本分区 Arrow ──▶ COPY TO 'data/part-i-<uuid>.parquet' (Parquet)
                                     ▶ 收集该文件的路径 + 行数 + 列统计（写 manifest 要用）
   ③ 原子提交（在 driver 上，一次）：
        Iceberg：把新数据文件写进一个新 manifest → 新 manifest list → 新 metadata.json
                 → 原子切换 catalog 指针（version-hint / catalog commit）
        Paimon： 写 LSM bucket 文件 → 追加一个新 snapshot（指向新 manifest）
```

- **哪些在 Rust**：分区决策（复用 `WorkerManager.partition_plan`）、提交的**编排**（收集各 worker 回报的
  文件元数据、决定这次提交包含哪些文件、串行化提交步骤）。
- **哪些在 Python/DuckDB**：worker 上 `COPY TO parquet` 写数据文件是 DuckDB 执行；Iceberg/Paimon 的
  **提交协议**（写 manifest、metadata.json、原子切换）第一版走 **pyiceberg / pypaimon**——它们已实现了
  正确的提交语义与并发控制，自己从零写 manifest 二进制格式风险高、收益低。诚实取舍：**数据面 Rust/DuckDB
  快路径，元数据/提交面先借成熟 Python 库**，等真成为瓶颈再 Rust 化。
- **原子性与并发**：提交必须原子（要么整批文件可见、要么都不可见），且要处理并发写冲突（Iceberg 的乐观
  并发：提交时检查 base snapshot 未变，变了则重试）。这套由 pyiceberg 保证；jude 的职责是把"这次写了哪些
  文件"正确地交给它，并在 driver 上串行化这一步。
- **mode**：`append` 加数据文件到新 snapshot；`overwrite` 用一次 replace 提交使旧文件失效（不物理删，靠
  snapshot 过期回收）。

### 与容错的关系（诚实）
分布式写的容错和分布式读的 shuffle 容错是同一块短板：若某个 worker 写文件失败，driver 提交前应丢弃其部分
文件（未提交的数据文件是孤儿，不影响表状态——这正是表格式"先写文件后提交指针"的好处）。真正的重试/清理
用 `FteTaskAttemptId` 词汇，与分布式设计的容错轨道共享，本文不重复规划。

## API 草案

```python
# 读
rel = jude.read_iceberg("s3://.../table", snapshot=None)     # -> Relation
rel = jude.read_paimon("/warehouse/db.db/table")

# 写（分布式）
rel.write_iceberg(path, partition_by=["dt"], mode="append")
rel.write_paimon(path, primary_key=["id"], partition_by=["dt"])
```

`write_*` 是一个物化/边界式的**汇**节点：它消费 relation 的分区、并行写文件、提交，返回提交结果（新
snapshot id、写入行数/文件数）。

## 分阶段计划

- **P1 Iceberg 读**：`read_iceberg` → `iceberg_scan`，谓词下推 + 快照选择。基本白拿。**✅ 已实现并测试**
  （`conn.read_iceberg`/`iceberg_snapshots`，与 SQL filter/aggregate 组合，往返验证）。
- **P2 单机 Iceberg 写**：数据文件用 Rust/DuckDB `COPY TO` 按分区写，pyiceberg 提交。**✅ 已实现并测试**
  （`rel.write_iceberg`，append/overwrite，往返 2500 行多分区验证；数据面在 Rust）。
- **P3 分布式 Iceberg 写**：worker 并行写数据文件、driver 一次提交（复用 WorkerManager）。**✅ 已实现并测试**
  （`RayRunner.distributed_write_iceberg`，Ray 上 2000 行 4 分区往返 + overwrite）。
- **P4 Paimon 读**：解析 snapshot/manifest → read_parquet；先全量、后 LSM/changelog。**⛔ 本环境阻塞**：
  `pypaimon` 要求 `pyarrow<20`（本项目用 25），其 sdist 在此构建失败；DuckDB 无 Paimon 扩展；本机也无法
  建 Paimon 表来验证。为守住"全绿、可验证"纪律，暂不落未验证代码——待 pyarrow 版本约束解决或有可测环境。
- **P5 Paimon 写**：LSM bucket + snapshot 提交（pypaimon 起步）。同 P4 阻塞。
- **P6 Rust 化提交面**：真成为瓶颈时，把 manifest/metadata 写入从 Python 挪到 Rust。（Iceberg 提交目前借
  pyiceberg；数据面已在 Rust。）
- **P7 Lance 读 + 分布式写**：**✅ 已实现并测试**。`jude.read_lance(path, columns=, filter=)`（列投影 +
  filter 下推进 Lance scan）；`rel.write_lance(path, mode=create|append|overwrite)` 单机；
  `RayRunner.distributed_write_lance`（每个 worker 用 `LanceFragment.create` 写一个数据 fragment，driver 用
  一次 `Append`/`Overwrite` operation 提交 fragment 集，2000 行 4 分区往返验证）。**Lance 的 writer 本身是
  Rust（pylance 底层 lance crate）**，所以数据面天然在 Rust——正符合"写入用 Rust"。不像 Paimon 被
  pyarrow<20 卡住，pylance 8.0.0 与 pyarrow 25 兼容。

## 测试

在 tmp 目录建小 Iceberg 表（pyiceberg + 本地 catalog）验证 `read_iceberg` 读出正确行与快照；写路径做
**往返**：`rel.write_iceberg(tmp) → jude.read_iceberg(tmp)` 行集相等；分布式写断言与单机写结果一致（同一
份数据、同一份 snapshot 内容）。Paimon 同法用小表。门禁：`pytest`（依赖 pyiceberg/pypaimon 的用
`importorskip` 守卫），全套 0 失败。
