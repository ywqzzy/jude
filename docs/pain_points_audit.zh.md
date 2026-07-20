# jude 能力痛点审计(诚实)

> 2026-07-20 对 5 大能力(向量检索 / 全文 / 分布式 / pipeline / curation)的代码级审计,带 `file:line`。目的:找**该深化的痛点**以收敛深耕,不铺新摊子。
> 状态标记:🔧 已修 · ⬜ 待办 · 📝 已知边界(暂不做)。

---

## 🔴 A. 静默正确性问题(无报错、结果直接错 —— 最优先)

- ⬜ **A1 常驻缓存永不失效** → 数据更新后静默返回旧结果。`vector._RESIDENT_VEC`(`vector.py:252`)、actor `_vec_cache`(`_ray_shim.py:105`)、`_DS_CACHE`(`_lance.py:20`)在 `append/delete/merge_insert/add_columns/optimize_indices` 后都不清(只在 `create_branch`/`commit_index_segments` 清)。**最尖锐的正确性缺口**。修法:写操作后 `_DS_CACHE.pop` + 版本号失效常驻矩阵。
- ⬜ **A2 分布式 BM25 比较分片本地分数** → 全局排序错。`distributed_fts` 用 `_score DESC` 归并(`vector.py:737`),但每片 BM25 用各自的 IDF+avgdl。修法:一次 map-reduce 预扫收集全局 `N`/`df`/`avgdl` 再重打分。
- ⬜ **A3 分布式 fuzzy dedup 只按第一个 band 路由 + 无跨桶连通分量**(`_ray_shim.py:377`,单机是全 band `curate.py:273-277`)→ 召回严格差于单机且静默。修法:全 band 路由 + 跨片 CC/label-propagation。
- ⬜ **A4 快/分布式向量路径硬编码整数 id + 无 payload**:`id_to_row={int(x)...}`(`vector.py:273`),无 `id` 列时用分片内行号当全局 id(`_ray_shim.py:65-66`)→ 字符串/UUID id 静默错。`id_column` 参数在 `knn_rerank`/`distributed_ann_knn` 是死参。
- ⬜ **A5 `semantic_dedup` 阈值图传递闭包**(A~B~C 链式合并)+ 全表 O(n²) 单机(`curate.rs:498-515`,`curate.py:305`)→ 过度合并、不 scale。已有 Rust kmeans(`curate_py.rs:363`)未接线。修法:聚类内按质心去重(真 SemDeDup)。

## 🟠 B. 规模悬崖(号称分布式,实卡单机)

- ⬜ **B1 每个 shuffle 算子先 driver 全量 `to_arrow()`**(`ray.py:165-166,613-614,761`)→ 卡单 driver 内存。
- ⬜ **B2 最终归并单点**(driver 或 worker0:`ray.py:461-464,586-592,783,810`)→ 高基数 GROUP BY / 全局 ORDER BY 汇一节点。
- 🔧 **B3 非可分解聚合** → 已修:`STDDEV/VARIANCE`(pop+samp)现在走**精确两阶段**(count/sum/sum²,实测与单机 0 误差);`MEDIAN/QUANTILE/COUNT(DISTINCT)/STRING_AGG` 等标记 `NotDecomposable` → **优雅 fallback 单机**(不再 `ValueError` 烧重试)。(`_agg.py` + `ray.py` `_collect_once`)待办:COUNT(DISTINCT) shuffle 精确化、分位数走可合并 t-digest sketch。
- ⬜ **B4 shuffle 无 spill + 无 skew 处理**:`hash%b` 无加盐(`_ray_shim.py:570`),reducer 全量 `concat_tables`,热 key OOM 拖垮整查询。
- ⬜ **B5 fuzzy dedup 热 band 单 worker 全量物化 + O(m²)**(`_ray_shim.py:437,459-462`)→ 真实模板页必 OOM。

## 🟡 C. LLM 数据引擎能力缺口(最贴定位,深耕方向)

- ⬜ **C1 缺 web 管线前半**:无 HTML/boilerplate 抽取、无行级 dedup、无精确子串/后缀数组 dedup、无 WARC 读(`llm_data_engine_plan.zh.md:52,118`)。
- ⬜ **C2 quality 欠 Gopher/C4**:缺重复 n-gram 家族、stopword 门、C4 行过滤、blocklist;`digit_ratio` 等三信号算了不用(死信号,`curate.rs:792-818`);无 perplexity/fastText 质量分。
- ⬜ **C3 LSH bands 不按 threshold 校准**(`curate.rs:101`)→ 单机也 ~38% 召回损失;默认 `ngram=2` 偏小。修法:datasketch 的 `(b,r)` 最优公式。
- ⬜ **C4 语言识别** 6 语启发式,日文 kanji 误判成 zh(`curate.rs:138`);无 fastText lid.176。
- ⬜ **C5 多模态 curation 仅图像 pHash + 浅质量**;无 CLIP-score/NSFW/aesthetic、无音频、无视频级 dedup;`image_dedup` 单机 + 硬编码 `bands=4`。
- ⬜ **C6 PII 无 Luhn/NER**(任意 9 位=SSN,`curate.rs:358`);去污染用文档侧比例(长文稀释,`curate.rs:399-411`)+ 合并 benchmark 边界。tokenizer-aware 长度缺(C15)。

## 🟢 D. Pipeline

- 🔧 **D1 cosmos 静默降级** → 已修:`cosmos_status()` + `_COSMOS_IMPORT_ERROR` 区分"未装 vs 导入失败(版本 skew)";`engine='cosmos'` 报真实原因。(`pipeline/__init__.py`)
- 🔧 **D2 每-stage funnel 是假的**(执行前就标 done)→ 已修:`_run_local` 记录真实 rows_in→out per stage 进注册表。(`pipeline/_multimodal.py`)
- ⬜ **D3 `from_datasource` 声称流式实则全量物化**(`_multimodal.py:322-324` `list(...)`)→ 大输入 OOM,文档误导。
- ⬜ **D4 GPU/模型 stage**:文档教在 `__init__` 加载权重(cosmos 会 cloudpickle 整模型,错;应 `setup`);fluent API 无 setup 路径 + 无 `batch_size`。
- ⬜ **D5 无失败处理**:一个坏文件/行 abort 整 pipeline;`LoadFiles` 仅本地 FS(无 S3/fsspec)。

## ⚪ E. 分布式引擎其他 + 多机

- 🔧 **E1 多机 shuffle bench** → 已加 `bench_multinode_shuffle.py`:模拟多节点(各自 object store)跑分布式 join/agg/sort + 正确性校验(此前 `bench_multinode` 只跑 UDF、无 shuffle)。
- ⬜ **E2 FT 仅整查询重试且只覆盖 `collect()`**(`ray.py:132-157,386`);长任务不收敛;重试丢常驻 actor 状态。
- 📝 **E3(已核实,措辞纠正)Rust 调度的现状**:**核心调度决策已在 Rust 且大量在用** —— `worker_for`(30 处)、`shuffle_bucket_count`(11)、`shuffle_bucket_workers`(10)、`target_partitions`/`partition_plan`/`dispatch_window`(分区大小、worker 路由、桶数、背压窗口)。**真正死的只有高级调度**:`ClusterScheduler.place`(跨查询装箱)+ `worker_for_locality`(局部性放置)—— 缺 node→worker plumbing 未接线。Python `ray.py` 里是**控制流胶水**(DAG 遍历、Ray ObjectRef 路由、SQL 拼接),这部分无法搬进 Rust(ObjectRef 是 Ray 句柄)。**下一步 Rust 化**:把 split 分配(文件/分区→worker,size-aware)从 Python 均分改为调 Rust `ArbitrarySplitAssigner`,并接线 locality/bin-pack —— 需 maturin 重编(本机磁盘 ~1.7GB 暂不够,待腾盘)。
- ⬜ **E4 backpressure + GPU admission 默认关**(`max_task_backlog=0`,`num_gpus_per_worker=0`)。

## 测试盲区(横跨)

- ⬜ 最快/最高吞吐向量路径(`knn_ann_resident`/`distributed_knn_resident(_batch)`)零测试。
- ⬜ FTS 只测 membership 不测 ranking;单机 `hybrid_search` 无测试。
- ⬜ 分布式 fuzzy dedup 不测与单机 parity;semantic_dedup 链式不测。
- ⬜ local↔cosmos parity 无测试;非整数 id、日文、Luhn、长文去污染稀释都无测试。

---

## 建议的推进顺序(收敛 + 深耕 curation)

1. **P0 正确性(便宜、必堵)**:A1 缓存失效、A4 payload+字符串 id、D3 pipeline 真流式。
2. **深耕 curation(最贴定位)**:C3 LSH 校准 + A5 真 SemDeDup + A3 分布式 fuzzy dedup CC;C1 行级 dedup;C2 Gopher/C4 质量补全。
3. **规模诚实**:B3 聚合 fallback、B4 shuffle 加盐/子分区。
4. **记录为已知边界**:B1 driver 物化、E3 死调度层、多机真机验证、C5 CLIP/音视频。

> 已修:D1 cosmos 降级诊断、D2 真 funnel、E1 多机 shuffle bench(见对应提交)。
