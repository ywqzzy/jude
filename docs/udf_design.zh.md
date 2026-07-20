# jude 的 UDF

## 两道门，以及为什么其中一道是陷阱

jude 今天有两套完全独立的方式在数据上跑 Python，而本文诚实的起点是：用户从自己代码的形状上，看不出
哪一套是快的。第一道门是 `conn.create_function("f", py_fn, ...)`，它把 `f` 注册成一个真正的 DuckDB 标量
函数，于是能在 SQL 里调用：`SELECT f(x) FROM t`。第二道门是
`rel.map_batches(py_fn, execution_backend="subprocess")`，它把函数发给一池 worker 解释器、把 Arrow 批
在 GIL 之外流给它们。第一道门是*读起来*最自然的那道——它是 SQL，能组合，是 DuckDB 用户预期的——但它是慢
的那道，且慢得多，因为它逐行调用 Python 且全程持有 GIL。第二道门是快的那道（多模态基准上约 6.5×），但只有
离开 SQL、把 pipeline 重构成 `map_batches` 才够得着。

Vane 没有这个陷阱，而理解*为什么*是本设计的主轴。Vane 也有两条标量注册路径，但它把 SQL 可调用的那条接到了
快执行层上，于是一个在查询内部被调用的 UDF，跑在与它的批处理 API 相同的进程外、无 GIL、动态分批的机件上。
jude 的任务是 Rust 优先地弥合这个缺口：把 in-SQL 标量路径从逐行改成向量化，并给它一座通往批处理池的桥来
应对重活——而不建第二个执行引擎，因为 jude 已经有一个好的。

## 原生标量路径，说实话

`src/expression_udf/registration.rs` 是全部 in-SQL 标量表面，它刚刚在两个真实方面被改进，值得精确陈述。
一个泛型适配器 `PyUdf` 现在为*任意* arity 实现 duckdb-rs 的 `VScalar`：DuckDB 把整个输入 chunk 交给
`invoke`，arity 从 `state.param_types.len()` 读出，一个类型服务 0 参到 N 参函数（registration.rs:171-203）
——旧的 1 到 4 上限没了。而 NULL 处理在边界两个方向都正确：SQL NULL 参数变成 Python `None`
（`extract_row_value` 在 `vec.row_is_null` 时返回 `py.None()`，registration.rs:91-93），UDF 返回 `None`
变成 SQL NULL（`write_row_output` 经 `FlatVector::set_null`，registration.rs:130-132）。这些是真实的正确性
改进。

但 `invoke` 的核心是那个陷阱。它是 `Python::attach(|py| { for row in 0..len { … 建 tuple … func.call1
… 写一个输出 … } })`（registration.rs:179-195）。GIL 被获取一次并*为整个 chunk 持有*，而在那个区域内我们
每行做一次 `call1`。对一个 2048 行的 DuckDB 向量，那是 2048 次 Python 调用、2048 次参数 tuple 分配、2048 次
逐值转换，全部在锁下串行。这是本系统最糟的单一性质，也是下面计划要修的东西。

类型面是第二个问题，且更隐蔽，因为它*静默*失败。`extract_row_value` 和 `write_row_output` 只处理五种逻辑
类型——Varchar、Integer、Bigint、Double、Boolean——其余一切落入一个 `_ =>` 分支、当作字符串处理
（registration.rs:119-124 和 163-166）。而 `type_str_to_id` 却乐意把 `"FLOAT"`、`"BLOB"`、`"DATE"`、
`"TIMESTAMP"` 映射到它们的 DuckDB type id（registration.rs:18-31）。后果是正确性 bug，而非仅仅缺功能：
用 `return_type="FLOAT"` 注册一个函数，它的结果会经字符串回落而非按 `f32` 写出；一个 DATE 参数会以
字符串化的 blob 到达 UDF。相比之下，Vane 的原生路径往返完整类型矩阵——TINYINT 到 HUGEINT、UUID、
DATE/TIME/TIMESTAMP/INTERVAL、BLOB、DECIMAL 以及嵌套 LIST/STRUCT——且其测试套件逐个断言
（`vane/tests/fast/udf/test_scalar.py:56-81`）。

原生路径还缺三样东西，Vane 全部作为 `create_function` 上的一等旋钮暴露。没有 **NULL 处理模式**：Vane 区分
`DEFAULT`（含任何 NULL 参数的行在调用前被过滤掉、结果置 NULL，且禁止 UDF 返回 NULL）与 `SPECIAL`（UDF
看得到 NULL 并自行决定），一个从字符串或 int 解析的二值枚举（`null_handling_enum.hpp:15-34`，语义在
`python_udf.cpp:147-178` 和 `202-227`）。没有 **异常处理模式**：Vane 提供 `FORWARD_ERROR`（重抛）对
`RETURN_NULL`（抛异常的行变 NULL、扫描继续），对一个十亿行的作业里一个损坏输入不该中止一切而言是个真实
选择（`exception_handling_enum.hpp:13`，应用于 `python_udf.cpp:237-247` 和 `346-363`）。也没有
**volatility / 副作用**控制：Vane 把 `side_effects=True` 映射到 `FunctionStability::VOLATILE`，让 DuckDB
不折叠重复调用（`python_udf.cpp:552-555`），这对任何非确定性的东西都重要。jude 的 `create_function` 签名
是 `(name, func, parameters=None, return_type=None, **_kwargs)`（connection.rs:422-439）——那个
`**_kwargs` 吞掉 `type=`、`null_handling=`、`exception_handling=`、`side_effects=` 并把它们丢在地上。
最后，反注册是空操作：DuckDB 拒绝 `DROP` 一个经 C API 注册的标量函数（"internal catalog entry"），所以
`detach_function` 吞掉错误（registration.rs:76-79），而 Vane 能原子地替换与移除其注册。

## 进程外批处理路径，说实话

这是 jude 的强项，而它强的原因恰是分布式设计文档所论证的：编排在 Rust，且 GIL 在整个 dispatch 期间被释放。
`src/udf/subprocess.rs` 是一池持久 worker 子进程，每个跑 `python -m jude.execution._worker`，通过
stdin/stdout 说长度前缀的 Arrow IPC。`map_batches` 轮询把输入批分给 worker，在各自的 OS 线程上跑每个
worker 的切片让管道重叠（subprocess.rs:112-154），而*调用方*在 `py.detach` 下进入整个区域
（relation.rs:353-356），于是 N 个 worker 是 N 个真实解释器在真正并行、同时其他 Python 线程继续跑。池按
`(python, worker_count, hash(init_payload))` 缓存，于是每 worker ~100ms 的 spawn 成本每个不同 UDF 只付
一次（subprocess.rs:243-256）——和 Vane 的 actor 池同一思路。字节感知重分批（`rechunk_batches_bytes`，
relation.rs:480-513）让调用方能按字节（也按行）限制批大小，这正是无论行宽如何都能把一个 GPU/模型批控制在
内存预算内的关键。

其上还有两个后端，由 `map_batches_py` 按 `execution_backend` 字符串路由（relation.rs:1396-1423）：
`ray_task`/`ray_actor` 经 `jude.execution.udf_ray`（`RayTaskExecutor`、`RayActorExecutor`），它们把 Arrow
表经 Ray 对象存储搬运并保持提交顺序，actor 池把 UDF（及其模型权重）加载一次并可选占用一个 GPU；而纯 `ray`
经分区级 `map_relation` runner。worker 里有四种调用模式（`map_batches`、`map_batches_rows`、`flat_map`、
`map`），其中标量 `map` 对第一列逐行施加函数（`python/jude/execution/_common.py:42-50`）。
`jude.func` / `jude.cls` / `jude.cls.batch` 装饰器（`python/jude/expression_udf.py`）标记一个可调用对象，
让子进程/Ray 路径知道要为有状态 actor 每 worker 实例化一次类。

对着 Vane 约 1.15 万行的执行层量，这条路径缺的是把"发批、拼结果"变成流式、有准入控制、容错的运行时的一切。
没有流式：`SubprocessPool::map_batches` 物化每个输入批、全跑完、再拼接（subprocess.rs:112-154）——Vane 经
一个 Ray block/metadata 生成器协议增量产出输出、带真实的跨进程背压（`udf_ray_stream_protocol.py`），且对
本地池有一个 `/dev/shm` 共享内存预算管理器带输入租约与输出授予（`ref_bundle.py`）。没有准入控制：jude 的
每 UDF 路径没有 Vane 从查询驱动 actor 获取的非阻塞单步前瞻任务租约的对应物
（`udf_task_admission.py:61-250`）。actor 生命周期也很裸——`RayActorExecutor` 建一个固定池、shutdown 时
`ray.kill`（`udf_ray.py:104-110`），没有 Vane 的两阶段就绪、构造后 payload 注入、有截止期的 init、
副作用感知的重试抑制、actor 丢失检测。这些是真实缺口，但它们是*分布式运行时*缺口，与分布式设计里已规划的
容错和背压工作大量重叠；本文把它们当作共享的后续工作，而非重新规划。

两个诚实的非缺口，让对比保持公平。异步*用户可调用对象*在此不支持，而它们在 Vane 里也是刻意不支持的——Vane
移除了 actor 异步模式、以 `max_concurrency=1` 串行跑 actor 方法，因为用户可调用对象不是线程安全的。而两个
引擎都没有 Python **聚合** UDF（UDAF）注册路径；Vane 唯一的聚合表面是一个实验性的 Spark `registerJavaUDAF`
桩。所以"没有异步 UDF"和"没有聚合 UDF"是 jude 与 Vane 持平而非落后之处，计划不追它们。

## 真正要紧的缺口：从 SQL 到池子没有桥

退一步看，画面很鲜明。jude 有一个快的批引擎和一条慢的 in-SQL 标量路径，而*没有东西连接它们*。若你写
`SELECT classify(text) FROM docs`，`classify` 逐行在 GIL 下跑——哪怕 `classify` 是个批量模型调用、在 Arrow
批上会快 100×——因为够到池子的唯一办法是抛弃 SQL、手写 `map_batches`。

Vane 恰好把这个补上了。它的 `create_function` native/arrow 路径（`python_udf.cpp`）与 jude 的是同一种引擎内
形状，但它的*另一条*标量路径——`vane.func` / `attach_function` 及其背后的 SQL `CREATE FUNCTION`——注册一个
占位标量函数，其 bind 步 `LowerRegisteredExpressionUDF`（`python_udf_utils.cpp:282-305`）把调用改写成携带
一个 pickle 的 UDF payload，并把执行经 `CreatePythonUDFExecutor`（`udf_executor.cpp:3718`，作为执行器工厂
装在 `udf_executor.cpp:3777`）路由——即与 `map_batches` 相同的子进程/Ray dispatcher，由烘进 payload 的一个
`execution_backend` 字段选择（`BuildExpressionScalarUDFPayload`，`python_udf_utils.cpp:517`；
`CreateVaneFunctionInternal`，`pyconnection.cpp:1070-1122`）。结果是：一个*在 SQL 查询内部被调用*的标量
UDF 在 Vane 里于批处理池上无 GIL 地执行。那根线是 jude 最重要的缺失，且必须说清 jude 无法字面照搬这个机制：
Vane 能拦截一棵已绑定的表达式树，因为它 fork 了引擎，而 jude 把 plan 下推成 SQL *字符串*、跑在原生 DuckDB
上，所以它没有可挂改写的已绑定表达式钩子。因此计划用另一条诚实的路到达同一目的地——jude 已为多模态建好的
那道物化边界接缝。

## 计划

### 阶段 1 — 向量化（arrow-native）标量 UDF

这是头条修复，而令人愉快的意外是难的部分已经替我们做好了。duckdb-rs 在 jude 所钉的版本上暴露一个
`VArrowScalar` trait（`vscalar/arrow.rs:75-102`），其一揽子 `impl VScalar`（arrow.rs:104-128）正好做我们
想要的转换：DuckDB 交出整个 `DataChunkHandle`，`data_chunk_to_arrow` 一次把它变成 Arrow `RecordBatch`，
`invoke(state, batch) -> Arc<dyn Array>` 对*整个 chunk 调用一次*，`write_arrow_array_to_vector` 把结果写回
（arrow.rs:115-116）。jude 的 `Cargo.toml` 已启用 `vscalar-arrow` feature。所以向量化路径不是研究问题，是
管道活。

设计：给 `create_function` 加一个 `type="arrow"`（等价 `vectorized=True`）模式。在它之下 jude 注册一个
`VArrowScalar` 适配器。其 `invoke` 拿 `RecordBatch`，一次导出为 pyarrow `RecordBatch`（jude 已有
`arrow_ffi::batches_to_pyarrow_table` 正好做这个 FFI 跳），以列作参数调用一次用户函数——对齐 Vane 的 arrow
语义，UDF 收到 `pa.ChunkedArray` 并返回数组/表（`python_udf.cpp:180-307`，契约见 `vane/duckdb/udf.py` 的
`vectorized` 装饰器）——拿回 pyarrow 数组，导入为 Arrow `ArrayRef`，返回。GIL **每 ~2048 行向量获取一次**
而非每行一次，而对于函数体本身就是 numpy/pyarrow 向量化的绝大多数情形，Python 侧是 C 速度的列式数学、完全
没有逐行解释器开销。这是逐行-under-GIL 弱点的直接解药，且完全在 Rust 里。

第二个免费的收获顺带来：arrow 路径从 duckdb-rs 的 Arrow↔DuckDB 转换继承**全类型覆盖**，因为
`data_chunk_to_arrow` / `write_arrow_array_to_vector` 已知整个类型矩阵。arrow 路径一存在，FLOAT、BLOB、
DATE、TIMESTAMP、DECIMAL、LIST、STRUCT 的 UDF 就能用，无需在 `extract_row_value` 里逐类型手写分支。原有
逐行原生路径为真正不可向量化的标量函数保留，但其静默字符串回落必须改成对不支持类型*报错*而非损坏数据
（registration.rs:119-124, 163-166），且其五类型集应扩到常见标量。

### 阶段 2 — NULL、错误、volatility 语义

arrow 适配器就位后，把 `create_function` 现在丢弃的旋钮接上。`null_handling` 变成对齐 Vane 的二模式枚举：
`default` 在调用前把含 NULL 参数的行滤出批、调用后再置回 NULL（在 arrow 路径这是一个 Arrow filter + 一次
scatter-back，与 `python_udf.cpp:202-227` 同形），`special` 让 NULL 透传。`exception_handling` 变成
`forward`（重抛，今天唯一行为）对 `null`（抛异常的 chunk、或原生路径的行，变 NULL 并继续）。
`side_effects=True` 把 duckdb-rs `ScalarFunction` 的 stability 设为 volatile 让 DuckDB 不折叠调用。变参从
`VArrowScalar` 的变参签名支持自然得到（`ArrowScalarParams::Variadic`，arrow.rs:16-20）。可选但便宜：在 Rust
里读 Python 签名的类型注解，在省略时推断 `parameters`/`return_dtype`，镜像 Vane 的 `AnalyzeSignature`
（`python_udf.cpp:497-525`），这样用户不被迫传 SQL 类型字符串。

### 阶段 3 — 桥：在池上跑的 in-SQL UDF

这是弥合要紧缺口的阶段，且它刻意复用多模态接缝而非发明任何东西。回忆多模态设计：一个
`LogicalPlan::MultimodalMap` 节点是*物化边界*：它无法下推为 SQL，于是 `to_sql` 返回"不可下推"，`materialize`
在其下方 DuckDB 产出的 Arrow 批上跑一个 Rust/Python 内核，结果作为物化叶子重新进入 plan、其上一切当它是普通
表。一个池支撑的标量 UDF 是同一种节点。当一个 UDF 用 `execution_backend="subprocess"|"ray_task"|
"ray_actor"` 注册时，jude *不*把它注册成 DuckDB 标量函数；而是表达式 API 把它暴露成一个访问器——
`col("text").udf(classify)`，与 `col("img").image.decode()` 同一流式形状——在被引用列上建一个 `MapBatches`/
边界节点。物化时该节点经**既有的** `serialize_udf` → `SubprocessPool` / `RayTaskExecutor` /
`RayActorExecutor` 机件路由（relation.rs:306-442）。列在 GIL 之外按批变换、缝回成新列；边界之上的一切——
filter、join、聚合、分布式 runner——仍是原生 DuckDB 上的普通 SQL，正如多模态列已免费组合那样。

这对 jude 的架构是诚实的，而非假装是 Vane。jude 无法像 fork 的引擎拦截已绑定表达式那样，拦截任意 SQL
*字符串*里的 `f(x)`，所以池支撑形式经关系/表达式 API 到达，而非经手写 SQL 文本。那是与 Vane 的一个真实
人机差异，文档不该藏它。它换来的、以零新引擎成本换来的，正是真正要紧的东西：一个每次调用昂贵的 UDF——模型
推理、远程 API、任何想要一个批的东西——在池上无 GIL 地跑、同时读起来像查询的一部分。对用户（以及日后对
自动规划器启发式）的判定规则很干净：便宜、可向量化的变换是 arrow-native 标量函数（阶段 1，引擎内，无进程跳）；
昂贵或模型支撑的调用是池支撑的边界 UDF（阶段 3）。两者共享类型系统与表达式表面，只在活儿跑在哪里上不同。

### 阶段 4 — Python 表函数（UDTF）

Vane 能经 `create_table_function` / `RegisterTableUDF`（`pyconnection.cpp:1177-1229`、`1240-1244`）把一个
Python 函数注册成 DuckDB 表函数，由同一进程外执行器支撑、且可分布。jude 的 `table_function` 只按名调用
*内建* DuckDB 表函数（connection.rs:487-512）——没法把一个 Python 生成器注册成表源。Rust 优先、与边界一致的
设计是把 UDTF 做成一个*叶子*而非 in-SQL 的 `FROM f(x)`：`jude.table_function(gen, schema=…)` 建一个物化/
边界源节点，经池跑 Python 生成器、以声明的 schema 作为 relation 回归。和阶段 3 一样，jude 没有引擎支持拿不到
真正的 in-SQL `FROM f(args)`，但一个关系级 UDTF 覆盖了目标负载真正用的"摄取并展开"形状（一个输入行扇出成
多个、一个文档到它的页），且复用池与边界、无新执行器。

### 阶段 5 — 输出 schema，与批量推理打磨

两个较小项。第一，**输出 schema** 现在被接受并忽略——`map_batches_py` 有一句字面的 `let _ = schema;`
（relation.rs:1408）——所以结果 schema 是函数碰巧返回的东西。Vane 在 payload 里声明输出 schema 并*强制*它，
包括张量 `fixed_shape_tensor` 输出、完整嵌套类型、类型正确的空结果（`udf_output_schema.py`）。jude 应把声明
的 schema 穿过边界节点，于是一个什么都不返回的 UDF 仍产出类型正确的空 relation、而不匹配的返回是清晰的错误
而非静默漂移——而这正是多模态 `TensorType` 工作（见多模态设计）与 UDF 输出汇合到一套 Arrow 张量编码之处。
第二，**批量推理**：jude 已有基于字节的动态分批（`rechunk_batches_bytes`）和一个 GPU 能力的 Ray actor 池，
这是真实地基。接入多模态 `embed_image` 轨道（那里的 P-E）的打磨是跨 actor 池的负载感知路由，以及专门对 LLM
推理的、在连续分批引擎上的可选异步执行器——那是 Vane 唯一真正每 actor 跑并发在途请求之处（其 vLLM 家族，
`duckdb/execution/vllm.py`）。这明确是*最后*一个阶段，且门控于模型运行时的存在、而非提前发明。

## 什么复用了、什么没有

计划的承重主张是：它恰好加**一个**新执行机制——阶段 1 的 arrow-native 进程内标量适配器——其余一切都是通往
jude 已有机件的新*前门*。阶段 3 的池支撑 UDF、阶段 4 的 UDTF、阶段 5 的推理打磨，全汇入同一个 `serialize_udf`
payload 和同一套 `SubprocessPool` / Ray 执行器，经多模态所用的同一道物化边界接缝；没有第二个调度器、没有并行
worker 协议。arrow 适配器是唯一真正的新代码路径，而它配得上这个位置，因为它服务池服务*得差*的那种情形：一个
便宜的每调用变换、在紧凑的引擎内循环里，那里跨一个进程边界的代价远超它省下的 GIL 获取。把线画在那儿——活儿
小而列式时在 Rust 向量化，活儿重时跨到池——jude 就在保持 Rust 优先、不 fork 论点完整的同时，匹配了 Vane 的
UDF 人机体验。

## 里程碑

已完成：一个任意 arity 的泛型 `VScalar` 适配器、边界处正确的 NULL 进/NULL 出；带缓存池、线程重叠管道、GIL
释放 dispatch 的进程外子进程池；Ray task/actor 后端与基于字节的动态分批；`func`/`cls`/`cls.batch` 装饰器。
接下来按优先级：阶段 1（经 `VArrowScalar` 的 arrow-native 标量，外加修静默 stringify 回落），大多是在既有 crate
支持上的管道活、且交付最大的单一收益；阶段 2（null/error/volatility 旋钮与签名推断）；阶段 3（从表达式 API 到
池的边界桥——架构上重要的那个）；然后阶段 4（关系级 UDTF）与阶段 5（schema 强制与推理打磨），后者与多模态和
分布式轨道汇合而非重复它们。
