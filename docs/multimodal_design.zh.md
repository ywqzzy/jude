# jude 的多模态

## SQL 看不见的问题

关系引擎有一个封闭的类型世界：数字、字符串、日期，偶尔有 list 或 struct。图像不属于其中任何一个。你可以把
编码后的 PNG 字节塞进 `BLOB` 列，DuckDB 会乐意存储和 shuffle 它们，但它无法*解码*、无法缩放、无法把它们变成
模型想要的 `(H, W, C)` 张量——SQL 里没有 `RESIZE(image, 224, 224)`，将来也不会有。于是每个做多模态的引擎都
面临同一个岔路：要么给 SQL 层堆一坨不透明的标量 UDF（慢、逐行、无类型），要么在 SQL 旁边长出*第二套*表达式
系统，让它懂像素、懂采样、懂帧。

Daft 走了第二条路，这是对的：`col("url").url.download().image.decode().image.resize(224, 224)
.image.to_tensor()` 读起来就像它旁边的关系代数，但每一步 `.image.*` 都是类型化的、列式的原生代码内核，而不是
被调用一百万次的 Python 函数。jude 瞄准的正是这个表面。本文讲的是：如何在原生 DuckDB 之上、不 fork 地
到达那里；Rust/Python 的线落在哪（用户的长期约束是"尽可能多用 Rust"）；以及最要紧的——让整件事成立的那一道
架构接缝：多模态算子是一个*物化边界*，而不是一个 SQL 表达式。

## 已有什么，为什么还不够

摄取与解码层已建成并测试通过。`jude.sources` 把一个 glob 或目录变成带 `path`、`size_bytes` 和一列打了逻辑
类型标签的编码字节的 relation——`ImageFileSource`、`AudioFileSource`、`VideoFrameSource`、
`DocumentSource`，四种模态全有，不只是图像。`jude.multimodal` 对每种都有批解码器：图像→张量用 PIL，音频→
浮点采样用 soundfile（带重采样），视频→一帧一行用 PyAV，文档→一页一行用 pypdf。而
`jude.pipeline.RelationPipeline` 把这些串成 cosmos-xenna 阶段，源与汇都是 relation。

缺的是让多模态感觉*原生*而非外挂的那件事：你还不能把解码写成查询的一部分。今天你得伸手去拿 `map_batches`
配一个手写函数，或者搭一条显式 pipeline。两者都能用；都不是把 `col("img").image.decode().image.resize(...)`
组合进一个 `.filter()` 和一个 `.aggregate()`。弥合这道缝就是本设计，它还顺带让我们把当前的解码器和 pipeline
的 `DecodeStage` 收拢到*一份*实现，而非现在这两份并行的。

## 接缝：多模态算子是物化边界

关键在这里。一个 jude relation 是一棵下推为 SQL、在 DuckDB 上跑的 `LogicalPlan` 树。多模态算子*无法*下推为
SQL——没有"解码这张 PNG"的 SQL。所以一旦 plan 里出现 `image.decode()`，那个节点就成了一堵墙：它*下方*的
一切仍是关系式的、照常在 DuckDB 上跑；算子本身作为一个 Rust 内核在 DuckDB 产出的 Arrow 批上跑；而*结果*作为
一个物化叶子重新进入 plan，于是它*上方*的一切——对新张量列的 filter、对它的 aggregate——又变回 DuckDB 上
普通的 SQL。

这不是为多模态新造的机制。它正是 `LogicalPlan::MapBatches` 和 `Materialized` 叶子已有的行为：
`to_subquery_sql` 有一个 `resolve` 闭包，按需把内存中的批注册成临时表，于是一个无 SQL 的节点能坐在 plan
中间，周围的层浑然不知。多模态节点 `LogicalPlan::MultimodalMap` 复用这套机件。它上面的 `to_sql` 故意返回
"不可下推"，这就是给 `materialize` 的信号：跑内核、暂存结果。复用既有边界的回报是，多模态列能与*一切*组合——
join、聚合、窗口函数、分布式 runner——且免费，因为在边界之上它只是一张带张量列的表，而 jude 早就会分布和查询
表了。

有一处这道接缝确实不确定，我宁愿点名也不愿在生产里撞见：张量列的 Arrow 类型。`TensorType` 想成为 Arrow 的
`fixed_shape_tensor` 扩展类型，但当我们为了跑边界*之上*的 SQL 而把一个批经 DuckDB 临时表往返时，DuckDB 可能
不保留扩展元数据——它也许把一个普通 `fixed_size_list<u8>` 递回来、把形状丢了。设计的答复是 `tensor.rs` 同时
拥有两种编码，能回落到 `list<u8>` + 一个显式 `shape` 列（我们本就需要的变形路径），但哪一种能挺过往返，是
P-A 要量的第一件事，因为它决定了"查询一个张量列"是无缝的还是需要一步重建。

## 边界的代价

边界不是免费的，假装它免费就是本文力避的那种天真。它有三笔代价，按你该在意的程度递减排列。

第一，它是**全停 barrier——无法跨它流水线**。边界下面的一切必须跑完、产出所有批，内核才开始；内核必须
跑完，上面的 SQL 才开始。这就是分布式设计里的那个 Model B barrier；你没法像全流水线引擎那样让行"边解码
边流过"。

第二，**临时表往返——一次拷贝**。内核输出注册成 DuckDB 临时表、被上层重扫。那是一次 Arrow→DuckDB 写入
加一次重扫：一次内存拷贝，成本随字节增长。

这两笔*对于该用边界的负载*都约等于零，而这正是重点。你只在算子昂贵时才把它放到边界后面——解码 JPEG、
resize、跑模型——那是**每行毫秒级**。而边界拷贝是**每批微秒级**。比例根本不接近：

```
   处理一批图片的时间：
   ├─ 解码 + resize（🦀 内核真正干活）  ████████████████████  ~95%+
   └─ 边界拷贝（临时表往返）             ▏                     几%
```

而且 jude 的目标流水线（scan→decode→map→sink）只有**一堵**墙、不是叠起来的一摞，所以 barrier 不累加。
边界也正是并行加速的来源（内核无 GIL / 走并行池），所以它的"代价"同时是替代慢的逐行-under-GIL 方案的机制。
因此判定规则很干净：便宜、可向量化的变换 → 引擎内、不建墙；昂贵或模型支撑的 → 建墙。你只在墙后的活远大于
墙本身时才付墙的代价，所以它永远不是瓶颈。

第三笔代价才是真能让你白干活的：**DuckDB 没法跨墙优化。** 尤其是，墙上面的 filter 不会被推到墙下面。若你的
filter 不依赖解码输出，先解码后过滤就意味着你解码了随后又扔掉的行：

```
   ❌ 慢——先全解码，再过滤：
        Filter(category = 'cat')          ← 墙上面
          └─ MultimodalMap(decode) ★      ← 解码了 1,000,000 张
               └─ Scan                       …只留 10,000 张 → 99% 白干

   ✅ 快——先过滤，再解码：
        MultimodalMap(decode) ★           ← 只解码 10,000 张
          └─ Filter(category = 'cat')     ← 墙下面（纯 SQL；DuckDB 先筛）
               └─ Scan
```

只要谓词不引用解码列（按 `category` 筛、而非按 `img.width`），它就应在边界**下面**——今天靠先写
`.filter(...)` 再 `.with_column(decode)`，以后靠一个优化器 pass 自动把与边界无关的谓词推到墙下
（谓词穿边界下推——一个已点名、尚未做的优化）。

## Rust 到哪结束，Python 从哪开始

约束是"尽可能多用 Rust"，而对多模态这有一条自然、站得住的边界，而非教条。图像的解码/缩放/裁剪/编码是
`image` crate 的地盘——成熟、快、在字节缓冲上操作、干净地释放 GIL——所以它是 Rust，直接在 Arrow
`BinaryArray`→张量再返回上操作，整批一次 `Python::detach`。音频解码+重采样是 `symphonia`，也是 Rust。URL
下载本地用 `std::fs`、HTTP 用 `ureq`，Rust，并对 `s3://`/`gs://` 留一个显式 `NotImplemented` 分支，让缺口
响亮而非沉默。

Rust *赢不了*的地方，我们不装。视频解封装和 PDF 抽文本没有哪个 Rust 库在格式覆盖上比得过 PyAV 和 pypdf，为
满足语言偏好而发一个更差的解码器是错的。所以 `python_fallback.rs` 通过 PyO3 在 Arrow 批上调用既有的
`decode_video_batch`/`decode_document_batch`。*表达式 API* 是统一的——`col("v").video.decode()` 看起来一样，
无论底下的内核是 Rust 还是 Python——但实现尊重现实：有好编解码库的地方用 Rust，只有没有的地方才用 Python。
这才是"尽可能多用 Rust"的诚实读法，而不是"为了证明观点用 Rust 重写 libav"。

## 融合，以及为什么算子链是个列表

一个多模态表达式是一条*链*：decode，然后 resize，然后 to-tensor。若每一环都是自己的 plan 节点，我们就会
解码出一个全分辨率张量列、物化它、resize 进第二列、再物化——三趟、两个丢弃列。而链是单个 `MultimodalMap`
携带 `ops: Vec<MmOp>`，分发器把整条链在一趟里折叠：解码每个元素，在寄存器里施加 resize 和 to-tensor，写一列
输出。中间的全分辨率张量从不变成 Arrow 数组。这就是为什么 Python 访问器
（`.image.decode().image.resize(...)`）累积一个算子列表而非构建嵌套 plan 节点——流式表面和融合执行是同一个
列表，从两端读。

分发器逐算子折叠 `ArrayRef -> ArrayRef`，传播 null（null 输入行是 null 输出行，绝不对垃圾字节尝试解码），并
尊重逐算子的 `on_error`：`raise`（默认，对齐 Daft）或 `null`（把一张损坏的图变成 null 而非中止一个十亿行的
作业——在规模上，这才是你真正想要的选项）。

## 一份实现，三道门

同一条 `MmOp` 链、同一个 `multimodal::apply_ops` 分发器服务三个入口，这种统一是特性而非巧合。**表达式**路径
（`col("x").image.decode()`）是新的类型化表面。**pipeline** 路径（`RelationPipeline.decode(kind)`）构建同一
条链，于是 cosmos 多阶段 pipeline 和查询引擎共享一份解码器——"在 pipeline 里解码"和"在查询里解码"之间不会
漂移。而 **`map_batches`** 作为逃生舱保留，给那些无法表达为类型化算子的任意 Python UDF。三道门，一间屋。

分布式正因为物化边界的设计而免费搭车：一个 `MultimodalMap` 跑在分区化 relation 上时，它的内核在拥有该分区
的 worker 上、对该分区的批运行，因为算子本就批形、边界本就会做按分区叶子。`col("img").image.decode()` 在
`execution_backend="ray"` 下就是 worker 上解码、无需额外代码，且结果必须与单机一致——这由测试套件检查。

## 阶段，诚实排序

P-A 是地基也是最险的部分，所以排第一：`MultimodalMap` 节点、物化边界执行、分发器、`tensor.rs`——用一个恒等
算子发布，让边界（含那次 DuckDB 临时表往返）在任何真实内核依赖它之前就端到端跑通。P-B 是图像内核，demo 的
主菜，也是 `DecodeStage` 收拢到共享实现的那一刻。P-C 加 URL 下载和音频。P-D 把视频和文档经 Python 回落接上，
让表达式 API 覆盖四种模态。P-E 是 `embed_image`——一个在 GPU 阶段跑批模型的算子——它接入单独的 GPU 推理
轨道，且在其模型运行时存在之前不假装存在。

每个阶段"做完"的标准相同：内核的 Rust 单测（解码一张合成 2×2 PNG 并断言像素、resize、encode/decode 往返、
null 传播、对垃圾字节的 `on_error` 行为），一个 Python 端到端测试（解码真实 fixture 并用 SQL 查询结果），一个
分布式一致性测试，以及整套保持零失败。
