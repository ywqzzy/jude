# 查看 Ray 集群状态 & jude 可观测性指南

有两个前端可以看集群状态：**jude 自带的仪表盘**（贴合 jude 的 query/stage/UDF 视角）和 **Ray 官方 Dashboard**（Ray 原生的 jobs/actors/logs/timeline）。两者互补，下面都讲。

---

## 一、jude 仪表盘（本项目 `frontend/`）

展示 jude 视角的东西：集群节点、集群资源利用率、分布式 stage 进度、query 列表、UDF 池利用率、活动流。数据来自 Rust `MetricsRegistry`（GIL-free），不干扰执行。

### 启动步骤

1) **在 Python 里起指标服务**（跑 jude 工作负载的那个进程里）：
```python
import jude
from jude import observe

observe.serve(port=8477)         # 起 HTTP 端点 /api/metrics（CORS 全开）
# ... 正常跑 jude 分布式/UDF 负载，会自动记录进 registry ...
```
> 如果连了 Ray 集群，`/api/metrics` 每次被拉取时会自动刷新 `ray.nodes()` 和集群资源利用率。

2) **起前端**（开发模式）：
```bash
cd frontend
npm install          # 首次
npm run dev          # http://localhost:5273，自动把 /api 代理到 :8477
```
- 指标服务在别的地址？`JUDE_METRICS_URL=http://host:8477 npm run dev`
- 生产：`npm run build` 生成 `dist/`，用任意静态服务器托管（它按同源请求 `/api/metrics`）。

3) 浏览器打开 `http://localhost:5273`，看到：
- **Cluster Nodes**：每个 Ray 节点的 id / 地址 / CPU·GPU / 存活状态
- **Cluster Resource Utilization**：CPU / GPU / heap / object-store 的 used/total 进度条（>85% 红、>60% 黄）
- **Distributed Stages**：每个 shuffle stage 的 task 进度条、rows/bytes、重试数
- **Queries**：最近 query 的 kind/status/耗时/stage 链
- **UDF Pools**：subprocess / ray_actor 池的 workers/batches/rows/busy
- **Activity**：滚动事件流
- 右上角 **"Ray dashboard ↗"** 链接：一键跳到 Ray 官方 Dashboard（见下）

---

## 二、Ray 官方 Dashboard

Ray 自带一个功能强大的 Web Dashboard，展示 **jobs、actors、每节点资源、日志、事件、timeline、内存分析**——比 jude 仪表盘更底层、更全。

### 怎么访问

- **默认地址**：`http://127.0.0.1:8265`
- Ray 初始化时会打印它，日志里那行就是：
  ```
  INFO worker.py:... -- Started a local Ray instance. View the dashboard at http://127.0.0.1:8265
  ```
- 确保装了 dashboard 依赖：`pip install "ray[default]"`（只装 `ray` 核心不含 dashboard UI）。

### 各种启动场景下的地址

| 场景 | 怎么拿地址 |
|---|---|
| 本地 `ray.init()` | 默认 `http://127.0.0.1:8265`（看 init 日志那行） |
| 指定端口 | `ray.init(dashboard_port=8266)` → `http://127.0.0.1:8266` |
| `ray start --head` | 命令输出里的 `Dashboard URL`；默认 `<head-ip>:8265` |
| 远程/多机集群 | dashboard 在 head 节点：`http://<head-node-ip>:8265`；本地看要 SSH 端口转发：`ssh -L 8265:localhost:8265 user@head-node` 然后开 `http://localhost:8265` |
| 代码里查地址 | `import ray; print(ray.get_dashboard_url())`（返回 `ip:port`，前面加 `http://`） |

### Dashboard 里能看什么
- **Cluster / Nodes**：每节点 CPU/GPU/内存/object-store 实时用量
- **Jobs**：提交的作业、状态、进度
- **Actors**：所有 actor（含 jude 的 `_JudeWorker` 和 UDF `_RayUDFActor`），状态/所在节点/调用栈
- **Logs**：按节点/actor 聚合的日志
- **Metrics**（需 Prometheus/Grafana）：时间序列指标
- **Timeline**：任务调度时间线（排查慢/长尾）

### jude 里怎么快速拿到 Ray dashboard 地址
```python
import ray
ray.init()                          # 或连到已有集群
print("Ray dashboard:", ray.get_dashboard_url())   # e.g. 127.0.0.1:8265
```
或者直接看 jude 仪表盘右上角的 "Ray dashboard ↗" 链接（默认指向 :8265，可用 `VITE_RAY_DASHBOARD` 环境变量改）。

---

## 三、两个前端怎么选

| 想看的东西 | 用哪个 |
|---|---|
| jude 的 query 走了哪些 stage、每个 stage 进度、UDF 池吞吐 | **jude 仪表盘** |
| 集群整体资源利用率（jude 视角，简洁） | **jude 仪表盘** |
| Ray actor 明细、日志、timeline、内存分析、jobs | **Ray Dashboard** |
| 排查某个 actor 卡住/OOM/长尾 | **Ray Dashboard** |
| 快速看"现在几台机器、每台用了多少" | 两个都行 |

一句话：**jude 仪表盘看"我的查询/UDF 在集群上怎么跑的"，Ray Dashboard 看"Ray 底层发生了什么"。** jude 仪表盘右上角有直达 Ray Dashboard 的链接。

---

## 四、命令行快速查看（不开前端）

```python
import ray
ray.init(address="auto")            # 连到本机已有集群
print(ray.nodes())                  # 每节点详情（存活/资源/地址）
print(ray.cluster_resources())      # 集群总资源
print(ray.available_resources())    # 当前空闲资源（total - available = 已用）
```
或用 jude 的 registry 快照：
```python
from jude import observe
observe.poll_cluster_nodes()
import json; print(json.dumps(observe.snapshot()["cluster"], indent=2))
```
命令行还有：`ray status`（集群自动扩缩状态）、`ray list nodes` / `ray list actors`（需 `ray[default]`）。
