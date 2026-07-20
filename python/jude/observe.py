"""jude.observe — observability facade over the Rust ``MetricsRegistry``.

The state and all counter mutations live in Rust (``jude.observe.MetricsRegistry``,
process-global behind a Mutex, GIL-free). This module is the thin Python side:

- a process singleton (``registry()``) so every connection/runner records into
  one view;
- small ``time.time()``-stamped helpers (Rust stays clock-free for deterministic
  tests, so Python owns the clock);
- ``poll_cluster_nodes()`` — snapshot ``ray.nodes()`` into the registry;
- ``serve(port)`` — a dependency-free HTTP endpoint exposing the JSON snapshot
  (consumed by the console and the React frontend);
- ``@track`` / ``query(...)`` — context helpers to time a query end-to-end.

Nothing here makes scheduling decisions — it only records what happened.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from typing import Any, Iterator

from .jude import _observe  # Rust submodule (registered as jude.jude._observe)

__all__ = [
    "registry",
    "reset",
    "snapshot",
    "query",
    "record_pool",
    "remove_pool",
    "poll_cluster_nodes",
    "serve",
    "stop_server",
    "start_node_poller",
    "run_dev_server",
    "audit_log",
    "audit_list",
    "audit_get",
    "audit_stats",
    "audit_clear",
    "set_audit_enabled",
]

_REGISTRY: Any = None
_REGISTRY_LOCK = threading.Lock()


def registry() -> Any:
    """Process-global ``MetricsRegistry`` (created on first use)."""
    global _REGISTRY
    if _REGISTRY is None:
        with _REGISTRY_LOCK:
            if _REGISTRY is None:
                _REGISTRY = _observe.MetricsRegistry()
    return _REGISTRY


def reset() -> None:
    """Clear all recorded state."""
    registry().reset()


def snapshot() -> dict:
    """Full metrics snapshot as a dict (queries/stages/pools/nodes/events/summary)."""
    return json.loads(registry().snapshot_json())


def _pct(sorted_vals: list, p: float) -> float:
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, int(round(p / 100.0 * (len(sorted_vals) - 1))))
    return float(sorted_vals[i])


def summary() -> dict:
    """Derived rollups over the current snapshot — the numbers a dashboard/alert
    wants but that aren't stored raw: query counts by status/kind, end-to-end
    latency percentiles (p50/p95/p99, done queries), rows/sec throughput, live
    stage/task progress, pool utilization, and curation data-quality (rows in→out,
    removed, keep-rate). Cheap to compute; safe to poll."""
    snap = snapshot()
    queries = snap.get("queries", [])
    stages = snap.get("stages", []) or list((snap.get("stages_by_id") or {}).values())
    pools = snap.get("pools", [])

    by_status: dict = {}
    by_kind: dict = {}
    durations: list = []
    rows_done = 0
    span_lo = None
    span_hi = None
    for q in queries:
        by_status[q.get("status", "?")] = by_status.get(q.get("status", "?"), 0) + 1
        by_kind[q.get("kind", "?")] = by_kind.get(q.get("kind", "?"), 0) + 1
        if q.get("status") == "done" and q.get("ended_at") and q.get("started_at"):
            durations.append((q["ended_at"] - q["started_at"]) * 1000.0)
            rows_done += int(q.get("rows", 0) or 0)
            span_lo = q["started_at"] if span_lo is None else min(span_lo, q["started_at"])
            span_hi = q["ended_at"] if span_hi is None else max(span_hi, q["ended_at"])
    durations.sort()
    wall = (span_hi - span_lo) if (span_lo is not None and span_hi and span_hi > span_lo) else 0.0

    # curation data-quality rollup: queries of kind "curate" carry rows_in (bytes
    # field, by convention from `curate()`) and rows_out (rows).
    cur_in = cur_out = 0
    curates = 0
    for q in queries:
        if q.get("kind") == "curate":
            curates += 1
            cur_in += int(q.get("bytes", 0) or 0)
            cur_out += int(q.get("rows", 0) or 0)

    live_tasks_done = sum(int(s.get("tasks_done", 0) or 0) for s in stages)
    live_tasks_total = sum(int(s.get("tasks_total", 0) or 0) for s in stages)
    tasks_failed = sum(int(s.get("tasks_failed", 0) or 0) for s in stages)

    return {
        "queries": {
            "total": len(queries),
            "by_status": by_status,
            "by_kind": by_kind,
            "latency_ms": {
                "p50": _pct(durations, 50), "p95": _pct(durations, 95),
                "p99": _pct(durations, 99), "max": (durations[-1] if durations else 0.0),
            },
            "rows_per_sec": (rows_done / wall) if wall > 0 else 0.0,
        },
        "stages": {"tasks_done": live_tasks_done, "tasks_total": live_tasks_total,
                   "tasks_failed": tasks_failed},
        "pools": {p.get("key", "?"): {
            "backend": p.get("backend"), "workers": p.get("num_workers"),
            "rows": p.get("rows"), "busy_ms": (p.get("busy_ns", 0) or 0) / 1e6,
        } for p in pools},
        "curation": {"ops": curates, "rows_in": cur_in, "rows_out": cur_out,
                     "removed": max(0, cur_in - cur_out),
                     "keep_rate": (cur_out / cur_in) if cur_in else 1.0},
    }


def _prom_lines(name: str, help_: str, typ: str, samples: list) -> list:
    """Format one Prometheus metric family. `samples` = [(labels_dict|None, value)]."""
    out = [f"# HELP {name} {help_}", f"# TYPE {name} {typ}"]
    for labels, val in samples:
        if labels:
            lbl = ",".join(f'{k}="{str(v)}"' for k, v in labels.items())
            out.append(f"{name}{{{lbl}}} {val}")
        else:
            out.append(f"{name} {val}")
    return out


def prometheus_text() -> str:
    """Expose jude's metrics in Prometheus text exposition format, so a Prometheus
    scrape (→ Grafana) can chart engine throughput/latency/curation alongside the
    Ray dashboard. Point a scrape job at ``/api/prometheus`` on the observe server."""
    s = summary()
    snap = snapshot()
    lines: list = []
    q = s["queries"]
    lines += _prom_lines("jude_queries_total", "Queries by status", "gauge",
                         [({"status": k}, v) for k, v in q["by_status"].items()])
    lines += _prom_lines("jude_query_latency_ms", "End-to-end query latency (done)", "gauge",
                         [({"quantile": "0.5"}, q["latency_ms"]["p50"]),
                          ({"quantile": "0.95"}, q["latency_ms"]["p95"]),
                          ({"quantile": "0.99"}, q["latency_ms"]["p99"])])
    lines += _prom_lines("jude_query_rows_per_second", "Row throughput (done queries)", "gauge",
                         [(None, round(q["rows_per_sec"], 3))])
    st = s["stages"]
    lines += _prom_lines("jude_stage_tasks", "Distributed stage task accounting", "gauge",
                         [({"state": "done"}, st["tasks_done"]),
                          ({"state": "total"}, st["tasks_total"]),
                          ({"state": "failed"}, st["tasks_failed"])])
    cur = s["curation"]
    lines += _prom_lines("jude_curation_rows", "Curation rows in/out/removed", "gauge",
                         [({"stage": "in"}, cur["rows_in"]), ({"stage": "out"}, cur["rows_out"]),
                          ({"stage": "removed"}, cur["removed"])])
    lines += _prom_lines("jude_curation_keep_rate", "Curation keep rate (out/in)", "gauge",
                         [(None, round(cur["keep_rate"], 4))])
    lines += _prom_lines("jude_pool_busy_ms", "UDF pool busy time", "gauge",
                         [({"pool": k}, round(v["busy_ms"], 1)) for k, v in s["pools"].items()])
    # cluster resources (from Ray)
    cr = snap.get("cluster_resources") or {}
    for dim, vals in (cr.items() if isinstance(cr, dict) else []):
        if isinstance(vals, dict):
            for field in ("total", "used", "available"):
                if field in vals:
                    lines += _prom_lines(f"jude_cluster_{dim}", f"Cluster {dim}", "gauge",
                                         [({"kind": field}, vals[field])])
    return "\n".join(lines) + "\n"


# --- durable audit log (redb) -----------------------------------------------

_AUDIT: Any = None
_AUDIT_LOCK = threading.Lock()
_AUDIT_ENABLED = True


def _default_audit_path() -> str:
    import os

    d = os.environ.get("JUDE_AUDIT_DIR") or os.path.join(os.path.expanduser("~"), ".jude")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "audit.redb")


def audit_log(path: str | None = None) -> Any:
    """Process-global durable ``AuditLog`` (redb), created on first use.

    Path resolution: explicit ``path`` > ``$JUDE_AUDIT_DIR/audit.redb`` >
    ``~/.jude/audit.redb``. Returns None if audit is disabled or redb can't open.
    """
    global _AUDIT
    if not _AUDIT_ENABLED:
        return None
    if _AUDIT is None:
        with _AUDIT_LOCK:
            if _AUDIT is None:
                try:
                    _AUDIT = _observe.AuditLog(path or _default_audit_path())
                except Exception:  # noqa: BLE001 — audit must never break execution
                    return None
    return _AUDIT


def set_audit_enabled(enabled: bool) -> None:
    """Turn durable audit recording on/off (on by default)."""
    global _AUDIT_ENABLED
    _AUDIT_ENABLED = bool(enabled)


def _audit_persist(handle: "_QueryHandle", *, status: str, ended: float,
                   rows: int = 0, nbytes: int = 0, error: str | None = None) -> None:
    """Persist a finished execution to the durable audit log (best-effort)."""
    log = audit_log()
    if log is None:
        return
    try:
        rec = {
            "label": handle._label,
            "kind": handle._kind,
            "status": status,
            "started_at": handle._started,
            "ended_at": ended,
            "duration_ms": max(0.0, (ended - handle._started) * 1000.0),
            "rows": rows,
            "bytes": nbytes,
            "stages": handle._stages,
            "error": error,
            "detail": handle._detail,
        }
        log.record(json.dumps(rec))
    except Exception:  # noqa: BLE001
        pass


def audit_list(limit: int = 100, kind: str | None = None, status: str | None = None) -> list[dict]:
    """List persisted execution records newest-first (optionally filtered)."""
    log = audit_log()
    if log is None:
        return []
    return json.loads(log.list(int(limit), kind, status))


def audit_get(audit_id: int) -> dict | None:
    """Fetch one persisted record by its audit id."""
    log = audit_log()
    if log is None:
        return None
    raw = log.get(int(audit_id))
    return json.loads(raw) if raw else None


def audit_stats() -> dict:
    """Aggregate audit stats (totals by status, rows)."""
    log = audit_log()
    if log is None:
        return {"total": 0, "done": 0, "error": 0, "rows_total": 0}
    return json.loads(log.stats_json())


def audit_clear() -> None:
    """Delete all persisted audit records (tests / retention reset)."""
    log = audit_log()
    if log is not None:
        log.clear()


# --- query timing -----------------------------------------------------------


class _QueryHandle:
    """Handle for an in-flight query; use its ``.stage()`` to register stages."""

    def __init__(self, qid: int):
        self.id = qid
        self._done = False
        self._label = ""
        self._kind = ""
        self._started = 0.0
        self._detail: dict = {}
        self._stages: list[str] = []

    def stage(self, name: str, tasks_total: int = 0) -> "_StageHandle":
        sid = registry().stage_start(self.id, name, int(tasks_total), time.time())
        self._stages.append(name)
        return _StageHandle(sid)

    def detail(self, **kv: Any) -> "_QueryHandle":
        """Attach arbitrary detail (sql, plan, num_workers, node, …) to be
        persisted with the audit record."""
        self._detail.update(kv)
        return self

    def done(self, rows: int = 0, nbytes: int = 0) -> None:
        if not self._done:
            now = time.time()
            registry().query_done(self.id, int(rows), int(nbytes), now)
            self._done = True
            _audit_persist(self, status="done", rows=int(rows), nbytes=int(nbytes), ended=now)

    def error(self, msg: str) -> None:
        if not self._done:
            now = time.time()
            registry().query_error(self.id, str(msg), now)
            self._done = True
            _audit_persist(self, status="error", error=str(msg), ended=now)


class _StageHandle:
    def __init__(self, sid: int):
        self.id = sid

    def progress(
        self,
        tasks_done: int = 0,
        tasks_failed: int = 0,
        attempts: int = 0,
        rows: int = 0,
        nbytes: int = 0,
    ) -> None:
        registry().stage_progress(
            self.id, int(tasks_done), int(tasks_failed), int(attempts), int(rows), int(nbytes)
        )

    def done(self) -> None:
        registry().stage_done(self.id, time.time())


@contextlib.contextmanager
def query(label: str, kind: str = "local") -> Iterator[_QueryHandle]:
    """Context manager timing a query end-to-end; records error on exception.

    >>> with observe.query("collect users", kind="distributed") as q:
    ...     st = q.stage("shuffle", tasks_total=8)
    ...     ...  # st.progress(tasks_done=1, rows=1000)
    ...     q.done(rows=total)
    """
    qid = registry().query_start(label, kind, time.time())
    h = _QueryHandle(qid)
    h._label = label
    h._kind = kind
    h._started = time.time()
    try:
        yield h
        h.done()
    except BaseException as e:  # noqa: BLE001 — record then re-raise
        h.error(f"{type(e).__name__}: {e}")
        raise


@contextlib.contextmanager
def curate(op: str, rows_in: int) -> "Iterator[_CurateHandle]":
    """Record a curation operator run for data-quality observability. Captures
    rows in → out (and thus removed / keep-rate), surfaced in ``summary()`` and
    the Prometheus endpoint. Reuses the query store (kind="curate"; rows_in is
    stashed in the byte field by convention).

    >>> with observe.curate("fuzzy_dedup", rows_in=len(corpus)) as c:
    ...     out = curate.fuzzy_dedup(corpus, ...)
    ...     c.done(rows_out=out.num_rows)
    """
    qid = registry().query_start(op, "curate", time.time())
    h = _CurateHandle(qid, int(rows_in))
    try:
        yield h
        h._finish()
    except BaseException as e:  # noqa: BLE001
        registry().query_error(qid, f"{type(e).__name__}: {e}", time.time())
        raise


class _CurateHandle:
    def __init__(self, qid: int, rows_in: int):
        self.id = qid
        self.rows_in = rows_in
        self.rows_out = rows_in
        self._done = False

    def done(self, rows_out: int) -> "_CurateHandle":
        self.rows_out = int(rows_out)
        return self

    def _finish(self) -> None:
        if not self._done:
            # rows = rows_out, bytes = rows_in (convention read back by summary()).
            registry().query_done(self.id, int(self.rows_out), int(self.rows_in), time.time())
            self._done = True


# --- UDF pool utilization ----------------------------------------------------


def record_pool(
    key: str,
    backend: str,
    num_workers: int,
    *,
    batches_in: int = 0,
    batches_out: int = 0,
    rows: int = 0,
    busy_ns: int = 0,
) -> None:
    """Upsert a UDF pool's utilization counters (called by the executor layer)."""
    registry().pool_update(
        str(key),
        str(backend),
        int(num_workers),
        int(batches_in),
        int(batches_out),
        int(rows),
        int(busy_ns),
        time.time(),
    )


def remove_pool(key: str) -> None:
    registry().pool_remove(str(key))


# --- cluster node inventory --------------------------------------------------


def poll_cluster_nodes() -> int:
    """Snapshot ``ray.nodes()`` into the registry with per-node capacity and
    cluster-wide used/total resources. Returns node count (0 if no Ray)."""
    try:
        import ray

        if not ray.is_initialized():
            return 0
        nodes = []
        for n in ray.nodes():
            res = n.get("Resources", {}) or {}
            nodes.append(
                {
                    "node_id": str(n.get("NodeID", "")),
                    "alive": bool(n.get("Alive", False)),
                    "cpus": float(res.get("CPU", 0.0)),
                    "gpus": float(res.get("GPU", 0.0)),
                    "address": str(n.get("NodeManagerAddress", "")),
                    "hosts": [],
                }
            )
        registry().set_nodes(nodes, time.time())
        # Cluster-wide utilization: total - available = used, per dimension.
        try:
            total = ray.cluster_resources() or {}
            avail = ray.available_resources() or {}
            cluster = {}
            for dim in ("CPU", "GPU", "memory", "object_store_memory"):
                t = float(total.get(dim, 0.0))
                a = float(avail.get(dim, 0.0))
                if t > 0:
                    cluster[dim] = {"total": t, "used": max(0.0, t - a), "available": a}
            registry().set_cluster_resources(json.dumps(cluster), time.time())
        except Exception:  # noqa: BLE001
            pass
        return len(nodes)
    except Exception:  # noqa: BLE001 — observability must never break execution
        return 0


# --- HTTP endpoint -----------------------------------------------------------

_SERVER: Any = None
_SERVER_THREAD: threading.Thread | None = None


def _frontend_dist() -> "str | None":
    """Path to the built React frontend (frontend/dist/), if present."""
    import os

    # observe.py -> jude/ -> python/ -> repo root -> frontend/dist
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    dist = os.path.join(root, "frontend", "dist")
    return dist if os.path.isdir(dist) and os.path.isfile(os.path.join(dist, "index.html")) else None


def serve(port: int = 8477, host: str = "127.0.0.1", poll_ray: bool = True, serve_ui: bool = True) -> str:
    """Start a dependency-free HTTP server exposing metrics AND the dashboard UI.

    Endpoints:
      GET /api/metrics -> full snapshot JSON (refreshes cluster nodes first)
      GET /api/health  -> {"ok": true}
      GET /            -> the built React dashboard (frontend/dist/) when
                          available and ``serve_ui=True``; else a tiny JSON index.
      GET /<asset>     -> static files from frontend/dist/ (SPA fallback to
                          index.html for client routes).

    So `serve()` alone gives you the full UI at http://host:port — no separate
    `npm run dev` needed once the frontend is built. Returns the base URL.
    CORS is open (``*``) so a separate React dev server can also fetch it.
    """
    global _SERVER, _SERVER_THREAD
    if _SERVER is not None:
        return f"http://{host}:{port}"

    import mimetypes
    import os
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    dist = _frontend_dist() if serve_ui else None

    class _Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_static(self, rel: str) -> bool:
            """Serve a file from dist/; return False if not found. Guards against
            path traversal by resolving under dist."""
            if not dist:
                return False
            target = os.path.normpath(os.path.join(dist, rel.lstrip("/")))
            if not target.startswith(os.path.abspath(dist)):
                return False
            if os.path.isfile(target):
                ctype = mimetypes.guess_type(target)[0] or "application/octet-stream"
                with open(target, "rb") as f:
                    self._send(200, f.read(), ctype)
                return True
            return False

        def do_GET(self) -> None:  # noqa: N802 — http.server API
            from urllib.parse import parse_qs, urlparse

            parsed = urlparse(self.path)
            path = parsed.path
            if path in ("/api/health", "/health"):
                self._send(200, b'{"ok":true}')
            elif path in ("/api/metrics", "/metrics"):
                if poll_ray:
                    poll_cluster_nodes()
                self._send(200, registry().snapshot_json().encode("utf-8"))
            elif path == "/api/summary":
                if poll_ray:
                    poll_cluster_nodes()
                self._send(200, json.dumps(summary()).encode("utf-8"))
            elif path in ("/api/prometheus", "/prometheus", "/metrics/prometheus"):
                if poll_ray:
                    poll_cluster_nodes()
                self._send(200, prometheus_text().encode("utf-8"),
                           ctype="text/plain; version=0.0.4; charset=utf-8")
            elif path == "/api/audit":
                qs = parse_qs(parsed.query)
                limit = int(qs.get("limit", ["100"])[0])
                kind = qs.get("kind", [None])[0]
                status = qs.get("status", [None])[0]
                body = {
                    "records": audit_list(limit=limit, kind=kind, status=status),
                    "stats": audit_stats(),
                }
                self._send(200, json.dumps(body).encode("utf-8"))
            elif path.startswith("/api/audit/"):
                try:
                    aid = int(path.rsplit("/", 1)[1])
                    rec = audit_get(aid)
                    self._send(200 if rec else 404, json.dumps(rec or {"error": "not found"}).encode("utf-8"))
                except ValueError:
                    self._send(400, b'{"error":"bad id"}')
            elif dist and path == "/":
                self._serve_static("index.html")
            elif dist and self._serve_static(path):
                pass  # served a static asset
            elif dist:
                # SPA fallback: unknown non-API path -> index.html
                self._serve_static("index.html")
            elif path == "/":
                self._send(200, b'{"ok":true,"service":"jude.observe","metrics":"/api/metrics","ui":"build frontend/ to serve it here"}')
            else:
                self._send(404, b'{"error":"not found"}')

        def log_message(self, *_args: Any) -> None:  # silence access log
            pass

    _SERVER = ThreadingHTTPServer((host, port), _Handler)
    _SERVER_THREAD = threading.Thread(target=_SERVER.serve_forever, daemon=True, name="jude-observe-http")
    _SERVER_THREAD.start()
    return f"http://{host}:{port}"


def stop_server() -> None:
    global _SERVER, _SERVER_THREAD
    if _SERVER is not None:
        _SERVER.shutdown()
        _SERVER.server_close()
        _SERVER = None
        _SERVER_THREAD = None


# --- background node poller (keeps the dashboard live while idle) ------------

_POLLER: threading.Thread | None = None
_POLLER_STOP: threading.Event | None = None


def start_node_poller(interval: float = 2.0) -> None:
    """Poll ``ray.nodes()`` into the registry every ``interval`` seconds in a
    daemon thread, so the dashboard shows live cluster state even when no query
    is running. Idempotent."""
    global _POLLER, _POLLER_STOP
    if _POLLER is not None and _POLLER.is_alive():
        return
    _POLLER_STOP = threading.Event()

    def _loop():
        while not _POLLER_STOP.is_set():
            poll_cluster_nodes()
            _POLLER_STOP.wait(interval)

    _POLLER = threading.Thread(target=_loop, daemon=True, name="jude-observe-poller")
    _POLLER.start()


def stop_node_poller() -> None:
    global _POLLER, _POLLER_STOP
    if _POLLER_STOP is not None:
        _POLLER_STOP.set()
    _POLLER = None


# --- resident dev server -----------------------------------------------------


def run_dev_server(port: int = 8477, ray_cpus: int | None = None, poll_interval: float = 2.0) -> None:
    """Start a **resident** Ray + metrics server for development and block.

    This is what `python -m jude.observe` runs: it initializes a local Ray
    cluster (so its dashboard is up at :8265), starts the jude metrics endpoint
    at ``port``, begins polling cluster state, seeds a demo event so the
    frontend has something to show, and then blocks until Ctrl-C. Point the
    React dev server (`cd frontend && npm run dev`) at it and it stays connected
    across your edit/run cycles.
    """
    import time as _t

    try:
        import os

        import ray

        if not ray.is_initialized():
            # Point Ray at a Prometheus/Grafana stack if one is up (e.g. started by
            # benchmarking/ray-metrics-docker.sh) so the Dashboard Metrics tab can
            # embed the Grafana panels. Honor pre-set env; default to the stack's
            # conventional ports. Harmless if no stack is running.
            os.environ.setdefault("RAY_PROMETHEUS_HOST", "http://localhost:9090")
            os.environ.setdefault("RAY_GRAFANA_HOST", "http://localhost:3000")
            os.environ.setdefault("RAY_GRAFANA_IFRAME_HOST", os.environ["RAY_GRAFANA_HOST"])
            kw = {"ignore_reinit_error": True, "log_to_driver": False}
            if ray_cpus:
                kw["num_cpus"] = ray_cpus
            # Prefer attaching to a real head node (`ray start --head`) so that
            # benchmarks/other processes using ray.init(address="auto") land on
            # THIS cluster and show up on this dashboard. Fall back to a local
            # cluster if no head node is running.
            try:
                ray.init(address="auto", **kw)
                print("[jude.observe] attached to existing Ray head node (address=auto)")
            except Exception:  # noqa: BLE001 — no head node; start a local cluster
                ray.init(**kw)
        try:
            dash = ray.get_dashboard_url()
        except Exception:  # noqa: BLE001
            dash = "127.0.0.1:8265"
        print(f"[jude.observe] Ray dashboard: http://{dash}")
    except Exception as e:  # noqa: BLE001
        print(f"[jude.observe] Ray not available ({e}); serving metrics without a cluster.")

    url = serve(port=port, poll_ray=True)
    start_node_poller(interval=poll_interval)
    # seed one event so the activity feed isn't empty on first load
    with query("dev server started", kind="local") as q:
        q.done()
    if _frontend_dist():
        print(f"[jude.observe] dashboard UI:   {url}/   (open this in your browser)")
    else:
        print("[jude.observe] frontend not built yet — run:  cd frontend && npm install && npm run build")
        print(f"[jude.observe]   ...then reload {url}/ , or dev-mode: cd frontend && JUDE_METRICS_URL={url} npm run dev")
    print(f"[jude.observe] metrics API:    {url}/api/metrics")
    print("[jude.observe] Ctrl-C to stop.")
    try:
        while True:
            _t.sleep(3600)
    except KeyboardInterrupt:
        print("\n[jude.observe] shutting down.")
        stop_node_poller()
        stop_server()


def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m jude.observe", description="Resident jude metrics + Ray dev server")
    ap.add_argument("--port", type=int, default=8477, help="metrics HTTP port")
    ap.add_argument("--ray-cpus", type=int, default=None, help="num_cpus for the local Ray cluster")
    ap.add_argument("--poll-interval", type=float, default=2.0, help="cluster poll interval (s)")
    args = ap.parse_args(argv)
    run_dev_server(port=args.port, ray_cpus=args.ray_cpus, poll_interval=args.poll_interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

