//! Observability — a Rust-side metrics/progress registry, GIL-free.
//!
//! Vane's `duckdb/runners/progress.py` (640+ lines of Python) builds per-operator
//! and per-pipeline progress snapshots — rows/bytes throughput, task counts,
//! elapsed time — and renders a dashboard. jude keeps the same *state* in Rust:
//! a single process-global registry that records query lifecycle, distributed
//! stage progress, UDF-pool activity, and cluster node inventory, then serializes
//! a snapshot to JSON for the console, the HTTP endpoint, and the React frontend.
//!
//! Everything lives behind one `Mutex` and is a plain counter mutation or a JSON
//! read — no Ray handles, no GIL contention. Timestamps are supplied by the
//! caller (Python owns the clock; Rust must stay deterministic for tests).

use std::collections::HashMap;
use std::sync::Mutex;

use once_cell::sync::Lazy;
use pyo3::prelude::*;
use pyo3::types::PyList;
use serde::Serialize;
use serde_json::{json, Value};

/// One tracked query (a `collect`/`fetchall`/distributed run).
#[derive(Clone, Debug, Serialize)]
struct QueryRecord {
    id: u64,
    label: String,
    /// "running" | "done" | "error"
    status: String,
    kind: String, // "local" | "distributed" | "pipeline"
    started_at: f64,
    ended_at: Option<f64>,
    rows: u64,
    bytes: u64,
    error: Option<String>,
    /// Ordered stage ids belonging to this query.
    stage_ids: Vec<u64>,
}

/// One distributed stage (shuffle-bounded fragment) within a query.
#[derive(Clone, Debug, Serialize)]
struct StageRecord {
    id: u64,
    query_id: u64,
    name: String,
    /// "pending" | "running" | "done" | "error"
    status: String,
    started_at: f64,
    ended_at: Option<f64>,
    /// Task accounting for progress bars.
    tasks_total: u64,
    tasks_done: u64,
    tasks_failed: u64,
    /// Attempt retries observed (fault-tolerance signal).
    attempts: u64,
    rows: u64,
    bytes: u64,
}

/// A UDF-execution pool (subprocess or Ray-actor) and its live utilization.
#[derive(Clone, Debug, Serialize)]
struct PoolRecord {
    key: String,
    backend: String, // "subprocess" | "ray_actor" | "ray_task"
    num_workers: u64,
    batches_in: u64,
    batches_out: u64,
    rows: u64,
    /// Wall-clock ns spent applying the UDF (sum across batches).
    busy_ns: u64,
    /// last-updated timestamp (from Python clock)
    updated_at: f64,
}

/// A cluster node as seen by Ray (id + resources + which pools it hosts).
#[derive(Clone, Debug, Serialize)]
struct NodeRecord {
    node_id: String,
    alive: bool,
    cpus: f64,
    gpus: f64,
    address: String,
    /// Actors/pools placed here (free-form labels).
    hosts: Vec<String>,
    updated_at: f64,
}

struct Inner {
    seq: u64,
    queries: Vec<QueryRecord>,
    stages: HashMap<u64, StageRecord>,
    pools: HashMap<String, PoolRecord>,
    nodes: HashMap<String, NodeRecord>,
    /// Cluster-wide resource utilization (CPU/GPU/memory/object_store) as a
    /// JSON object: {dim: {total, used, available}}. Populated from Ray.
    cluster_resources: Value,
    /// Rolling event log (bounded) for the frontend's activity feed.
    events: Vec<Value>,
    max_events: usize,
    max_queries: usize,
}

impl Inner {
    fn next_id(&mut self) -> u64 {
        self.seq += 1;
        self.seq
    }

    fn push_event(&mut self, at: f64, kind: &str, msg: String) {
        self.events
            .push(json!({"at": at, "kind": kind, "msg": msg}));
        let n = self.events.len();
        if n > self.max_events {
            self.events.drain(0..n - self.max_events);
        }
    }

    fn trim_queries(&mut self) {
        // Keep running queries + the most recent finished ones.
        if self.queries.len() <= self.max_queries {
            return;
        }
        let mut keep: Vec<QueryRecord> = Vec::with_capacity(self.max_queries);
        // running first (never drop a live query), then newest finished.
        let (running, finished): (Vec<_>, Vec<_>) =
            self.queries.drain(..).partition(|q| q.status == "running");
        keep.extend(running);
        let budget = self.max_queries.saturating_sub(keep.len());
        let mut fin = finished;
        // finished are in insertion (time) order; keep the tail.
        let start = fin.len().saturating_sub(budget);
        // drop stages of the queries we're evicting
        for q in &fin[..start] {
            for sid in &q.stage_ids {
                self.stages.remove(sid);
            }
        }
        keep.extend(fin.drain(start..));
        self.queries = keep;
    }
}

static REGISTRY: Lazy<Mutex<Inner>> = Lazy::new(|| {
    Mutex::new(Inner {
        seq: 0,
        queries: Vec::new(),
        stages: HashMap::new(),
        pools: HashMap::new(),
        nodes: HashMap::new(),
        cluster_resources: json!({}),
        events: Vec::new(),
        max_events: 500,
        max_queries: 200,
    })
});

/// Process-global observability registry. All methods are `&self` (state is in a
/// process-global behind a Mutex) so any connection/runner can record into the
/// same view. `now` is passed in from Python (`time.time()`) to keep Rust
/// clock-free and deterministic under test.
#[pyclass(module = "jude.observe")]
pub struct MetricsRegistry;

#[pymethods]
impl MetricsRegistry {
    #[new]
    fn new() -> Self {
        MetricsRegistry
    }

    /// Begin a query; returns its id.
    #[pyo3(signature = (label, kind="local", now=0.0))]
    fn query_start(&self, label: &str, kind: &str, now: f64) -> u64 {
        let mut g = REGISTRY.lock().unwrap();
        let id = g.next_id();
        g.queries.push(QueryRecord {
            id,
            label: label.to_string(),
            status: "running".into(),
            kind: kind.to_string(),
            started_at: now,
            ended_at: None,
            rows: 0,
            bytes: 0,
            error: None,
            stage_ids: Vec::new(),
        });
        g.push_event(now, "query_start", format!("{label} ({kind})"));
        id
    }

    /// Finish a query successfully with final row/byte counts.
    #[pyo3(signature = (query_id, rows=0, bytes=0, now=0.0))]
    fn query_done(&self, query_id: u64, rows: u64, bytes: u64, now: f64) {
        let mut g = REGISTRY.lock().unwrap();
        let mut label = String::new();
        if let Some(q) = g.queries.iter_mut().find(|q| q.id == query_id) {
            q.status = "done".into();
            q.ended_at = Some(now);
            q.rows = rows;
            q.bytes = bytes;
            label = q.label.clone();
        }
        g.push_event(now, "query_done", format!("{label}: {rows} rows"));
        g.trim_queries();
    }

    /// Finish a query with an error.
    #[pyo3(signature = (query_id, error, now=0.0))]
    fn query_error(&self, query_id: u64, error: &str, now: f64) {
        let mut g = REGISTRY.lock().unwrap();
        let mut label = String::new();
        if let Some(q) = g.queries.iter_mut().find(|q| q.id == query_id) {
            q.status = "error".into();
            q.ended_at = Some(now);
            q.error = Some(error.to_string());
            label = q.label.clone();
        }
        g.push_event(now, "query_error", format!("{label}: {error}"));
        g.trim_queries();
    }

    /// Register a stage under a query; returns its id.
    #[pyo3(signature = (query_id, name, tasks_total=0, now=0.0))]
    fn stage_start(&self, query_id: u64, name: &str, tasks_total: u64, now: f64) -> u64 {
        let mut g = REGISTRY.lock().unwrap();
        let id = g.next_id();
        g.stages.insert(
            id,
            StageRecord {
                id,
                query_id,
                name: name.to_string(),
                status: "running".into(),
                started_at: now,
                ended_at: None,
                tasks_total,
                tasks_done: 0,
                tasks_failed: 0,
                attempts: 0,
                rows: 0,
                bytes: 0,
            },
        );
        if let Some(q) = g.queries.iter_mut().find(|q| q.id == query_id) {
            q.stage_ids.push(id);
        }
        id
    }

    /// Record progress on a stage (incremental task completion + throughput).
    #[pyo3(signature = (stage_id, tasks_done=0, tasks_failed=0, attempts=0, rows=0, bytes=0))]
    fn stage_progress(
        &self,
        stage_id: u64,
        tasks_done: u64,
        tasks_failed: u64,
        attempts: u64,
        rows: u64,
        bytes: u64,
    ) {
        let mut g = REGISTRY.lock().unwrap();
        if let Some(s) = g.stages.get_mut(&stage_id) {
            s.tasks_done += tasks_done;
            s.tasks_failed += tasks_failed;
            s.attempts += attempts;
            s.rows += rows;
            s.bytes += bytes;
        }
    }

    #[pyo3(signature = (stage_id, now=0.0))]
    fn stage_done(&self, stage_id: u64, now: f64) {
        let mut g = REGISTRY.lock().unwrap();
        if let Some(s) = g.stages.get_mut(&stage_id) {
            s.status = "done".into();
            s.ended_at = Some(now);
        }
    }

    /// Upsert a UDF pool's utilization counters (called by the executor layer).
    #[pyo3(signature = (key, backend, num_workers, batches_in=0, batches_out=0, rows=0, busy_ns=0, now=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn pool_update(
        &self,
        key: &str,
        backend: &str,
        num_workers: u64,
        batches_in: u64,
        batches_out: u64,
        rows: u64,
        busy_ns: u64,
        now: f64,
    ) {
        let mut g = REGISTRY.lock().unwrap();
        let e = g
            .pools
            .entry(key.to_string())
            .or_insert_with(|| PoolRecord {
                key: key.to_string(),
                backend: backend.to_string(),
                num_workers,
                batches_in: 0,
                batches_out: 0,
                rows: 0,
                busy_ns: 0,
                updated_at: now,
            });
        e.backend = backend.to_string();
        e.num_workers = num_workers;
        e.batches_in += batches_in;
        e.batches_out += batches_out;
        e.rows += rows;
        e.busy_ns += busy_ns;
        e.updated_at = now;
    }

    fn pool_remove(&self, key: &str) {
        let mut g = REGISTRY.lock().unwrap();
        g.pools.remove(key);
    }

    /// Replace the cluster node inventory (called after polling `ray.nodes()`).
    #[pyo3(signature = (nodes, now=0.0))]
    fn set_nodes(&self, nodes: &Bound<'_, PyList>, now: f64) -> PyResult<()> {
        let mut g = REGISTRY.lock().unwrap();
        g.nodes.clear();
        for item in nodes.iter() {
            let node_id: String = item.get_item("node_id")?.extract()?;
            let alive: bool = item
                .get_item("alive")
                .ok()
                .and_then(|v| v.extract().ok())
                .unwrap_or(true);
            let cpus: f64 = item
                .get_item("cpus")
                .ok()
                .and_then(|v| v.extract().ok())
                .unwrap_or(0.0);
            let gpus: f64 = item
                .get_item("gpus")
                .ok()
                .and_then(|v| v.extract().ok())
                .unwrap_or(0.0);
            let hosts: Vec<String> = item
                .get_item("hosts")
                .ok()
                .and_then(|v| v.extract().ok())
                .unwrap_or_default();
            let address: String = item
                .get_item("address")
                .ok()
                .and_then(|v| v.extract().ok())
                .unwrap_or_default();
            g.nodes.insert(
                node_id.clone(),
                NodeRecord {
                    node_id,
                    alive,
                    cpus,
                    gpus,
                    address,
                    hosts,
                    updated_at: now,
                },
            );
        }
        Ok(())
    }

    /// Replace the cluster-wide resource utilization view (from Ray).
    #[pyo3(signature = (resources_json, _now=0.0))]
    fn set_cluster_resources(&self, resources_json: &str, _now: f64) {
        let mut g = REGISTRY.lock().unwrap();
        g.cluster_resources = serde_json::from_str(resources_json).unwrap_or_else(|_| json!({}));
    }

    /// Full JSON snapshot for the console / HTTP endpoint / frontend.
    fn snapshot_json(&self) -> String {
        let g = REGISTRY.lock().unwrap();
        let stages: Vec<&StageRecord> = {
            let mut v: Vec<&StageRecord> = g.stages.values().collect();
            v.sort_by_key(|s| s.id);
            v
        };
        let pools: Vec<&PoolRecord> = {
            let mut v: Vec<&PoolRecord> = g.pools.values().collect();
            v.sort_by(|a, b| a.key.cmp(&b.key));
            v
        };
        let nodes: Vec<&NodeRecord> = {
            let mut v: Vec<&NodeRecord> = g.nodes.values().collect();
            v.sort_by(|a, b| a.node_id.cmp(&b.node_id));
            v
        };
        let running = g.queries.iter().filter(|q| q.status == "running").count();
        let snap = json!({
            "queries": g.queries,
            "stages": stages,
            "pools": pools,
            "nodes": nodes,
            "cluster": g.cluster_resources,
            "events": g.events,
            "summary": {
                "queries_total": g.queries.len(),
                "queries_running": running,
                "pools": g.pools.len(),
                "nodes_alive": g.nodes.values().filter(|n| n.alive).count(),
                "nodes_total": g.nodes.len(),
            }
        });
        serde_json::to_string(&snap).unwrap_or_else(|_| "{}".into())
    }

    /// Reset all state (tests / a fresh session).
    fn reset(&self) {
        let mut g = REGISTRY.lock().unwrap();
        g.seq = 0;
        g.queries.clear();
        g.stages.clear();
        g.pools.clear();
        g.nodes.clear();
        g.cluster_resources = json!({});
        g.events.clear();
    }
}

/// Register the `jude.observe` submodule.
pub fn register_bound(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MetricsRegistry>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    // Logic tests avoid pyclass methods (which would pull PyO3 symbols into the
    // test binary); they exercise Inner directly.
    use super::*;

    fn fresh() -> Inner {
        Inner {
            seq: 0,
            queries: Vec::new(),
            stages: HashMap::new(),
            pools: HashMap::new(),
            nodes: HashMap::new(),
            cluster_resources: json!({}),
            events: Vec::new(),
            max_events: 4,
            max_queries: 3,
        }
    }

    #[test]
    fn event_log_is_bounded() {
        let mut inner = fresh();
        for i in 0..10 {
            inner.push_event(i as f64, "k", format!("e{i}"));
        }
        assert_eq!(inner.events.len(), 4);
        // oldest dropped, newest kept
        assert_eq!(inner.events.last().unwrap()["msg"], "e9");
        assert_eq!(inner.events.first().unwrap()["msg"], "e6");
    }

    #[test]
    fn trim_keeps_running_and_newest_finished() {
        let mut inner = fresh();
        // 2 running, 4 finished -> max_queries=3 keeps 2 running + 1 newest finished
        for i in 0..4 {
            let id = inner.next_id();
            inner.queries.push(QueryRecord {
                id,
                label: format!("f{i}"),
                status: "done".into(),
                kind: "local".into(),
                started_at: i as f64,
                ended_at: Some(i as f64),
                rows: 0,
                bytes: 0,
                error: None,
                stage_ids: vec![],
            });
        }
        for i in 0..2 {
            let id = inner.next_id();
            inner.queries.push(QueryRecord {
                id,
                label: format!("r{i}"),
                status: "running".into(),
                kind: "local".into(),
                started_at: 100.0,
                ended_at: None,
                rows: 0,
                bytes: 0,
                error: None,
                stage_ids: vec![],
            });
        }
        inner.trim_queries();
        assert_eq!(inner.queries.len(), 3);
        let running = inner
            .queries
            .iter()
            .filter(|q| q.status == "running")
            .count();
        assert_eq!(running, 2, "never drop a running query");
        // the one kept finished query is the newest (f3)
        let fin: Vec<&str> = inner
            .queries
            .iter()
            .filter(|q| q.status == "done")
            .map(|q| q.label.as_str())
            .collect();
        assert_eq!(fin, vec!["f3"]);
    }

    #[test]
    fn trim_evicts_stages_of_dropped_queries() {
        let mut inner = fresh();
        for i in 0..4 {
            let id = inner.next_id();
            let sid = inner.next_id();
            inner.stages.insert(
                sid,
                StageRecord {
                    id: sid,
                    query_id: id,
                    name: "s".into(),
                    status: "done".into(),
                    started_at: 0.0,
                    ended_at: Some(0.0),
                    tasks_total: 1,
                    tasks_done: 1,
                    tasks_failed: 0,
                    attempts: 0,
                    rows: 0,
                    bytes: 0,
                },
            );
            inner.queries.push(QueryRecord {
                id,
                label: format!("f{i}"),
                status: "done".into(),
                kind: "distributed".into(),
                started_at: i as f64,
                ended_at: Some(i as f64),
                rows: 0,
                bytes: 0,
                error: None,
                stage_ids: vec![sid],
            });
        }
        assert_eq!(inner.stages.len(), 4);
        inner.trim_queries();
        // max_queries=3, 0 running -> keep 3 newest, evict 1 -> its stage gone
        assert_eq!(inner.queries.len(), 3);
        assert_eq!(inner.stages.len(), 3);
    }
}
