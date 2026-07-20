//! WorkerManager — the distributed scheduling brain, in Rust.
//!
//! Vane's distributed control plane is ~28.7k lines of GIL-bound Python. jude's
//! thesis is to put the *decisions* in Rust and leave Python as a thin Ray RPC
//! executor. This type owns all scheduling state and algorithms:
//!
//! - partition sizing (Spark-style size grouping over an open-cost byte target),
//! - the row-slice partition plan,
//! - round-robin worker assignment,
//! - the bounded-backlog in-flight window policy,
//! - hash-shuffle bucket routing for joins (via `HashSplitAssigner`).
//!
//! The Python shim (`jude.runners._ray_shim`) only forwards the resulting
//! decisions to Ray — it computes nothing.

use pyo3::prelude::*;

use crate::dist::fte::FteSplit;
use crate::dist::split_assigner::{HashSplitAssigner, SplitAssigner};

/// Scheduling brain for the Ray runner. All fields are decisions/config; no Ray
/// handles live here (those stay in the Python shim).
#[pyclass(module = "jude.dist")]
pub struct WorkerManager {
    #[pyo3(get)]
    num_workers: usize,
    #[pyo3(get)]
    num_gpus_per_worker: i64,
    #[pyo3(get)]
    size_grouping: bool,
    #[pyo3(get)]
    max_task_backlog: i64,
    #[pyo3(get)]
    open_cost_bytes: u64,
    #[pyo3(get)]
    min_partition_num: usize,
    /// Optional worker→node-address map (worker index i lives on node
    /// `worker_nodes[i]`). Empty when locality isn't tracked; used by
    /// `worker_for_locality` to co-locate a task with its data.
    worker_nodes: Vec<String>,
    /// Round-robin cursor per node, so locality assignment still balances load
    /// across workers on the same node. Keyed by node address.
    node_cursors: std::sync::Mutex<std::collections::HashMap<String, usize>>,
}

#[pymethods]
impl WorkerManager {
    /// Mirrors `RayRunner.__init__`'s config surface. Defaults match the Python
    /// runner (open_cost 4 MiB, size_grouping on, unbounded backlog).
    #[new]
    #[pyo3(signature = (
        num_workers,
        num_gpus_per_worker = 0,
        size_grouping = true,
        max_task_backlog = 0,
        open_cost_bytes = 4 * 1024 * 1024,
        min_partition_num = 0,
        worker_nodes = None,
    ))]
    fn new(
        num_workers: usize,
        num_gpus_per_worker: i64,
        size_grouping: bool,
        max_task_backlog: i64,
        open_cost_bytes: u64,
        min_partition_num: usize,
        worker_nodes: Option<Vec<String>>,
    ) -> Self {
        Self {
            num_workers: num_workers.max(1),
            num_gpus_per_worker,
            size_grouping,
            max_task_backlog,
            open_cost_bytes: open_cost_bytes.max(1),
            min_partition_num,
            worker_nodes: worker_nodes.unwrap_or_default(),
            node_cursors: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Spark-style target partition count from data size + worker count.
    ///
    /// Exact port of `RayRunner._target_partitions`: floor at
    /// `min_partition_num or num_workers`; with size grouping off or an empty
    /// table, `max(floor, num_workers)`; else also at least
    /// `ceil(nbytes / open_cost_bytes)`.
    fn target_partitions(&self, nbytes: u64, num_rows: u64) -> usize {
        let floor = if self.min_partition_num > 0 {
            self.min_partition_num
        } else {
            self.num_workers
        };
        if !self.size_grouping || num_rows == 0 {
            return floor.max(self.num_workers);
        }
        let by_size = nbytes.div_ceil(self.open_cost_bytes).max(1) as usize;
        floor.max(self.num_workers).max(by_size)
    }

    /// The row `(start, len)` slices for a relation. `hint > 1` (a repartition
    /// hint) pins the count; otherwise size grouping decides. Python performs
    /// the actual `table.slice`. Port of `RayRunner._partition_tables` slicing.
    fn partition_plan(&self, num_rows: usize, nbytes: u64, hint: usize) -> Vec<(usize, usize)> {
        let n = if hint > 1 {
            hint
        } else {
            self.target_partitions(nbytes, num_rows as u64)
        };
        let n = n.max(1);
        if num_rows == 0 {
            return vec![(0, 0)];
        }
        let step = num_rows.div_ceil(n);
        let mut out = Vec::new();
        let mut s = 0;
        while s < num_rows {
            out.push((s, step.min(num_rows - s)));
            s += step;
        }
        out
    }

    /// Round-robin worker index for the `task_index`-th task.
    fn worker_for(&self, task_index: usize) -> usize {
        task_index % self.num_workers
    }

    /// Locality-aware worker choice: given a task's data-locality hints
    /// (`addresses` — node addresses that host the split), pick a worker on a
    /// matching node, round-robining among the workers of that node to balance
    /// load. Falls back to plain round-robin when there's no worker→node map or
    /// no hint matches. This is where `FteSplit.addresses` finally gets used.
    #[pyo3(signature = (task_index, addresses))]
    fn worker_for_locality(&self, task_index: usize, addresses: Vec<String>) -> usize {
        if self.worker_nodes.is_empty() || addresses.is_empty() {
            return self.worker_for(task_index);
        }
        // Workers whose node is one of the preferred addresses.
        let local: Vec<usize> = (0..self.num_workers)
            .filter(|&w| {
                self.worker_nodes
                    .get(w)
                    .map(|node| addresses.iter().any(|a| a == node))
                    .unwrap_or(false)
            })
            .collect();
        if local.is_empty() {
            return self.worker_for(task_index);
        }
        // Round-robin within the matching node's workers (keyed by the first
        // matching address for a stable cursor).
        let key = addresses[0].clone();
        let mut cursors = self.node_cursors.lock().unwrap();
        let c = cursors.entry(key).or_insert(0);
        let chosen = local[*c % local.len()];
        *c += 1;
        chosen
    }

    /// The bounded-backlog in-flight window for `n_tasks`. `0` means unbounded
    /// (submit everything at once); otherwise the shim keeps this many tasks in
    /// flight. Port of the policy branch of `RayRunner._dispatch_bounded`.
    fn dispatch_window(&self, n_tasks: usize) -> usize {
        if self.max_task_backlog <= 0 || (self.max_task_backlog as usize) >= n_tasks {
            0
        } else {
            self.max_task_backlog as usize
        }
    }

    /// Bucket count for a hash-shuffle join (`num_buckets or num_workers`).
    #[pyo3(signature = (num_buckets = None))]
    fn shuffle_bucket_count(&self, num_buckets: Option<usize>) -> usize {
        num_buckets
            .filter(|&b| b > 0)
            .unwrap_or(self.num_workers)
            .max(1)
    }

    /// Worker assignment per bucket for a hash-shuffle join: element `i` is the
    /// worker index that should join bucket `i`. Driven through the ported
    /// `HashSplitAssigner` so the routing decision lives in the Rust assigner,
    /// not in Python.
    #[pyo3(signature = (num_buckets = None))]
    fn shuffle_bucket_workers(&self, num_buckets: Option<usize>) -> Vec<usize> {
        let b = self.shuffle_bucket_count(num_buckets);
        // Route synthetic one-split-per-bucket inputs through HashSplitAssigner
        // to obtain the canonical partition set, then round-robin them onto
        // workers. This keeps the "which partition" decision in the assigner.
        let mut assigner = HashSplitAssigner::new(b as u32);
        let splits: Vec<FteSplit> = (0..b)
            .map(|i| {
                let mut s = FteSplit::exchange("shuffle", i as u64, "", i as u32);
                s.source_partition_id = i as u32;
                s
            })
            .collect();
        let result = assigner.assign("shuffle", splits, true);
        // Canonical bucket ids are 0..b (created partitions); assign workers
        // round-robin over them in id order.
        let mut ids: Vec<u32> = result.partitions_added;
        ids.sort_unstable();
        ids.into_iter()
            .map(|pid| (pid as usize) % self.num_workers)
            .collect()
    }
}

/// Default partition count for a dataset of `num_rows`/`nbytes`, using the same
/// Spark-style size grouping as the runner but without needing a live cluster.
/// Used by non-runner call sites (e.g. distributed writes) that want jude's
/// partitioning decision from Rust. Worker floor is the local CPU count.
pub fn default_partition_count(num_rows: usize, nbytes: u64) -> usize {
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .max(1);
    let mgr = WorkerManager::new(workers, 0, true, 0, 4 * 1024 * 1024, 0, None);
    mgr.partition_plan(num_rows, nbytes, 1).len().max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mgr(
        workers: usize,
        size_grouping: bool,
        backlog: i64,
        open_cost: u64,
        min_part: usize,
    ) -> WorkerManager {
        WorkerManager::new(
            workers,
            0,
            size_grouping,
            backlog,
            open_cost,
            min_part,
            None,
        )
    }

    #[test]
    fn target_partitions_size_grouping_off() {
        let m = mgr(4, false, 0, 4 * 1024 * 1024, 0);
        // Grouping off => max(floor, num_workers) regardless of size.
        assert_eq!(m.target_partitions(1_000_000_000, 1000), 4);
    }

    #[test]
    fn target_partitions_empty_table() {
        let m = mgr(4, true, 0, 4 * 1024 * 1024, 0);
        assert_eq!(m.target_partitions(0, 0), 4);
    }

    #[test]
    fn target_partitions_by_size() {
        let m = mgr(2, true, 0, 1000, 0);
        // 10_000 bytes / 1000 = 10 tasks, above worker floor of 2.
        assert_eq!(m.target_partitions(10_000, 500), 10);
        // Small data still floored at num_workers.
        assert_eq!(m.target_partitions(100, 5), 2);
    }

    #[test]
    fn target_partitions_min_floor() {
        let m = mgr(2, true, 0, 4 * 1024 * 1024, 6);
        // min_partition_num floor of 6 dominates.
        assert_eq!(m.target_partitions(0, 0), 6);
        assert_eq!(m.target_partitions(100, 10), 6);
    }

    #[test]
    fn partition_plan_slices_cover_all_rows() {
        let m = mgr(3, false, 0, 4 * 1024 * 1024, 0);
        let plan = m.partition_plan(100, 0, 4); // hint pins 4 partitions
        assert_eq!(plan.len(), 4);
        let total: usize = plan.iter().map(|(_, l)| l).sum();
        assert_eq!(total, 100);
        // Contiguous, non-overlapping.
        let mut expected = 0;
        for (s, l) in &plan {
            assert_eq!(*s, expected);
            expected += l;
        }
    }

    #[test]
    fn partition_plan_empty() {
        let m = mgr(3, true, 0, 4 * 1024 * 1024, 0);
        assert_eq!(m.partition_plan(0, 0, 1), vec![(0, 0)]);
    }

    #[test]
    fn dispatch_window_bounds() {
        let m = mgr(4, true, 2, 4 * 1024 * 1024, 0);
        assert_eq!(m.dispatch_window(10), 2); // bounded
        assert_eq!(m.dispatch_window(2), 0); // limit >= n => unbounded
        let unbounded = mgr(4, true, 0, 4 * 1024 * 1024, 0);
        assert_eq!(unbounded.dispatch_window(100), 0);
    }

    #[test]
    fn worker_for_round_robin() {
        let m = mgr(3, true, 0, 4 * 1024 * 1024, 0);
        let assigns: Vec<usize> = (0..7).map(|i| m.worker_for(i)).collect();
        assert_eq!(assigns, vec![0, 1, 2, 0, 1, 2, 0]);
    }

    #[test]
    fn shuffle_buckets() {
        let m = mgr(4, true, 0, 4 * 1024 * 1024, 0);
        assert_eq!(m.shuffle_bucket_count(None), 4);
        assert_eq!(m.shuffle_bucket_count(Some(8)), 8);
        let workers = m.shuffle_bucket_workers(Some(6));
        assert_eq!(workers.len(), 6);
        // buckets 0..6 round-robin over 4 workers.
        assert_eq!(workers, vec![0, 1, 2, 3, 0, 1]);
    }

    #[test]
    fn worker_for_locality_prefers_matching_node() {
        // 4 workers across 2 nodes: workers 0,2 on nodeA; 1,3 on nodeB.
        let m = WorkerManager::new(
            4,
            0,
            true,
            0,
            4 * 1024 * 1024,
            0,
            Some(vec![
                "nodeA".into(),
                "nodeB".into(),
                "nodeA".into(),
                "nodeB".into(),
            ]),
        );
        // A task whose data is on nodeB must land on a nodeB worker (1 or 3),
        // round-robining between them.
        let w1 = m.worker_for_locality(0, vec!["nodeB".into()]);
        let w2 = m.worker_for_locality(0, vec!["nodeB".into()]);
        assert!(w1 == 1 || w1 == 3);
        assert!(w2 == 1 || w2 == 3);
        assert_ne!(w1, w2); // balanced across the node's two workers

        // No matching node -> plain round-robin fallback.
        assert_eq!(m.worker_for_locality(2, vec!["nodeC".into()]), 2 % 4);
        // No hint -> fallback.
        assert_eq!(m.worker_for_locality(3, vec![]), 3 % 4);
    }

    #[test]
    fn worker_for_locality_no_map_is_round_robin() {
        let m = mgr(3, true, 0, 4 * 1024 * 1024, 0); // no worker_nodes
        assert_eq!(m.worker_for_locality(4, vec!["nodeA".into()]), 4 % 3);
    }
}
