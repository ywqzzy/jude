//! FTE (fault-tolerant execution) core data structures — Rust.
//!
//! The vocabulary of jude's distributed engine, ported from Vane's Python
//! `duckdb/runners/fte/fte_types.py` into Rust so the orchestration layer runs
//! GIL-free. Ray is only touched at the RPC boundary (a thin Python shim);
//! everything here — identity, splits, task descriptors — is pure Rust.

use std::collections::{HashMap, HashSet};

/// default split size when a split doesn't declare its own bytes (64 MiB).
pub const STANDARD_SPLIT_SIZE_BYTES: u64 = 64 * 1024 * 1024;

/// Split kinds.
pub const SCAN_TASK: &str = "scan_task";
pub const EXCHANGE_SOURCE_TASK: &str = "exchange_source_task";

/// Logical identity of a task: (query, fragment execution, partition).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FteTaskId {
    pub query_id: String,
    pub fragment_execution_id: u32,
    pub partition_id: u32,
}

impl FteTaskId {
    pub fn new(query_id: impl Into<String>, fragment_execution_id: u32, partition_id: u32) -> Self {
        Self {
            query_id: query_id.into(),
            fragment_execution_id,
            partition_id,
        }
    }
    pub fn as_string(&self) -> String {
        format!(
            "{}.{}.{}",
            self.query_id, self.fragment_execution_id, self.partition_id
        )
    }
}

/// A specific attempt of a task (retries / speculative execution).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FteTaskAttemptId {
    pub task_id: FteTaskId,
    pub attempt_id: u32,
}

impl FteTaskAttemptId {
    pub fn as_string(&self) -> String {
        format!("{}.{}", self.task_id.as_string(), self.attempt_id)
    }
}

/// One unit of input work for a fragment — a scan split or an upstream shuffle
/// output partition. Carries size (for byte-based packing) + locality hints.
#[derive(Clone, Debug)]
pub struct FteSplit {
    pub source_node_id: String,
    pub sequence_id: u64,
    pub kind: String,
    /// Opaque payload describing the work (e.g. a list of parquet paths, or a
    /// shuffle object-ref key). Stored as a string the worker interprets.
    pub data: String,
    pub source_partition_id: u32,
    pub size_bytes: Option<u64>,
    pub addresses: Vec<String>,
    pub remotely_accessible: bool,
}

impl FteSplit {
    pub fn scan(
        source_node_id: impl Into<String>,
        sequence_id: u64,
        data: impl Into<String>,
        size_bytes: Option<u64>,
    ) -> Self {
        Self {
            source_node_id: source_node_id.into(),
            sequence_id,
            kind: SCAN_TASK.to_string(),
            data: data.into(),
            source_partition_id: 0,
            size_bytes,
            addresses: Vec::new(),
            remotely_accessible: true,
        }
    }

    pub fn exchange(
        source_node_id: impl Into<String>,
        sequence_id: u64,
        data: impl Into<String>,
        source_partition_id: u32,
    ) -> Self {
        Self {
            source_node_id: source_node_id.into(),
            sequence_id,
            kind: EXCHANGE_SOURCE_TASK.to_string(),
            data: data.into(),
            source_partition_id,
            size_bytes: None,
            addresses: Vec::new(),
            remotely_accessible: true,
        }
    }

    pub fn effective_size(&self) -> u64 {
        self.size_bytes.unwrap_or(STANDARD_SPLIT_SIZE_BYTES)
    }
}

/// The mutable per-partition scheduling record.
///
/// Splits are appended incrementally as upstream fragments produce output;
/// every mutation bumps `descriptor_version` so a delta can be shipped to an
/// already-running worker. `no_more_splits` seals a source when exhausted.
#[derive(Clone, Debug)]
pub struct TaskDescriptor {
    pub task_id: FteTaskId,
    pub fragment_id: String,
    pub context: HashMap<String, String>,
    pub splits: HashMap<String, Vec<FteSplit>>,
    pub no_more_splits: HashSet<String>,
    pub descriptor_version: u64,
    source_node_ids: HashSet<String>,
    seen_sequences: HashMap<String, HashSet<u64>>,
}

impl TaskDescriptor {
    pub fn new(task_id: FteTaskId, fragment_id: impl Into<String>) -> Self {
        Self {
            task_id,
            fragment_id: fragment_id.into(),
            context: HashMap::new(),
            splits: HashMap::new(),
            no_more_splits: HashSet::new(),
            descriptor_version: 0,
            source_node_ids: HashSet::new(),
            seen_sequences: HashMap::new(),
        }
    }

    /// Append splits for a source, deduping by sequence_id. Returns how many new
    /// splits were added and bumps the version if any.
    pub fn append_splits(&mut self, source_node_id: &str, splits: Vec<FteSplit>) -> usize {
        let seen = self
            .seen_sequences
            .entry(source_node_id.to_string())
            .or_default();
        let bucket = self.splits.entry(source_node_id.to_string()).or_default();
        self.source_node_ids.insert(source_node_id.to_string());
        let mut added = 0;
        for s in splits {
            if seen.insert(s.sequence_id) {
                bucket.push(s);
                added += 1;
            }
        }
        if added > 0 {
            self.descriptor_version += 1;
        }
        added
    }

    pub fn seal_source(&mut self, source_node_id: &str) {
        if self.no_more_splits.insert(source_node_id.to_string()) {
            self.descriptor_version += 1;
        }
    }

    pub fn is_sealed(&self) -> bool {
        !self.source_node_ids.is_empty() && self.source_node_ids.is_subset(&self.no_more_splits)
    }

    pub fn total_splits(&self) -> usize {
        self.splits.values().map(|v| v.len()).sum()
    }
}
