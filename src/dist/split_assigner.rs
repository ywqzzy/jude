//! Split assigners — pack input splits into task partitions (Rust).
//!
//! Ported from Vane's `fte_split_assigner.py`. The assigner turns a stream of
//! `FteSplit`s into task partitions. `ArbitrarySplitAssigner` does size-based
//! bin-packing with adaptive growth (early partitions small for fast first
//! results, later ones larger for throughput). `HashSplitAssigner` routes by
//! `source_partition_id` for hash shuffles.

use std::collections::HashSet;

use crate::dist::fte::{FteSplit, STANDARD_SPLIT_SIZE_BYTES};

/// A per-partition update emitted by an assigner.
#[derive(Clone, Debug, Default)]
pub struct PartitionUpdate {
    pub partition_id: u32,
    pub source_node_id: String,
    pub splits: Vec<FteSplit>,
    pub no_more_splits: bool,
    pub ready_for_scheduling: bool,
}

#[derive(Clone, Debug, Default)]
pub struct AssignmentResult {
    pub partitions_added: Vec<u32>,
    pub partition_updates: Vec<PartitionUpdate>,
    pub sealed_partitions: Vec<u32>,
    pub no_more_partitions: bool,
}

impl AssignmentResult {
    pub fn merge(&mut self, other: AssignmentResult) {
        self.partitions_added.extend(other.partitions_added);
        self.partition_updates.extend(other.partition_updates);
        self.sealed_partitions.extend(other.sealed_partitions);
        self.no_more_partitions |= other.no_more_partitions;
    }
}

pub trait SplitAssigner {
    fn assign(
        &mut self,
        source_node_id: &str,
        splits: Vec<FteSplit>,
        no_more_inputs: bool,
    ) -> AssignmentResult;
}

/// All splits go to partition 0.
pub struct SingleSplitAssigner {
    created: bool,
    sources: HashSet<String>,
    completed: HashSet<String>,
}

impl Default for SingleSplitAssigner {
    fn default() -> Self {
        Self {
            created: false,
            sources: HashSet::new(),
            completed: HashSet::new(),
        }
    }
}

impl SplitAssigner for SingleSplitAssigner {
    fn assign(
        &mut self,
        source_node_id: &str,
        splits: Vec<FteSplit>,
        no_more_inputs: bool,
    ) -> AssignmentResult {
        self.sources.insert(source_node_id.to_string());
        let mut r = AssignmentResult::default();
        if !self.created {
            self.created = true;
            r.partitions_added.push(0);
        }
        r.partition_updates.push(PartitionUpdate {
            partition_id: 0,
            source_node_id: source_node_id.to_string(),
            splits,
            ready_for_scheduling: true,
            ..Default::default()
        });
        if no_more_inputs {
            self.completed.insert(source_node_id.to_string());
            r.partition_updates.push(PartitionUpdate {
                partition_id: 0,
                source_node_id: source_node_id.to_string(),
                no_more_splits: true,
                ..Default::default()
            });
            if self.sources.is_subset(&self.completed) {
                r.sealed_partitions.push(0);
                r.no_more_partitions = true;
            }
        }
        r
    }
}

/// Route each split to the partition matching its `source_partition_id`
/// (mod num_partitions) — for hash shuffles.
pub struct HashSplitAssigner {
    num_partitions: u32,
    created: HashSet<u32>,
    sources: HashSet<String>,
    completed: HashSet<String>,
}

impl HashSplitAssigner {
    pub fn new(num_partitions: u32) -> Self {
        Self {
            num_partitions: num_partitions.max(1),
            created: HashSet::new(),
            sources: HashSet::new(),
            completed: HashSet::new(),
        }
    }
}

impl SplitAssigner for HashSplitAssigner {
    fn assign(
        &mut self,
        source_node_id: &str,
        splits: Vec<FteSplit>,
        no_more_inputs: bool,
    ) -> AssignmentResult {
        self.sources.insert(source_node_id.to_string());
        let mut r = AssignmentResult::default();
        for pid in 0..self.num_partitions {
            if self.created.insert(pid) {
                r.partitions_added.push(pid);
            }
        }
        let mut by_part: std::collections::HashMap<u32, Vec<FteSplit>> =
            std::collections::HashMap::new();
        for s in splits {
            by_part
                .entry(s.source_partition_id % self.num_partitions)
                .or_default()
                .push(s);
        }
        for (pid, part_splits) in by_part {
            r.partition_updates.push(PartitionUpdate {
                partition_id: pid,
                source_node_id: source_node_id.to_string(),
                splits: part_splits,
                ready_for_scheduling: true,
                ..Default::default()
            });
        }
        if no_more_inputs {
            self.completed.insert(source_node_id.to_string());
            for pid in 0..self.num_partitions {
                r.partition_updates.push(PartitionUpdate {
                    partition_id: pid,
                    source_node_id: source_node_id.to_string(),
                    no_more_splits: true,
                    ..Default::default()
                });
            }
            if self.sources.is_subset(&self.completed) {
                r.sealed_partitions.extend(0..self.num_partitions);
                r.no_more_partitions = true;
            }
        }
        r
    }
}

struct Part {
    partition_id: u32,
    data_size: u64,
    split_count: u64,
}

/// Size-based bin-packing with adaptive growth (the workhorse assigner).
#[allow(dead_code)]
pub struct ArbitrarySplitAssigner {
    standard: u64,
    max_splits: u64,
    growth_period: u32,
    growth_factor: f64,
    min_target: u64,
    max_target: u64,
    target: u64,
    adaptive_counter: u32,
    next_pid: u32,
    replicated_sources: HashSet<String>,
    replicated_splits: std::collections::HashMap<String, Vec<FteSplit>>,
    completed: HashSet<String>,
    open: Option<Part>,
    all_parts: Vec<u32>,
}

impl ArbitrarySplitAssigner {
    pub fn new(
        replicated_sources: HashSet<String>,
        max_task_split_count: u64,
        standard_split_size_bytes: u64,
        min_target_partition_size_bytes: Option<u64>,
        max_target_partition_size_bytes: Option<u64>,
        adaptive_growth_period: u32,
        adaptive_growth_factor: f64,
    ) -> Self {
        let max_splits = max_task_split_count.max(1);
        let standard = standard_split_size_bytes.max(1);
        let min_target = min_target_partition_size_bytes
            .unwrap_or(max_splits * standard)
            .max(1);
        let max_target = max_target_partition_size_bytes
            .unwrap_or(min_target)
            .max(min_target);
        Self {
            standard,
            max_splits,
            growth_period: adaptive_growth_period.max(1),
            growth_factor: adaptive_growth_factor.max(1.0),
            min_target,
            max_target,
            target: min_target,
            adaptive_counter: 0,
            next_pid: 0,
            replicated_sources,
            replicated_splits: std::collections::HashMap::new(),
            completed: HashSet::new(),
            open: None,
            all_parts: Vec::new(),
        }
    }

    /// Vane defaults: 64MiB standard, 2048 split cap, growth 1.26x every 64.
    pub fn with_defaults(replicated_sources: HashSet<String>) -> Self {
        Self::new(
            replicated_sources,
            2048,
            STANDARD_SPLIT_SIZE_BYTES,
            None,
            None,
            64,
            1.26,
        )
    }

    fn grow_target(&mut self) {
        self.adaptive_counter += 1;
        if self.adaptive_counter >= self.growth_period {
            self.adaptive_counter = 0;
            self.target = ((self.target as f64 * self.growth_factor) as u64).min(self.max_target);
        }
    }
}

impl SplitAssigner for ArbitrarySplitAssigner {
    fn assign(
        &mut self,
        source_node_id: &str,
        splits: Vec<FteSplit>,
        no_more_inputs: bool,
    ) -> AssignmentResult {
        if self.replicated_sources.contains(source_node_id) {
            let mut r = AssignmentResult::default();
            self.replicated_splits
                .entry(source_node_id.to_string())
                .or_default()
                .extend(splits.clone());
            for &pid in &self.all_parts {
                r.partition_updates.push(PartitionUpdate {
                    partition_id: pid,
                    source_node_id: source_node_id.to_string(),
                    splits: splits.clone(),
                    no_more_splits: no_more_inputs,
                    ..Default::default()
                });
            }
            if no_more_inputs {
                self.completed.insert(source_node_id.to_string());
            }
            return r;
        }

        let mut r = AssignmentResult::default();
        for split in splits {
            let size = split.effective_size();
            let need_new = match &self.open {
                Some(p) => p.data_size + size > self.target || p.split_count + 1 > self.max_splits,
                None => true,
            };
            if need_new {
                if let Some(p) = self.open.take() {
                    r.sealed_partitions.push(p.partition_id);
                    self.grow_target();
                }
                let pid = self.next_pid;
                self.next_pid += 1;
                self.all_parts.push(pid);
                self.open = Some(Part {
                    partition_id: pid,
                    data_size: 0,
                    split_count: 0,
                });
                r.partitions_added.push(pid);
                // attach already-seen replicated splits
                for (rsrc, rsplits) in &self.replicated_splits {
                    r.partition_updates.push(PartitionUpdate {
                        partition_id: pid,
                        source_node_id: rsrc.clone(),
                        splits: rsplits.clone(),
                        no_more_splits: self.completed.contains(rsrc),
                        ..Default::default()
                    });
                }
            }
            let p = self.open.as_mut().unwrap();
            r.partition_updates.push(PartitionUpdate {
                partition_id: p.partition_id,
                source_node_id: source_node_id.to_string(),
                splits: vec![split],
                ready_for_scheduling: true,
                ..Default::default()
            });
            p.data_size += size;
            p.split_count += 1;
        }

        if no_more_inputs {
            self.completed.insert(source_node_id.to_string());
            let parts = self.all_parts.clone();
            for pid in parts {
                r.partition_updates.push(PartitionUpdate {
                    partition_id: pid,
                    source_node_id: source_node_id.to_string(),
                    no_more_splits: true,
                    ..Default::default()
                });
            }
            if let Some(p) = self.open.take() {
                r.sealed_partitions.push(p.partition_id);
            }
            r.no_more_partitions = true;
        }
        r
    }
}
