//! Distributed execution engine — orchestration in Rust.
//!
//! The performance thesis: Vane's distributed control plane is ~28.7k lines of
//! Python (driver, scheduler, split assigner, resource manager) — all
//! GIL-bound. jude puts the orchestration in Rust: stage planning, split
//! packing, scheduling, and shuffle coordination all run here, GIL-free. Ray is
//! touched only at the RPC boundary through a thin Python shim.

pub mod cluster;
pub mod fte;
pub mod physical;
pub mod resource;
pub mod split_assigner;
pub mod stage;
pub mod worker_manager;

pub use fte::{
    FteSplit, FteTaskAttemptId, FteTaskId, TaskDescriptor, EXCHANGE_SOURCE_TASK, SCAN_TASK,
    STANDARD_SPLIT_SIZE_BYTES,
};
pub use split_assigner::{
    ArbitrarySplitAssigner, AssignmentResult, HashSplitAssigner, PartitionUpdate,
    SingleSplitAssigner, SplitAssigner,
};
pub use worker_manager::default_partition_count;
pub use worker_manager::WorkerManager;
