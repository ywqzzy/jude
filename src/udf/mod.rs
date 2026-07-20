//! Out-of-process UDF execution engine.
//!
//! A pool of persistent Python worker subprocesses, each running
//! `python -m jude.execution._worker`. Batches are shipped as Arrow IPC over
//! pipes; because each worker is a separate process with its own interpreter,
//! N workers give N-way real parallelism free of the GIL. The Rust pool
//! dispatches and collects with the GIL released, so the orchestration itself
//! never contends for the GIL either.

pub mod subprocess;

pub use subprocess::{get_or_create_pool, shutdown_all_pools, SubprocessPool};
