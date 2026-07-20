//! ResourceManager — multi-dimensional resource admission control, in Rust.
//!
//! Vane's `QueryResourceManager` is ~2.7k lines of GIL-bound Python that admits
//! tasks against a 4-dim resource vector (cpu / gpu / heap / object-store) with
//! output-block leasing for backpressure. jude puts the same *decisions* in Rust
//! and keeps Python as a thin executor.
//!
//! This is the incremental core that matters for GPU batch inference: reserve
//! capacity before launching a task and release it on completion, so a fleet of
//! model-inference tasks never oversubscribes GPUs, host memory, or the Ray
//! object store (the two most common ways such a pipeline OOMs / stalls).

use std::sync::Mutex;

use pyo3::prelude::*;

/// A point in resource space. All dimensions are additive; `cpu`/`gpu` are
/// fractional (Ray-style), byte dimensions are absolute.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ResourceVector {
    pub cpu: f64,
    pub gpu: f64,
    pub memory_bytes: i64,
    pub object_store_bytes: i64,
}

impl ResourceVector {
    fn fits_within(&self, cap: &ResourceVector) -> bool {
        self.cpu <= cap.cpu + 1e-9
            && self.gpu <= cap.gpu + 1e-9
            && self.memory_bytes <= cap.memory_bytes
            && self.object_store_bytes <= cap.object_store_bytes
    }
}

struct Inner {
    /// Total capacity across the cluster.
    capacity: ResourceVector,
    /// Currently reserved (sum of live leases).
    used: ResourceVector,
    /// Hard cap on concurrently admitted tasks (0 = unlimited).
    max_inflight: usize,
    /// Number of tasks currently admitted (reserved, not yet released).
    inflight: usize,
}

impl Inner {
    fn available(&self) -> ResourceVector {
        ResourceVector {
            cpu: (self.capacity.cpu - self.used.cpu).max(0.0),
            gpu: (self.capacity.gpu - self.used.gpu).max(0.0),
            memory_bytes: (self.capacity.memory_bytes - self.used.memory_bytes).max(0),
            object_store_bytes: (self.capacity.object_store_bytes - self.used.object_store_bytes)
                .max(0),
        }
    }
}

/// Resource admission brain. Thread-safe (all state behind a Mutex); every
/// method is a pure decision or a reserve/release mutation — no Ray handles.
#[pyclass(module = "jude.dist")]
pub struct ResourceManager {
    inner: Mutex<Inner>,
}

#[pymethods]
impl ResourceManager {
    /// Capacity across the cluster. `max_inflight` caps concurrently admitted
    /// tasks regardless of per-task demand (0 = unlimited). A byte capacity of 0
    /// means "unconstrained on that dimension" (treated as effectively infinite).
    #[new]
    #[pyo3(signature = (cpu=0.0, gpu=0.0, memory_bytes=0, object_store_bytes=0, max_inflight=0))]
    fn new(
        cpu: f64,
        gpu: f64,
        memory_bytes: i64,
        object_store_bytes: i64,
        max_inflight: usize,
    ) -> Self {
        let big = i64::MAX / 4;
        Self {
            inner: Mutex::new(Inner {
                capacity: ResourceVector {
                    cpu: if cpu > 0.0 { cpu } else { f64::INFINITY },
                    gpu: if gpu > 0.0 { gpu } else { f64::INFINITY },
                    memory_bytes: if memory_bytes > 0 { memory_bytes } else { big },
                    object_store_bytes: if object_store_bytes > 0 {
                        object_store_bytes
                    } else {
                        big
                    },
                },
                used: ResourceVector::default(),
                max_inflight,
                inflight: 0,
            }),
        }
    }

    /// Try to reserve one task's demand. Returns True and records the lease if it
    /// fits within remaining capacity and the in-flight cap; False otherwise.
    #[pyo3(signature = (cpu=0.0, gpu=0.0, memory_bytes=0, object_store_bytes=0))]
    fn try_reserve(&self, cpu: f64, gpu: f64, memory_bytes: i64, object_store_bytes: i64) -> bool {
        let demand = ResourceVector {
            cpu,
            gpu,
            memory_bytes,
            object_store_bytes,
        };
        let mut g = self.inner.lock().unwrap();
        if g.max_inflight > 0 && g.inflight >= g.max_inflight {
            return false;
        }
        if !demand.fits_within(&g.available()) {
            return false;
        }
        g.used.cpu += demand.cpu;
        g.used.gpu += demand.gpu;
        g.used.memory_bytes += demand.memory_bytes;
        g.used.object_store_bytes += demand.object_store_bytes;
        g.inflight += 1;
        true
    }

    /// Reserve unconditionally — used to guarantee forward progress when a
    /// single task's demand exceeds total capacity (it runs alone rather than
    /// deadlocking). Records the lease regardless of remaining capacity.
    #[pyo3(signature = (cpu=0.0, gpu=0.0, memory_bytes=0, object_store_bytes=0))]
    fn reserve(&self, cpu: f64, gpu: f64, memory_bytes: i64, object_store_bytes: i64) {
        let mut g = self.inner.lock().unwrap();
        g.used.cpu += cpu;
        g.used.gpu += gpu;
        g.used.memory_bytes += memory_bytes;
        g.used.object_store_bytes += object_store_bytes;
        g.inflight += 1;
    }

    /// Release one previously-reserved lease (call on task completion).
    #[pyo3(signature = (cpu=0.0, gpu=0.0, memory_bytes=0, object_store_bytes=0))]
    fn release(&self, cpu: f64, gpu: f64, memory_bytes: i64, object_store_bytes: i64) {
        let mut g = self.inner.lock().unwrap();
        g.used.cpu = (g.used.cpu - cpu).max(0.0);
        g.used.gpu = (g.used.gpu - gpu).max(0.0);
        g.used.memory_bytes = (g.used.memory_bytes - memory_bytes).max(0);
        g.used.object_store_bytes = (g.used.object_store_bytes - object_store_bytes).max(0);
        if g.inflight > 0 {
            g.inflight -= 1;
        }
    }

    /// How many more tasks of the given per-task demand can be admitted right now
    /// — the min over resource dimensions and the remaining in-flight budget.
    /// Pure (does not reserve); the dispatcher uses it to size the next wave.
    #[pyo3(signature = (cpu=0.0, gpu=0.0, memory_bytes=0, object_store_bytes=0))]
    fn admit_count(&self, cpu: f64, gpu: f64, memory_bytes: i64, object_store_bytes: i64) -> usize {
        let g = self.inner.lock().unwrap();
        let avail = g.available();
        let mut n = usize::MAX;
        if cpu > 0.0 {
            n = n.min((avail.cpu / cpu).floor() as usize);
        }
        if gpu > 0.0 {
            n = n.min((avail.gpu / gpu).floor() as usize);
        }
        if memory_bytes > 0 {
            n = n.min((avail.memory_bytes / memory_bytes) as usize);
        }
        if object_store_bytes > 0 {
            n = n.min((avail.object_store_bytes / object_store_bytes) as usize);
        }
        if g.max_inflight > 0 {
            let budget = g.max_inflight.saturating_sub(g.inflight);
            n = n.min(budget);
        }
        if n == usize::MAX {
            // No binding dimension (all unconstrained) and no in-flight cap.
            usize::MAX
        } else {
            n
        }
    }

    /// (cpu, gpu, memory_bytes, object_store_bytes) currently available.
    fn available(&self) -> (f64, f64, i64, i64) {
        let g = self.inner.lock().unwrap();
        let a = g.available();
        (a.cpu, a.gpu, a.memory_bytes, a.object_store_bytes)
    }

    #[getter]
    fn inflight(&self) -> usize {
        self.inner.lock().unwrap().inflight
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_admission_bounds_wave_size() {
        // 4 GPUs total, each task wants 1 GPU -> at most 4 admitted at once.
        let m = ResourceManager::new(0.0, 4.0, 0, 0, 0);
        assert_eq!(m.admit_count(0.0, 1.0, 0, 0), 4);
        assert!(m.try_reserve(0.0, 1.0, 0, 0));
        assert!(m.try_reserve(0.0, 1.0, 0, 0));
        assert_eq!(m.admit_count(0.0, 1.0, 0, 0), 2);
        m.release(0.0, 1.0, 0, 0);
        assert_eq!(m.admit_count(0.0, 1.0, 0, 0), 3);
    }

    #[test]
    fn reserve_fails_when_oversubscribed() {
        let m = ResourceManager::new(0.0, 1.0, 0, 0, 0);
        assert!(m.try_reserve(0.0, 1.0, 0, 0));
        assert!(!m.try_reserve(0.0, 1.0, 0, 0)); // no GPU left
        assert_eq!(m.inflight(), 1);
    }

    #[test]
    fn memory_dimension_binds() {
        // 100 bytes cap, 30 bytes/task -> 3 fit.
        let m = ResourceManager::new(0.0, 0.0, 100, 0, 0);
        assert_eq!(m.admit_count(0.0, 0.0, 30, 0), 3);
    }

    #[test]
    fn max_inflight_caps_regardless_of_resources() {
        let m = ResourceManager::new(0.0, 0.0, 0, 0, 2); // unconstrained resources, cap 2
        assert_eq!(m.admit_count(0.0, 0.0, 0, 0), 2);
        assert!(m.try_reserve(0.0, 0.0, 0, 0));
        assert!(m.try_reserve(0.0, 0.0, 0, 0));
        assert!(!m.try_reserve(0.0, 0.0, 0, 0)); // in-flight cap hit
    }

    #[test]
    fn unconstrained_is_unbounded() {
        let m = ResourceManager::new(0.0, 0.0, 0, 0, 0);
        assert_eq!(m.admit_count(0.0, 0.0, 0, 0), usize::MAX);
    }
}
