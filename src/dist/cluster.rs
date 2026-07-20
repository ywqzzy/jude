//! ClusterScheduler — cross-query placement / bin-packing, in Rust.
//!
//! Vane's `cluster_resource_coordinator.py` reads Ray node capacities and packs
//! actor/task bundles onto nodes so multiple queries share a cluster without
//! oversubscribing any single node. jude puts that placement decision in Rust;
//! the Python shim only reads `ray.state` node capacities and forwards them.
//!
//! Placement is worst-fit (balance load): each bundle goes to the node with the
//! most headroom that still fits it, falling back to the most-headroom node when
//! none fits (overflow is placed, not dropped — the caller can see it via
//! `loads()` / `overflow_count()`).

use std::sync::Mutex;

use pyo3::prelude::*;

#[derive(Clone)]
struct NodeState {
    id: String,
    cap_cpu: f64,
    cap_gpu: f64,
    cap_mem: i64,
    used_cpu: f64,
    used_gpu: f64,
    used_mem: i64,
}

impl NodeState {
    fn fits(&self, cpu: f64, gpu: f64, mem: i64) -> bool {
        self.used_cpu + cpu <= self.cap_cpu + 1e-9
            && self.used_gpu + gpu <= self.cap_gpu + 1e-9
            && self.used_mem + mem <= self.cap_mem
    }

    /// Headroom = the tightest remaining fraction across dimensions (0..1). A
    /// node with more headroom is preferred (worst-fit balances load).
    fn headroom(&self) -> f64 {
        let mut h = f64::INFINITY;
        if self.cap_cpu > 0.0 {
            h = h.min((self.cap_cpu - self.used_cpu) / self.cap_cpu);
        }
        if self.cap_gpu > 0.0 {
            h = h.min((self.cap_gpu - self.used_gpu) / self.cap_gpu);
        }
        if self.cap_mem > 0 {
            h = h.min((self.cap_mem - self.used_mem) as f64 / self.cap_mem as f64);
        }
        if h.is_infinite() {
            1.0
        } else {
            h
        }
    }
}

struct Inner {
    nodes: Vec<NodeState>,
    overflow: usize,
}

#[pyclass(module = "jude.dist")]
pub struct ClusterScheduler {
    inner: Mutex<Inner>,
}

impl ClusterScheduler {
    /// Pure placement (no pyo3): assign each bundle to a node by worst-fit,
    /// overflowing to the most-headroom node when nothing fits. Mutates load.
    /// Assumes at least one node exists (the pymethod checks).
    fn place_inner(&self, bundles: Vec<(f64, f64, i64)>) -> Vec<String> {
        let mut g = self.inner.lock().unwrap();
        let mut out = Vec::with_capacity(bundles.len());
        for (cpu, gpu, mem) in bundles {
            let mut best_fit: Option<usize> = None;
            let mut best_any: usize = 0;
            let mut best_fit_h = f64::NEG_INFINITY;
            let mut best_any_h = f64::NEG_INFINITY;
            for (i, n) in g.nodes.iter().enumerate() {
                let h = n.headroom();
                if h > best_any_h {
                    best_any_h = h;
                    best_any = i;
                }
                if n.fits(cpu, gpu, mem) && h > best_fit_h {
                    best_fit_h = h;
                    best_fit = Some(i);
                }
            }
            let idx = match best_fit {
                Some(i) => i,
                None => {
                    g.overflow += 1;
                    best_any
                }
            };
            let n = &mut g.nodes[idx];
            n.used_cpu += cpu;
            n.used_gpu += gpu;
            n.used_mem += mem;
            out.push(n.id.clone());
        }
        out
    }
}

#[pymethods]
impl ClusterScheduler {
    /// `nodes` is a list of (node_id, cpu, gpu, memory_bytes). A 0 capacity on a
    /// dimension means "unconstrained" there.
    #[new]
    fn new(nodes: Vec<(String, f64, f64, i64)>) -> Self {
        let big = i64::MAX / 4;
        let nodes = nodes
            .into_iter()
            .map(|(id, cpu, gpu, mem)| NodeState {
                id,
                cap_cpu: if cpu > 0.0 { cpu } else { f64::INFINITY },
                cap_gpu: if gpu > 0.0 { gpu } else { f64::INFINITY },
                cap_mem: if mem > 0 { mem } else { big },
                used_cpu: 0.0,
                used_gpu: 0.0,
                used_mem: 0,
            })
            .collect();
        Self {
            inner: Mutex::new(Inner { nodes, overflow: 0 }),
        }
    }

    /// Place each bundle (cpu, gpu, memory_bytes) on a node and return the chosen
    /// node id per bundle (in input order). Worst-fit for load balance; a bundle
    /// that fits nowhere still lands on the most-headroom node (counted as
    /// overflow). Mutates node load so successive calls share the cluster.
    fn place(&self, bundles: Vec<(f64, f64, i64)>) -> PyResult<Vec<String>> {
        {
            let g = self.inner.lock().unwrap();
            if g.nodes.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "ClusterScheduler has no nodes",
                ));
            }
        }
        Ok(self.place_inner(bundles))
    }

    /// Reset all node loads (start a fresh placement round).
    fn reset(&self) {
        let mut g = self.inner.lock().unwrap();
        for n in g.nodes.iter_mut() {
            n.used_cpu = 0.0;
            n.used_gpu = 0.0;
            n.used_mem = 0;
        }
        g.overflow = 0;
    }

    /// Per-node current load: (node_id, used_cpu, used_gpu, used_mem).
    fn loads(&self) -> Vec<(String, f64, f64, i64)> {
        let g = self.inner.lock().unwrap();
        g.nodes
            .iter()
            .map(|n| (n.id.clone(), n.used_cpu, n.used_gpu, n.used_mem))
            .collect()
    }

    /// How many bundles were placed as overflow (fit nowhere) since the last reset.
    #[getter]
    fn overflow_count(&self) -> usize {
        self.inner.lock().unwrap().overflow
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sched(nodes: &[(&str, f64, f64, i64)]) -> ClusterScheduler {
        ClusterScheduler::new(
            nodes
                .iter()
                .map(|(a, b, c, d)| (a.to_string(), *b, *c, *d))
                .collect(),
        )
    }

    #[test]
    fn balances_gpu_bundles_across_nodes() {
        // 2 nodes, 2 GPUs each; 4 one-GPU bundles -> 2 per node.
        let s = sched(&[("A", 0.0, 2.0, 0), ("B", 0.0, 2.0, 0)]);
        let placement = s.place_inner(vec![(0.0, 1.0, 0); 4]);
        let a = placement.iter().filter(|x| *x == "A").count();
        let b = placement.iter().filter(|x| *x == "B").count();
        assert_eq!((a, b), (2, 2));
        assert_eq!(s.overflow_count(), 0);
    }

    #[test]
    fn respects_capacity_then_overflows() {
        // 1 node, 1 GPU; two 1-GPU bundles -> second overflows onto the same node.
        let s = sched(&[("A", 0.0, 1.0, 0)]);
        let placement = s.place_inner(vec![(0.0, 1.0, 0), (0.0, 1.0, 0)]);
        assert_eq!(placement, vec!["A".to_string(), "A".to_string()]);
        assert_eq!(s.overflow_count(), 1);
    }

    #[test]
    fn memory_dimension_packs() {
        let s = sched(&[("A", 0.0, 0.0, 100), ("B", 0.0, 0.0, 100)]);
        // Two 60-byte bundles can't share a node -> one each.
        let placement = s.place_inner(vec![(0.0, 0.0, 60), (0.0, 0.0, 60)]);
        assert_ne!(placement[0], placement[1]);
    }

    #[test]
    fn reset_clears_load() {
        let s = sched(&[("A", 0.0, 1.0, 0)]);
        s.place_inner(vec![(0.0, 1.0, 0)]);
        s.reset();
        assert_eq!(s.overflow_count(), 0);
        assert!(s.place_inner(vec![(0.0, 1.0, 0)]) == vec!["A".to_string()]);
    }
}
