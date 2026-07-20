#!/usr/bin/env python3
"""Multi-machine benchmark for jude, on a *simulated* multi-node Ray cluster.

Docker isn't available in this environment, so we use Ray's own multi-node test
harness (`ray.cluster_utils.Cluster`): it starts a head node plus N worker nodes
as separate raylets with their own object stores — genuinely multi-node (node-aware
placement + cross-node object transfer), the same mechanism Ray uses to test
distributed behavior. jude's RayRunner connects to it unchanged.

We run the CPU-bound multimodal map over the cluster and report throughput plus
how many distinct nodes actually executed work (proving it spread across
"machines"), and scaling from 1 -> N worker nodes.

    python benchmarking/bench_multinode.py --nodes 3 --cpus-per-node 4 --images 1500 --work 60000
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pyarrow as pa


class Infer:
    def __init__(self, dim: int, work: int):
        self.dim, self.work = dim, work

    def __call__(self, batch: "pa.Table") -> "pa.Table":
        import numpy as np

        out = []
        for blob in batch["img"].to_pylist():
            arr = np.frombuffer(blob, dtype="uint8").astype("float32")
            acc = 0.0
            for i in range(self.work):
                acc += (arr[i % arr.size] * i) % 7.0
            out.append(acc % 1000.0)
        return batch.append_column("e", pa.array(out, type=pa.float32()))


def make_images(n: int, size: int = 64) -> pa.Table:
    rng = np.random.default_rng(0)
    blobs = [rng.integers(0, 256, size, dtype="uint8").tobytes() for _ in range(n)]
    return pa.table({"id": list(range(n)), "img": pa.array(blobs, type=pa.binary())})


def _run_on_cluster(n_worker_nodes: int, cpus_per_node: int, images: pa.Table, work: int, batch_size: int) -> tuple[float, int, int]:
    from ray.cluster_utils import Cluster
    import ray

    cluster = Cluster(initialize_head=True, head_node_args={"num_cpus": 2})
    for _ in range(n_worker_nodes):
        cluster.add_node(num_cpus=cpus_per_node)
    ray.init(address=cluster.address, ignore_reinit_error=True, log_to_driver=False)
    try:
        import jude

        jude.runners._reset_runner()
        workers = n_worker_nodes * cpus_per_node  # one actor per worker-node CPU
        con = jude.connect()
        con.register("images", images)
        fn = Infer(64, work)
        # warm the resident actor pool (spread across nodes)
        con.sql("SELECT * FROM images LIMIT 1").map_batches(fn, execution_backend="ray_actor", num_workers=workers).num_rows
        rel = con.sql("SELECT * FROM images")
        t0 = time.perf_counter()
        out = rel.map_batches(fn, batch_size=batch_size, execution_backend="ray_actor", num_workers=workers)
        rows = out.num_rows
        t = time.perf_counter() - t0
        # how many distinct nodes ran a UDF actor? (inspect the resident pool)
        from jude.execution import _ACTOR_POOLS, shutdown_ray_pools

        node_ids: list[str] = []
        for ex in _ACTOR_POOLS.values():
            actors = getattr(ex, "_actors", [])
            if actors:
                node_ids += ray.get([a.node_id.remote() for a in actors])
        nodes_used = len(set(node_ids))
        try:
            shutdown_ray_pools()
        except Exception:
            pass
        return t, rows, nodes_used
    finally:
        ray.shutdown()
        cluster.shutdown()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=3, help="worker nodes (machines)")
    ap.add_argument("--cpus-per-node", type=int, default=4)
    ap.add_argument("--images", type=int, default=1500)
    ap.add_argument("--work", type=int, default=60000)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    images = make_images(args.images)
    print(f"\njude multi-node bench — up to {args.nodes} nodes x {args.cpus_per_node} CPU, "
          f"{args.images} images, work={args.work}/img")
    print("-" * 66)
    base = None
    for n in range(1, args.nodes + 1):
        t, rows, nodes_used = _run_on_cluster(n, args.cpus_per_node, images, args.work, args.batch_size)
        thr = rows / t
        if base is None:
            base = thr
        print(f"  {n} node(s) x{args.cpus_per_node}cpu: {t*1000:8.1f} ms  {thr:8.0f} img/s  "
              f"nodes_used={nodes_used}  scale={thr/base:4.2f}x")
    print("-" * 66)


if __name__ == "__main__":
    main()
