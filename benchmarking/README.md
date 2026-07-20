# Benchmarking

Benchmarks for jude, mirroring the shape of Vane's
`multimodal_inference_benchmarks/`.

## `bench_multimodal_inference.py`

An image-classification-shaped pipeline: a stream of images is decoded and run
through a CPU-bound stand-in for a model that emits an embedding tensor. It
compares jude's execution backends on end-to-end throughput to isolate the
**scheduling / GIL** dimension — where jude's Rust orchestration and
out-of-process workers win.

```bash
python benchmarking/bench_multimodal_inference.py --images 800 --dim 64 --work 40000 --workers 8
```

Representative result on a 14-core M-series laptop (800 images, `work=40000`
iterations/image):

| backend | throughput | speedup |
|---|---|---|
| in-process (GIL-bound) | ~161 img/s | 1.00x |
| subprocess pool (Rust, GIL-free) | ~700 img/s | **~4.4x** |
| ray_actor (distributed) | ~400 img/s | ~2.5x |

Takeaways:

- **The GIL is the bottleneck for in-process UDFs.** All per-image Python work
  contends for one interpreter lock; a 14-core machine runs one core's worth.
- **Out-of-process workers bypass it.** N worker processes = N interpreters in
  true parallel. The Rust subprocess pool dispatches with the GIL released
  (`Python::detach`) and reuses a warm pool, so it wins on a single node.
- **ray_actor** carries higher per-batch overhead (object store + actor RPC) on
  one node, but is the path that scales to a multi-node cluster.

The per-item work here is a deterministic CPU load (no GPU/network), so the
benchmark runs anywhere and measures the orchestration overhead that separates
a Rust control plane from a Python one. For workloads dominated by GPU model
time, all backends converge on GPU throughput; jude's edge shows in the
scheduling-heavy, high-concurrency, mixed CPU/IO/GPU regime.
