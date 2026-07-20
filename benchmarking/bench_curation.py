#!/usr/bin/env python3
"""Large-scale curation benchmark — jude's LLM-data-processing workload.

Measures throughput (docs/s) of the curation operators jude is positioned for,
single-node vs distributed, at scale. This is the benchmark for jude's actual
differentiator (a distributed large-model data-processing engine), not a generic
SQL/UDF micro-bench.

Workloads (a synthetic web-like corpus with duplicates + near-dups + junk):
  - quality_filter   (C3, map-style, embarrassingly parallel)
  - exact_dedup      (C2, hash shuffle)
  - fuzzy_dedup      (C1, MinHash-LSH band shuffle) — the flagship, most compute

For each: single-node jude.curate vs distributed jude.curate_dist over N workers,
reporting docs/s and speedup.

    python benchmarking/bench_curation.py --docs 50000 --workers 8
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pyarrow as pa


def make_corpus(n: int, seed: int = 0) -> pa.Table:
    """Synthetic web-like corpus: ~60% unique, ~25% exact dups, ~15% near-dups,
    plus a sprinkling of junk (too-short / symbol-spam) — the shape a real
    dedup+quality pass faces."""
    rng = np.random.default_rng(seed)
    topics = [
        "the history of natural language processing spans several decades of research",
        "distributed systems rely on consensus protocols to stay consistent under failure",
        "photosynthesis converts sunlight into chemical energy stored in glucose molecules",
        "the industrial revolution transformed manufacturing through steam and machinery",
        "neural networks learn hierarchical representations from large labelled datasets",
        "ocean currents redistribute heat around the planet shaping regional climates",
        "the printing press accelerated the spread of knowledge across early modern europe",
        "quantum mechanics describes the behaviour of matter at atomic length scales",
    ]
    docs: list[str] = []
    for i in range(n):
        r = rng.random()
        base = topics[i % len(topics)]
        if r < 0.60:  # unique-ish: append distinct tail
            docs.append(f"{base} — entry {i} with some unique elaboration number {i * 7 % 9973}")
        elif r < 0.85:  # exact dup of a base topic
            docs.append(base)
        elif r < 0.97:  # near-dup: one word changed
            docs.append(base + (" indeed" if i % 2 else " truly"))
        elif r < 0.985:  # junk: too short
            docs.append("short")
        else:  # junk: symbol spam
            docs.append("!@#$ %^&* " * 6)
    return pa.table({"id": list(range(n)), "text": docs})


def _bench(fn, *, warmup: bool = False) -> tuple[float, int]:
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    return dt, out.num_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=50000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ops", nargs="+", default=["quality_filter", "exact_dedup", "fuzzy_dedup"])
    args = ap.parse_args()

    from jude import curate, curate_dist

    corpus = make_corpus(args.docs)
    print(f"\njude curation bench — {args.docs:,} docs, {args.workers} workers")
    print("-" * 68)

    runner = None
    if any(o in ("exact_dedup", "fuzzy_dedup") or True for o in args.ops):
        try:
            from _bench_ray import connect_ray

            connect_ray(num_cpus=args.workers)
            from jude.runners.ray import RayRunner

            runner = RayRunner(num_workers=args.workers)
        except Exception as e:  # pragma: no cover
            print(f"(ray unavailable: {e}; single-node only)")

    local_ops = {
        "quality_filter": lambda: curate.quality_filter(corpus, min_words=8),
        "exact_dedup": lambda: curate.exact_dedup(corpus),
        "fuzzy_dedup": lambda: curate.fuzzy_dedup(corpus, threshold=0.7, bands=16),
    }
    dist_ops = {
        "quality_filter": lambda: curate_dist.dist_quality_filter(corpus, runner=runner, min_words=8),
        "exact_dedup": lambda: curate_dist.dist_exact_dedup(corpus, runner=runner),
        "fuzzy_dedup": lambda: curate_dist.dist_fuzzy_dedup(corpus, runner=runner, threshold=0.7, bands=16),
    }

    print("  " + "op".ljust(16) + "single-node".rjust(16) + "distributed".rjust(16) + "speedup".rjust(10))
    print("  " + "-" * 58)
    for op in args.ops:
        ldt, lrows = _bench(local_ops[op])
        lthr = args.docs / ldt
        line = "  " + op.ljust(16) + f"{lthr:>11,.0f}/s" + " " * 4
        if runner is not None:
            ddt, drows = _bench(dist_ops[op])
            dthr = args.docs / ddt
            line = ("  " + op.ljust(16)
                    + f"{lthr:>11,.0f}/s".rjust(16)
                    + f"{dthr:>11,.0f}/s".rjust(16)
                    + f"{dthr / lthr:>8.2f}x".rjust(10))
        print(line)
    print("-" * 68)
    print("(dedup speedup depends on corpus size vs shuffle overhead; larger --docs favors distributed)")


if __name__ == "__main__":
    main()
