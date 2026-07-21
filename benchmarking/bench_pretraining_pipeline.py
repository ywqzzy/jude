"""End-to-end pretraining data pipeline: raw text/HTML shards -> clean corpus ->
training-ready token shards, with a per-stage funnel + throughput report (X.4).

This is the harness that ties the roadmap together into one runnable job:

    WARC/HTML/text  ->  extract  ->  language filter  ->  quality filter
      ->  fuzzy dedup  ->  tokenize  ->  pack  ->  write shuffled token shards

Run it on a local sample to eyeball the funnel; point ``--source`` at s3:// and
scale the workers for the real thing. Real PB/100-node numbers need a real
cluster — this produces the numbers to fill X.4 in when one is available.

    python -m benchmarking.bench_pretraining_pipeline --docs 5000 --seq-len 512
"""

from __future__ import annotations

import argparse
import tempfile
import time

import numpy as np
import pyarrow as pa

from jude import curate
from jude import tokenize as tk


def _synth_corpus(n: int, seed: int = 0) -> pa.Table:
    """A synthetic web-ish corpus: prose + junk + duplicates + a non-English slice."""
    rng = np.random.default_rng(seed)
    vocab = ("the model reads data across many diverse documents and learns "
             "patterns from the cleaned high quality training corpus over time").split()
    docs = []
    for i in range(n):
        r = rng.random()
        if r < 0.15:                      # junk (short / symbols) -> quality drops it
            docs.append(rng.choice(["!!!", "aa aa", "<div>nav</div> home | about"]))
        elif r < 0.30:                    # duplicate of an earlier doc -> dedup drops it
            docs.append(" ".join(rng.choice(vocab, size=60)))
            docs.append(docs[-1])
        elif r < 0.38:                    # non-English -> language filter drops it
            docs.append("这是一段中文文本 " * 12)
        else:                             # good English prose
            docs.append(" ".join(rng.choice(vocab, size=int(rng.integers(60, 120)))))
    return pa.table({"text": docs[:n]})


def run(docs: int = 5000, seq_len: int = 512, tokenizer: str = "bytes") -> dict:
    t = _synth_corpus(docs)
    funnel = []
    stages: list = [
        ("language_filter", lambda x: curate.language_filter(x, keep="en", min_confidence=0.5)),
        ("quality_filter", lambda x: curate.quality_filter(x, min_words=40)),
        ("exact_dedup", lambda x: curate.exact_dedup(x)),
        ("fuzzy_dedup", lambda x: curate.fuzzy_dedup(x, threshold=0.7)),
    ]
    cur = t
    t0 = time.perf_counter()
    for name, fn in stages:
        rin = cur.num_rows
        s0 = time.perf_counter()
        cur = fn(cur)
        funnel.append({"stage": name, "rows_in": rin, "rows_out": cur.num_rows,
                       "dropped": rin - cur.num_rows, "sec": round(time.perf_counter() - s0, 3)})
    # tokenize -> pack -> shuffled token shards
    s0 = time.perf_counter()
    toks = tk.tokenize(cur, tokenizer=tokenizer)
    packed = tk.pack_sequences(toks, seq_len=seq_len, drop_remainder=True)
    out_dir = tempfile.mkdtemp() + "/shards"
    metas = tk.write_shuffled_shards(packed, out_dir, n_shards=4, fmt="bin", seed=0)
    total_tokens = sum(m["total_tokens"] for m in metas)
    funnel.append({"stage": "tokenize+pack+shard", "rows_in": cur.num_rows,
                   "sequences": packed.num_rows, "tokens": total_tokens,
                   "sec": round(time.perf_counter() - s0, 3)})
    elapsed = time.perf_counter() - t0
    return {
        "input_docs": docs,
        "kept_docs": cur.num_rows,
        "keep_rate": round(cur.num_rows / docs, 3) if docs else 0.0,
        "packed_sequences": packed.num_rows,
        "total_tokens": total_tokens,
        "seconds": round(elapsed, 3),
        "docs_per_sec": round(docs / elapsed, 1) if elapsed else 0.0,
        "funnel": funnel,
        "shards": len(metas),
    }


def _print(report: dict) -> None:
    print(f"\n=== jude pretraining pipeline: {report['input_docs']} docs "
          f"-> {report['kept_docs']} kept ({report['keep_rate']:.0%}) "
          f"-> {report['packed_sequences']} sequences / {report['total_tokens']} tokens ===")
    print(f"{report['seconds']}s  ({report['docs_per_sec']} docs/s), {report['shards']} shards\n")
    print(f"{'stage':<24}{'in':>10}{'out/seq':>10}{'dropped':>10}{'sec':>8}")
    for f in report["funnel"]:
        out = f.get("rows_out", f.get("sequences", "-"))
        dropped = f.get("dropped", "-")
        print(f"{f['stage']:<24}{f['rows_in']:>10}{out:>10}{dropped:>10}{f['sec']:>8}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=5000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--tokenizer", default="bytes")
    args = ap.parse_args()
    _print(run(docs=args.docs, seq_len=args.seq_len, tokenizer=args.tokenizer))
