"""jude.profile — cheap, scale-friendly corpus profiling.

Before/after a curation run you want to know: how many docs, how many DISTINCT
docs (dup rate), the length distribution, the language mix. At PB scale you can't
afford exact distinct counts, so cardinality uses **HyperLogLog** (fixed memory,
~1% error) over content hashes; lengths are summarized as percentiles. One pass.

    stats = profile(table, column="text", langs=True)
    # {'rows', 'approx_distinct', 'dup_rate', 'chars':{p50,p90,p99,mean},
    #  'words':{...}, 'langs':{'en':.., 'zh':..}}
"""

from __future__ import annotations

import hashlib
from typing import Any

import pyarrow as pa


class HyperLogLog:
    """Compact HLL for approximate distinct counts (p=14 -> 16384 registers,
    ~0.8% std error, ~16 KB). Deterministic (blake2b hash)."""

    def __init__(self, p: int = 14):
        self.p = p
        self.m = 1 << p
        self.registers = bytearray(self.m)

    def add(self, item: Any) -> None:
        h = int.from_bytes(hashlib.blake2b(str(item).encode("utf-8", "ignore"),
                                           digest_size=8).digest(), "big")
        idx = h >> (64 - self.p)                       # first p bits -> register
        rest = (h << self.p) & ((1 << 64) - 1)         # remaining bits
        rank = 1
        while rest and not (rest >> 63):               # count leading zeros + 1
            rank += 1
            rest <<= 1
        if rank > self.registers[idx]:
            self.registers[idx] = rank

    def estimate(self) -> int:
        import math

        m = self.m
        alpha = 0.7213 / (1 + 1.079 / m)               # bias constant for large m
        raw = alpha * m * m / sum(2.0 ** -r for r in self.registers)
        zeros = self.registers.count(0)
        if raw <= 2.5 * m and zeros:                    # small-range linear counting
            raw = m * math.log(m / zeros)
        return int(raw)


def _percentiles(vals: list[int]) -> dict:
    if not vals:
        return {"mean": 0.0, "p50": 0, "p90": 0, "p99": 0, "min": 0, "max": 0}
    s = sorted(vals)
    n = len(s)

    def pct(q: float) -> int:
        return s[min(n - 1, int(q * n))]

    return {"mean": sum(s) / n, "p50": pct(0.50), "p90": pct(0.90),
            "p99": pct(0.99), "min": s[0], "max": s[-1]}


def profile(table: pa.Table, *, column: str = "text", langs: bool = False,
            normalize_for_dup: bool = True) -> dict:
    """One-pass corpus profile: row count, approximate distinct docs (HLL over
    content — dup_rate = 1 - distinct/rows), char/word length percentiles, and
    (if ``langs``) a language histogram via the heuristic detector. Cardinality
    is approximate by design so it stays O(1) memory at PB scale."""
    texts = table.column(column).to_pylist()
    n = len(texts)
    hll = HyperLogLog()
    char_lens: list[int] = []
    word_lens: list[int] = []
    for t in texts:
        s = t or ""
        key = " ".join(s.lower().split()) if normalize_for_dup else s
        hll.add(key)
        char_lens.append(len(s))
        word_lens.append(len(s.split()))
    distinct = min(hll.estimate(), n) if n else 0
    out: dict = {
        "rows": n,
        "approx_distinct": distinct,
        "dup_rate": round(1.0 - distinct / n, 4) if n else 0.0,
        "chars": _percentiles(char_lens),
        "words": _percentiles(word_lens),
    }
    if langs:
        from jude.jude import _curate

        hist: dict = {}
        for lang, _conf in _curate.detect_language_batch(texts):
            hist[lang] = hist.get(lang, 0) + 1
        out["langs"] = dict(sorted(hist.items(), key=lambda kv: -kv[1]))
    return out
