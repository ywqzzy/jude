"""jude.curate_flow — a curation pipeline with a governance funnel.

Chains curation ops and records, for each stage, how many rows entered and how
many survived — the **data funnel** every corpus-curation workflow needs to
answer "how did my dataset shrink, and which filter dropped what". The funnel is
persisted to jude.observe (redb audit + dashboard), so a curation run is
inspectable after the fact like any other execution.

    from jude import curate_flow as cf
    flow = (cf.CurationFlow(raw_table)
            .quality_filter(min_words=50)
            .exact_dedup()
            .fuzzy_dedup(threshold=0.7)
            .redact_pii())
    result = flow.run()          # records a 'curation' query + a stage per op
    print(flow.funnel)           # [{op, rows_in, rows_out, dropped, pct_kept}, ...]
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from jude import curate as _c
from jude import curate_mm as _cm

__all__ = ["CurationFlow"]

# ops that DROP rows (funnel narrows) vs TRANSFORM (row count may change per op)
_KNOWN = {
    # text
    "quality_filter": _c.quality_filter,
    "quality_signals": _c.quality_signals,
    "exact_dedup": _c.exact_dedup,
    "fuzzy_dedup": _c.fuzzy_dedup,
    "semantic_dedup": _c.semantic_dedup,
    "chunk_text": _c.chunk_text,
    "add_content_hash": _c.add_content_hash,
    "detect_language": _c.detect_language,
    "language_filter": _c.language_filter,
    "redact_pii": _c.redact_pii,
    "detect_pii": _c.detect_pii,
    "global_shuffle": _c.global_shuffle,
    # image
    "image_dedup": _cm.image_dedup,
    "image_quality_filter": _cm.image_quality_filter,
    "add_image_quality": _cm.add_image_quality,
    "add_image_hash": _cm.add_image_hash,
}


class CurationFlow:
    """A chainable curation pipeline that records a per-stage row funnel."""

    def __init__(self, table: Any, *, label: str = "curation"):
        self._table = table if isinstance(table, pa.Table) else table.to_arrow()
        self._steps: list[tuple[str, dict]] = []
        self._label = label
        self.funnel: list[dict] = []

    def add(self, op: str, **kwargs: Any) -> "CurationFlow":
        if op not in _KNOWN:
            raise ValueError(f"unknown curation op {op!r}; known: {sorted(_KNOWN)}")
        self._steps.append((op, kwargs))
        return self

    # chainable sugar for the common ops
    def quality_filter(self, **kw): return self.add("quality_filter", **kw)
    def exact_dedup(self, **kw): return self.add("exact_dedup", **kw)
    def fuzzy_dedup(self, **kw): return self.add("fuzzy_dedup", **kw)
    def semantic_dedup(self, **kw): return self.add("semantic_dedup", **kw)
    def chunk_text(self, **kw): return self.add("chunk_text", **kw)
    def detect_language(self, **kw): return self.add("detect_language", **kw)
    def language_filter(self, **kw): return self.add("language_filter", **kw)
    def redact_pii(self, **kw): return self.add("redact_pii", **kw)
    def decontaminate(self, benchmark_texts, **kw):
        # decontaminate has a positional arg; store it in kwargs
        return self.add_decontaminate(benchmark_texts, **kw)

    def add_decontaminate(self, benchmark_texts, **kw) -> "CurationFlow":
        self._steps.append(("__decontaminate__", {"benchmark_texts": benchmark_texts, **kw}))
        return self

    def image_dedup(self, **kw): return self.add("image_dedup", **kw)
    def image_quality_filter(self, **kw): return self.add("image_quality_filter", **kw)

    def run(self) -> pa.Table:
        """Execute the flow, recording the funnel to jude.observe."""
        from jude import observe

        tbl = self._table
        self.funnel = []
        with observe.query(self._label, kind="pipeline") as q:
            q.detail(input_rows=tbl.num_rows, steps=[s[0] for s in self._steps])
            for op, kwargs in self._steps:
                rows_in = tbl.num_rows
                st = q.stage(op)
                if op == "__decontaminate__":
                    bench = kwargs.pop("benchmark_texts")
                    tbl = _c.decontaminate(tbl, bench, **kwargs)
                    op_name = "decontaminate"
                else:
                    tbl = _KNOWN[op](tbl, **kwargs)
                    op_name = op
                rows_out = tbl.num_rows
                st.progress(rows=rows_out)
                st.done()
                self.funnel.append({
                    "op": op_name,
                    "rows_in": rows_in,
                    "rows_out": rows_out,
                    "dropped": rows_in - rows_out,
                    "pct_kept": round(100.0 * rows_out / rows_in, 2) if rows_in else 100.0,
                })
            q.detail(funnel=self.funnel, output_rows=tbl.num_rows)
            q.done(rows=tbl.num_rows, nbytes=tbl.nbytes)
        return tbl
