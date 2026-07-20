"""Curation flow with governance funnel + observe recording."""

from __future__ import annotations

import tempfile

import pyarrow as pa
import pytest

from jude import curate_flow as cf
from jude import observe


@pytest.fixture(autouse=True)
def _audit(monkeypatch):
    monkeypatch.setenv("JUDE_AUDIT_DIR", tempfile.mkdtemp(prefix="jude_flow_"))
    observe._AUDIT = None
    observe.audit_clear()
    yield


def _docs():
    good = ("The history of natural language processing began in the nineteen fifties when "
            "researchers first explored automated translation across many human languages "
            "using early digital computers and hand written grammar rules for parsing text.")
    return [good, good, "too short", "!@#$ " * 40, good + " and more distinct content here now"]


def test_funnel_records_per_stage():
    t = pa.table({"text": _docs()})
    flow = cf.CurationFlow(t).quality_filter(min_words=20).exact_dedup()
    out = flow.run()
    assert len(flow.funnel) == 2
    assert flow.funnel[0]["op"] == "quality_filter"
    assert flow.funnel[0]["rows_in"] == 5
    # quality drops the short + symbol-spam rows
    assert flow.funnel[0]["rows_out"] < 5
    # exact_dedup then drops the duplicate 'good' copies
    assert flow.funnel[1]["op"] == "exact_dedup"
    assert flow.funnel[1]["rows_out"] <= flow.funnel[0]["rows_out"]
    assert out.num_rows == flow.funnel[-1]["rows_out"]
    assert all(0 <= s["pct_kept"] <= 100 for s in flow.funnel)


def test_funnel_persisted_to_audit():
    t = pa.table({"text": _docs()})
    cf.CurationFlow(t, label="my curation").quality_filter(min_words=20).exact_dedup().run()
    recs = observe.audit_list(kind="pipeline")
    assert any(r["label"] == "my curation" for r in recs)
    rec = next(r for r in recs if r["label"] == "my curation")
    # detail carries the funnel + input/output row counts
    assert "funnel" in rec["detail"]
    assert rec["detail"]["input_rows"] == 5
    assert rec["detail"]["output_rows"] == rec["rows"]
    assert [s["op"] for s in rec["detail"]["funnel"]] == ["quality_filter", "exact_dedup"]


def test_decontaminate_in_flow():
    bench = ["the mitochondria is the powerhouse of the cell in biology class"]
    t = pa.table({"text": [
        "recall the mitochondria is the powerhouse of the cell in biology class today",
        "rust ownership prevents data races at compile time without a garbage collector",
    ]})
    out = cf.CurationFlow(t).add_decontaminate(bench, ngram=4, threshold=0.1).run()
    assert out.num_rows == 1  # contaminated doc dropped


def test_unknown_op_raises():
    t = pa.table({"text": ["x"]})
    with pytest.raises(ValueError):
        cf.CurationFlow(t).add("nonexistent_op")
