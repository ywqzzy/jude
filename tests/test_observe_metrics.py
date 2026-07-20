"""Observability: derived summary rollups, Prometheus exposition, and the
curation data-quality recorder. Pure-Python (no Ray/Rust rebuild needed)."""

from __future__ import annotations

import jude
from jude import observe


def test_summary_rollups_and_percentiles():
    observe.reset()
    with observe.query("q1", kind="distributed") as q:
        st = q.stage("shuffle", tasks_total=4)
        st.progress(tasks_done=4, rows=1000)
        st.done()
        q.done(rows=1000, nbytes=4096)
    with observe.query("q2", kind="local") as q:
        q.done(rows=500)

    s = observe.summary()
    assert s["queries"]["total"] == 2
    assert s["queries"]["by_status"].get("done") == 2
    assert s["queries"]["by_kind"].get("distributed") == 1
    # percentiles present and ordered
    lat = s["queries"]["latency_ms"]
    assert lat["p50"] <= lat["p95"] <= lat["p99"] <= lat["max"] + 1e-9
    assert s["stages"]["tasks_done"] == 4


def test_curate_recorder_tracks_rows_removed():
    observe.reset()
    with observe.curate("fuzzy_dedup", rows_in=1000) as c:
        c.done(rows_out=720)
    with observe.curate("quality_filter", rows_in=720) as c:
        c.done(rows_out=610)
    cur = observe.summary()["curation"]
    assert cur["ops"] == 2
    assert cur["rows_in"] == 1720
    assert cur["rows_out"] == 1330
    assert cur["removed"] == 390
    assert 0.0 < cur["keep_rate"] < 1.0


def test_curate_records_error_on_exception():
    observe.reset()
    try:
        with observe.curate("boom", rows_in=10):
            raise ValueError("nope")
    except ValueError:
        pass
    snap = observe.snapshot()
    q = [x for x in snap["queries"] if x["label"] == "boom"]
    assert q and q[0]["status"] == "error"


def test_prometheus_text_format():
    observe.reset()
    with observe.query("q", kind="local") as q:
        q.done(rows=42)
    with observe.curate("dedup", rows_in=100) as c:
        c.done(rows_out=80)
    text = observe.prometheus_text()
    # valid-ish exposition: HELP/TYPE headers + sample lines, ends with newline
    assert "# HELP jude_queries_total" in text
    assert "# TYPE jude_queries_total gauge" in text
    assert 'jude_curation_rows{stage="removed"} 20' in text
    assert "jude_curation_keep_rate 0.8" in text
    assert text.endswith("\n")
    # every non-comment line is "name value" or "name{labels} value"
    for line in text.splitlines():
        if line and not line.startswith("#"):
            assert len(line.rsplit(" ", 1)) == 2
