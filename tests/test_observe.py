"""Observability subsystem: Rust MetricsRegistry + Python facade + HTTP endpoint."""

from __future__ import annotations

import json
import urllib.request

import pytest

from jude import observe


@pytest.fixture(autouse=True)
def _reset():
    observe.reset()
    yield
    observe.reset()


def test_query_lifecycle_recorded():
    with observe.query("select 1", kind="local") as q:
        q.done(rows=1, nbytes=8)
    snap = observe.snapshot()
    assert snap["summary"]["queries_total"] == 1
    assert snap["summary"]["queries_running"] == 0
    (rec,) = snap["queries"]
    assert rec["label"] == "select 1"
    assert rec["status"] == "done"
    assert rec["rows"] == 1


def test_query_error_recorded():
    with pytest.raises(ValueError):
        with observe.query("boom", kind="local"):
            raise ValueError("nope")
    snap = observe.snapshot()
    (rec,) = snap["queries"]
    assert rec["status"] == "error"
    assert "nope" in rec["error"]


def test_stages_and_progress():
    with observe.query("dist agg", kind="distributed") as q:
        st = q.stage("partial", tasks_total=8)
        st.progress(tasks_done=4, rows=400, nbytes=1600)
        st.progress(tasks_done=4, rows=400, nbytes=1600, attempts=1)
        st.done()
        q.done(rows=800)
    snap = observe.snapshot()
    (stage,) = snap["stages"]
    assert stage["name"] == "partial"
    assert stage["tasks_done"] == 8
    assert stage["tasks_total"] == 8
    assert stage["rows"] == 800
    assert stage["attempts"] == 1
    assert stage["status"] == "done"


def test_pool_utilization_accumulates():
    observe.record_pool("p", "subprocess", 4, batches_in=2, rows=200, busy_ns=1000)
    observe.record_pool("p", "subprocess", 4, batches_in=3, rows=300, busy_ns=2000)
    snap = observe.snapshot()
    (pool,) = snap["pools"]
    assert pool["batches_in"] == 5
    assert pool["rows"] == 500
    assert pool["busy_ns"] == 3000
    observe.remove_pool("p")
    assert observe.snapshot()["pools"] == []


def test_running_query_kept_in_summary():
    with observe.query("long", kind="local") as q:
        snap = observe.snapshot()
        assert snap["summary"]["queries_running"] == 1
        q.done()
    assert observe.snapshot()["summary"]["queries_running"] == 0


def test_http_endpoint_serves_snapshot():
    with observe.query("http q", kind="local") as q:
        q.done(rows=3)
    url = observe.serve(port=8479, poll_ray=False)
    try:
        body = urllib.request.urlopen(url + "/api/metrics", timeout=5).read()
        data = json.loads(body)
        assert data["summary"]["queries_total"] == 1
        health = urllib.request.urlopen(url + "/api/health", timeout=5).read()
        assert json.loads(health)["ok"] is True
    finally:
        observe.stop_server()


def test_poll_cluster_nodes_never_errors():
    # poll_cluster_nodes must be safe whether or not Ray is initialized: it
    # returns a node count (>=0) and never raises. (Ray may already be up from
    # earlier tests in the same process, so we can't assert exactly 0.)
    n = observe.poll_cluster_nodes()
    assert isinstance(n, int)
    assert n >= 0
