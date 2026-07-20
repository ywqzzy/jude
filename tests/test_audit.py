"""Durable audit log (redb): persistence, filtering, HTTP endpoints."""

from __future__ import annotations

import json
import os
import tempfile
import urllib.request

import pytest

from jude import observe


@pytest.fixture(autouse=True)
def _audit_dir(monkeypatch):
    d = tempfile.mkdtemp(prefix="jude_audit_test_")
    monkeypatch.setenv("JUDE_AUDIT_DIR", d)
    # force a fresh audit handle bound to this dir
    observe._AUDIT = None
    observe.set_audit_enabled(True)
    observe.audit_clear()
    yield d
    observe.audit_clear()
    observe._AUDIT = None


def test_query_persists_to_audit():
    with observe.query("q1", kind="local") as q:
        q.detail(sql="SELECT 1")
        q.stage("scan").done()
        q.done(rows=5, nbytes=40)
    recs = observe.audit_list()
    assert len(recs) == 1
    r = recs[0]
    assert r["label"] == "q1"
    assert r["kind"] == "local"
    assert r["status"] == "done"
    assert r["rows"] == 5
    assert r["stages"] == ["scan"]
    assert r["detail"]["sql"] == "SELECT 1"
    assert "audit_id" in r


def test_error_persisted():
    with pytest.raises(ValueError):
        with observe.query("boom", kind="local"):
            raise ValueError("nope")
    recs = observe.audit_list()
    assert recs[0]["status"] == "error"
    assert "nope" in recs[0]["error"]


def test_filter_and_stats():
    with observe.query("a", kind="local") as q:
        q.done(rows=1)
    with observe.query("b", kind="distributed") as q:
        q.done(rows=2)
    with pytest.raises(RuntimeError):
        with observe.query("c", kind="local"):
            raise RuntimeError("x")
    assert len(observe.audit_list(kind="local")) == 2
    assert len(observe.audit_list(kind="distributed")) == 1
    assert len(observe.audit_list(status="error")) == 1
    stats = observe.audit_stats()
    assert stats["total"] == 3
    assert stats["done"] == 2
    assert stats["error"] == 1
    assert stats["rows_total"] == 3


def test_get_by_id():
    with observe.query("findme", kind="local") as q:
        q.done(rows=7)
    recs = observe.audit_list()
    aid = recs[0]["audit_id"]
    got = observe.audit_get(aid)
    assert got["label"] == "findme"
    assert observe.audit_get(999999) is None


def test_newest_first_order():
    for i in range(5):
        with observe.query(f"q{i}", kind="local") as q:
            q.done(rows=i)
    recs = observe.audit_list()
    labels = [r["label"] for r in recs]
    assert labels == ["q4", "q3", "q2", "q1", "q0"]


def test_persists_across_reopen(_audit_dir):
    with observe.query("durable", kind="local") as q:
        q.done(rows=99)
    # drop the in-process handle, reopen from disk
    observe._AUDIT = None
    recs = observe.audit_list()
    assert len(recs) == 1
    assert recs[0]["label"] == "durable"
    assert recs[0]["rows"] == 99


def test_http_audit_endpoints():
    with observe.query("http audit", kind="local") as q:
        q.done(rows=3)
    url = observe.serve(port=8482, poll_ray=False)
    try:
        body = json.loads(urllib.request.urlopen(url + "/api/audit?limit=10", timeout=5).read())
        assert body["stats"]["total"] >= 1
        assert any(r["label"] == "http audit" for r in body["records"])
        aid = body["records"][0]["audit_id"]
        one = json.loads(urllib.request.urlopen(f"{url}/api/audit/{aid}", timeout=5).read())
        assert one["audit_id"] == aid
    finally:
        observe.stop_server()
