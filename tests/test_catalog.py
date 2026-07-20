"""jude.catalog — storage governance: auto-discovery across formats, rich
metadata, queryable information-schema, column stats, drift validation."""
import os
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import jude


def _warehouse():
    wh = tempfile.mkdtemp()
    c = jude.connect()
    pytest.importorskip("lance")
    c.sql("SELECT range AS id, range*2 AS v FROM range(50)").write_lance(os.path.join(wh, "docs.lance"), "create")
    for dt in ["2024-01", "2024-02"]:
        d = os.path.join(wh, "events", f"dt={dt}")
        os.makedirs(d)
        pq.write_table(pa.table({"e": [1, 2]}), os.path.join(d, "p.parquet"))
    pq.write_table(pa.table({"a": [1, 2, 3]}), os.path.join(wh, "loose.parquet"))
    return wh


@pytest.fixture
def cat():
    return jude.catalog.Catalog(store=os.path.join(tempfile.mkdtemp(), "c.json"))


def test_discovery_across_formats(cat):
    wh = _warehouse()
    found = {f["name"]: f["format"] for f in cat.discover(wh)}
    assert found.get("docs.lance") == "lance"
    assert found.get("events") == "hive"
    assert found.get("loose") == "parquet"


def test_rich_describe(cat):
    wh = _warehouse()
    cat.discover(wh)
    d = cat.describe("docs.lance")
    assert d["num_rows"] == 50 and d["num_columns"] == 2
    assert d["size_bytes"] > 0 and d["num_files"] >= 1
    assert d["num_versions"] == 1
    assert "updated_at" in d


def test_hive_partition_columns_detected(cat):
    wh = _warehouse()
    cat.discover(wh)
    assert cat.get("events").get("partition_columns") == ["dt"]


def test_information_schema_is_queryable(cat):
    wh = _warehouse()
    cat.discover(wh)
    isch = cat.information_schema()
    assert set(isch.columns) >= {"name", "format", "num_rows", "size_bytes"}
    big = isch.filter("num_rows > 10").fetchall()
    assert any(row[0] == "docs.lance" for row in big)


def test_column_stats(cat):
    wh = _warehouse()
    cat.discover(wh)
    assert cat.column_stats("docs.lance").aggregate("count(*)").fetchone()[0] >= 1


def test_validate_detects_drift(cat):
    wh = _warehouse()
    cat.discover(wh)
    assert cat.validate() == []
    import shutil
    shutil.rmtree(os.path.join(wh, "docs.lance"))
    drift = cat.validate()
    assert any(m["name"] == "docs.lance" for m in drift)
