"""Iceberg storage — reading Iceberg tables into jude relations (storage P1).

Backed by DuckDB's iceberg extension (iceberg_scan / iceberg_snapshots). Tables
are created with pyiceberg (guarded by importorskip) so the roundtrip is real.
"""

import tempfile

import pytest

import jude

pa = pytest.importorskip("pyarrow")
pytest.importorskip("pyiceberg")


def _make_table(rows):
    """Create a small local Iceberg table, return its metadata location."""
    from pyiceberg.catalog.sql import SqlCatalog

    wh = tempfile.mkdtemp()
    cat = SqlCatalog("t", uri=f"sqlite:///{wh}/cat.db", warehouse=f"file://{wh}")
    cat.create_namespace("db")
    data = pa.table(rows)
    tbl = cat.create_table("db.t", schema=data.schema)
    tbl.append(data)
    return tbl


class TestIcebergRead:
    def test_read_iceberg_roundtrip(self):
        tbl = _make_table({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        rel = jude.read_iceberg(tbl.metadata_location)
        assert rel.columns == ["id", "name"]
        assert sorted(rel.fetchall()) == [(1, "a"), (2, "b"), (3, "c")]

    def test_read_iceberg_composes_with_sql(self):
        tbl = _make_table({"id": [1, 2, 3, 4], "g": ["x", "x", "y", "y"]})
        con = jude.connect()
        rel = con.read_iceberg(tbl.metadata_location)
        # an Iceberg scan is an ordinary relation: filter/aggregate compose
        got = rel.filter("id > 1").aggregate("g, count(*)").order("g").fetchall()
        assert got == [("x", 1), ("y", 2)]

    def test_iceberg_snapshots(self):
        tbl = _make_table({"id": [1, 2]})
        con = jude.connect()
        snaps = con.iceberg_snapshots(tbl.metadata_location)
        # at least one snapshot from the append
        assert snaps.num_rows >= 1


class TestIcebergWrite:
    """jude as a write engine: data files written by Rust/DuckDB COPY TO parquet
    (partitioned), commit via the thin pyiceberg shim. Roundtrip write -> read."""

    def test_write_read_roundtrip(self):
        wh = tempfile.mkdtemp()
        con = jude.connect()
        # >1 partition to exercise the partitioned Rust write path
        rel = con.sql("SELECT range AS id, ('v' || range::VARCHAR) AS name FROM range(2500)")
        meta = rel.write_iceberg(wh, "db.t", mode="append")
        back = jude.read_iceberg(meta)
        assert back.columns == ["id", "name"]
        rows = back.fetchall()
        assert len(rows) == 2500
        assert back.aggregate("sum(id)").fetchone()[0] == 2499 * 2500 // 2

    def test_write_append_accumulates(self):
        wh = tempfile.mkdtemp()
        con = jude.connect()
        con.sql("SELECT range AS id FROM range(3)").write_iceberg(wh, "db.acc", mode="append")
        meta = con.sql("SELECT range + 100 AS id FROM range(3)").write_iceberg(wh, "db.acc", mode="append")
        assert len(jude.read_iceberg(meta).fetchall()) == 6

    def test_write_overwrite_replaces(self):
        wh = tempfile.mkdtemp()
        con = jude.connect()
        con.sql("SELECT range AS id FROM range(100)").write_iceberg(wh, "db.ow", mode="append")
        meta = con.sql("SELECT range AS id FROM range(3)").write_iceberg(wh, "db.ow", mode="overwrite")
        assert len(jude.read_iceberg(meta).fetchall()) == 3

