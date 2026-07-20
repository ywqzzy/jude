"""jude <-> Daft bridge (zero-copy via Arrow): run Daft's multimodal / embedding
ops on jude data and bring results back. Gated on daft."""
import pytest

pytest.importorskip("daft")
import daft

import jude


def test_to_daft_and_back():
    c = jude.connect()
    rel = c.sql("SELECT range AS x FROM range(5)")
    df = rel.to_daft()
    assert df.count_rows() == 5
    back = c.from_daft(df)
    assert sorted(r[0] for r in back.fetchall()) == [0, 1, 2, 3, 4]


def test_daft_transform_roundtrip():
    c = jude.connect()
    rel = c.sql("SELECT range AS x FROM range(5)")
    out = rel.daft_transform(lambda df: df.with_column("sq", daft.col("x") * daft.col("x")))
    assert out.order("x").fetchall() == [(0, 0), (1, 1), (2, 4), (3, 9), (4, 16)]


def test_from_daft_pydict():
    c = jude.connect()
    back = c.from_daft(daft.from_pydict({"a": [10, 20], "b": ["p", "q"]}))
    assert back.fetchall() == [(10, "p"), (20, "q")]
