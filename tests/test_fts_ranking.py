"""Full-text search: BM25 RANKING (not just membership), distributed FTS merge,
RRF hybrid fusion, and the DuckDB hybrid-analytical bridge. (Audit blind spot:
FTS only had membership tests, no ranking; single-node hybrid had none.)"""

from __future__ import annotations

import tempfile

import numpy as np
import pyarrow as pa
import pytest

import jude

lance = pytest.importorskip("lance")


def _fts_dataset(docs, ids=None, vecs=None, dim=8):
    p = tempfile.mkdtemp() + "/ds"
    cols = {"id": pa.array(ids or list(range(len(docs))), type=pa.int64()),
            "text": pa.array(docs)}
    if vecs is not None:
        cols["v"] = pa.array(list(vecs), type=pa.list_(pa.float32(), dim))
    jude._lance.write(pa.table(cols), p, mode="create")
    jude._lance.create_fts_index(p, "text")
    return p


def test_fts_ranks_by_relevance():
    docs = [
        "the ocean is deep and the ocean is blue and full of ocean life",  # 3x "ocean"
        "a short note mentioning the ocean once here",                      # 1x "ocean"
        "completely unrelated text about mountains and forests",           # 0x "ocean"
    ]
    p = _fts_dataset(docs)
    out = jude._lance.full_text_search(p, "text", "ocean", k=3)
    ids = out.column("id").to_pylist()
    assert ids[0] == 0                      # most "ocean" mentions ranks first
    assert 2 not in ids[:2]                 # the unrelated doc isn't top
    assert "_score" in out.column_names     # BM25 score present
    scores = out.column("_score").to_pylist()
    assert scores == sorted(scores, reverse=True)  # descending BM25 order


def test_fts_no_match_returns_empty_or_excludes():
    p = _fts_dataset(["alpha beta gamma", "delta epsilon zeta"])
    out = jude._lance.full_text_search(p, "text", "nonexistentword", k=5)
    assert out.num_rows == 0


def test_distributed_fts_merges_shards():
    ray = pytest.importorskip("ray")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=4)
    from jude.runners.ray import RayRunner
    from jude import vector

    # two shards; the strongest match lives in shard 1
    p0 = _fts_dataset(["cats and dogs", "birds and fish"], ids=[0, 1])
    p1 = _fts_dataset(["dogs dogs dogs everywhere dogs", "trees and rocks"], ids=[2, 3])
    r = RayRunner(num_workers=2)
    out = vector.distributed_fts([p0, p1], "text", "dogs", k=3, columns=["id", "text"], runner=r)
    ids = out.column("id").to_pylist()
    assert ids[0] == 2                       # the "dogs"-heavy doc ranks first globally
    assert "_score" in out.column_names


def test_distributed_hybrid_rrf_fuses_text_and_vector():
    ray = pytest.importorskip("ray")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=4)
    from jude.runners.ray import RayRunner
    from jude import vector

    rng = np.random.default_rng(0)
    v = rng.standard_normal((4, 8)).astype("float32")
    docs = ["quantum computing breakthrough", "classical physics notes",
            "quantum entanglement study", "cooking recipes book"]
    p = _fts_dataset(docs, vecs=v)
    jude._lance.create_vector_index(p, "v", index_type="IVF_FLAT", num_partitions=2)
    r = RayRunner(num_workers=2)
    out = vector.distributed_hybrid([p], "text", "v", "quantum", v[0].tolist(),
                                    k=3, metric="cosine", runner=r)
    assert out.num_rows >= 1
    assert "id" in out.column_names


def test_hybrid_analytical_duckdb_bridge():
    from jude import retrieval

    rng = np.random.default_rng(1)
    v = rng.standard_normal((5, 8)).astype("float32")
    p = _fts_dataset(["red apple", "green apple", "blue sky", "red car", "green grass"],
                     vecs=v)
    jude._lance.create_vector_index(p, "v", index_type="IVF_FLAT", num_partitions=2)
    con = jude.connect()
    # vector search feeds a DuckDB SQL aggregation over the hits
    out = retrieval.hybrid_analytical(
        con, p, "SELECT COUNT(*) n FROM hits",
        vector_query=v[0].tolist(), vector_column="v", k=3, name="hits",
    )
    tbl = out if isinstance(out, pa.Table) else out.to_arrow()
    assert tbl.column("n")[0].as_py() >= 1
