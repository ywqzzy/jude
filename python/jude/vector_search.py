"""jude.vector_search — encapsulated vector search (exact / ANN, single /
distributed) with one clean API and tunable parameters.

Wraps the primitives in jude.vector into a reusable ``VectorSearch`` object so
you configure the dataset + strategy once and issue many queries. Chooses the
right strategy (exact brute-force vs two-stage ANN) and scales exact search
across the cluster via the distributed map-reduce top-k.

    # in-memory table, exact distributed
    vs = VectorSearch(table, column="v", metric="cosine", distributed=True)
    hits = vs.search(query, k=1000)                 # 100% recall
    batch = vs.search_batch([q1, q2, q3], k=100)    # many queries

    # a large Lance dataset, two-stage ANN (build index first)
    vs = VectorSearch("/data/emb.lance", column="v", strategy="ann",
                      overfetch=5, nprobes=100)
    vs.build_index(index_type="IVF_FLAT", num_partitions=1000)
    hits = vs.search(query, k=1000)

Distributed exact KNN principle: partition -> each worker computes its shard's
local top-k (DuckDB native array distance) -> driver merges the local top-ks
and takes the global top-k. This is EXACT (a global top-k row is necessarily in
its own shard's top-k), so recall is 100%; it just spreads the O(N·d) scan
across workers.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from jude import vector as _v

__all__ = ["VectorSearch"]


class VectorSearch:
    """Configured vector search over one dataset.

    Parameters — and how to tune them:

    - ``data``: a pyarrow Table / jude Relation (in-memory) OR a Lance dataset
      path (str, required for ``strategy='ann'``).
    - ``column``: the embedding column (``FLOAT[dim]`` or list<float>).
    - ``metric``: ``cosine`` (default) | ``l2`` | ``l2sq`` | ``ip``. Match how the
      embeddings were trained (most sentence/image encoders → cosine).
    - ``strategy``:
        * ``'exact'`` (default) — brute force. 100% recall. Best for ≤ a few
          million vectors and/or large k (e.g. 100k). No index needed.
        * ``'ann'`` — two-stage: ANN over-fetch + exact re-rank over a Lance
          index. For datasets too big to brute-force. Needs ``build_index`` first.
    - ``distributed``: for ``exact``, run the map-reduce top-k across the Ray
      cluster (each worker scans its shard). Turn on when one machine can't hold
      / scan the data fast enough. Result is identical to single-node exact.
    - ``overfetch`` (ann): fetch ``k*overfetch`` candidates before exact re-rank.
      Higher → higher recall, slower. Start 5, raise if recall < target.
    - ``nprobes`` (ann): IVF cells to scan. The recall knob. Start ≈√(num_partitions),
      raise toward ``num_partitions`` for higher recall (→ exact as it hits all).
    """

    def __init__(
        self,
        data: Any,
        *,
        column: str = "v",
        metric: str = "cosine",
        strategy: str = "exact",
        distributed: bool = False,
        overfetch: int = 5,
        nprobes: int | None = None,
        runner: Any = None,
    ):
        if metric not in _v.METRICS:
            raise ValueError(f"unknown metric {metric!r}; use one of {list(_v.METRICS)}")
        if strategy not in ("exact", "ann"):
            raise ValueError("strategy must be 'exact' or 'ann'")
        if strategy == "ann" and not isinstance(data, str):
            raise ValueError("strategy='ann' needs a Lance dataset path (str)")
        self.column = column
        self.metric = metric
        self.strategy = strategy
        self.distributed = distributed
        self.overfetch = overfetch
        self.nprobes = nprobes
        self._runner = runner
        self._path = data if isinstance(data, str) else None
        self._table = None if isinstance(data, str) else (
            data if isinstance(data, pa.Table) else data.to_arrow()
        )

    # --- index (ann strategy) ---

    def build_index(self, *, index_type: str = "IVF_FLAT", num_partitions: int | None = None,
                    num_sub_vectors: int | None = None) -> "VectorSearch":
        """Build a Lance vector index for ANN search. Use IVF_FLAT for high
        recall (no compression); IVF_PQ only if memory-bound (caps recall).
        ``num_partitions`` ≈ √(rows). Returns self for chaining."""
        import jude

        if self._path is None:
            raise ValueError("build_index requires a Lance dataset path")
        con = jude.connect()
        n = con.read_lance(self._path).count() if hasattr(con, "read_lance") else None
        nparts = num_partitions or (max(1, int((n or 1_000_000) ** 0.5)))
        kw: dict = {"index_type": index_type, "metric": self.metric, "num_partitions": nparts}
        if index_type == "IVF_PQ":
            kw["num_sub_vectors"] = num_sub_vectors or 16
        con.create_lance_vector_index(self._path, self.column, **kw)
        return self

    # --- search ---

    def search(self, query: list[float], k: int = 10) -> pa.Table:
        """Top-k nearest neighbours of ``query``. Returns rows + ``_distance``."""
        if self.strategy == "ann":
            return _v.knn_rerank(self._path, self.column, query, k=k, overfetch=self.overfetch,
                                 nprobes=self.nprobes, metric=self.metric)
        # exact
        if self.distributed:
            return _v.distributed_knn(self._table, self.column, query, k=k,
                                      metric=self.metric, runner=self._runner)
        import jude

        con = jude.connect()
        con.register("_vs", self._table)
        return _v.knn(con, "_vs", self.column, query, k=k, metric=self.metric)

    def search_batch(self, queries: list, k: int = 10) -> list:
        """Run several queries; returns a list of result tables (one per query)."""
        return [self.search(q, k=k) for q in queries]

    def recall_vs_exact(self, query: list[float], k: int = 10, id_column: str = "id") -> float:
        """Measure this config's recall@k against exact brute force for one query
        — use it to tune overfetch/nprobes until recall meets your target."""
        import jude

        con = jude.connect()
        tbl = self._table
        if tbl is None:
            tbl = con.read_lance(self._path).to_arrow()
        con.register("_vs_gt", tbl)
        exact_ids = _v.knn(con, "_vs_gt", self.column, query, k=k, metric=self.metric).column(id_column).to_pylist()
        got = self.search(query, k=k).column(id_column).to_pylist()
        return _v.recall_at_k(got, exact_ids, k=k)
