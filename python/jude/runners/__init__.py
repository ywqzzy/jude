"""jude.runners — execution backend selection and lifecycle.

Mirrors Vane's ``duckdb.runners`` surface: ``get_or_create_runner()`` returns a
runner whose ``run_iter`` / ``run_iter_tables`` / ``run_write`` execute a
relation. The runner type is chosen by the ``JUDE_RUNNER`` (or ``VANE_RUNNER``)
environment variable — ``"ray"`` (default) or ``"local"``.

Phase 1a ships the local runner (partition-parallel, in-process). The Ray
runner is registered in Phase 4; until then, selecting ``ray`` transparently
falls back to local so pipelines keep working.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Optional

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = [
    "MaterializedResult",
    "PartitionMetadata",
    "Runner",
    "LocalRunner",
    "get_or_create_runner",
    "get_or_infer_runner_type",
    "set_runner_local",
    "set_runner_ray",
]


@dataclass(frozen=True)
class PartitionMetadata:
    num_rows: int
    size_bytes: Optional[int] = None


class MaterializedResult:
    """Access to a single result partition."""

    def __init__(self, table: "pa.Table"):
        self._table = table

    def partition(self) -> "pa.Table":
        return self._table

    def metadata(self) -> PartitionMetadata:
        return PartitionMetadata(num_rows=self._table.num_rows, size_bytes=self._table.nbytes)

    def cancel(self) -> None:  # noqa: D401 - protocol method
        pass


class Runner:
    name: str

    def run_iter(self, relation: Any, results_buffer_size: int | None = None) -> Iterator[MaterializedResult]:
        raise NotImplementedError

    def run_iter_tables(self, relation: Any, results_buffer_size: int | None = None) -> Iterator["pa.Table"]:
        raise NotImplementedError

    def run_write(self, relation: Any) -> dict[str, Any]:
        raise NotImplementedError


class LocalRunner(Runner):
    """Local runner: materializes a relation and yields it partition-by-partition.

    Partition count follows the relation's ``num_partitions`` hint (from
    ``repartition``/``local_exchange``); the underlying DuckDB engine already
    parallelizes SQL execution across threads.
    """

    name = "local"

    def __init__(self, num_workers: int = 1):
        self.num_workers = num_workers

    def _partition(self, relation: Any) -> list["pa.Table"]:
        import pyarrow as pa

        table = relation.to_arrow()
        n = max(1, int(getattr(relation, "num_partitions", 1) or 1))
        if n <= 1 or table.num_rows == 0:
            return [table]
        # Even row-count split into n partitions.
        rows = table.num_rows
        step = (rows + n - 1) // n
        parts = []
        for start in range(0, rows, step):
            parts.append(table.slice(start, min(step, rows - start)))
        return parts

    def run_iter(self, relation: Any, results_buffer_size: int | None = None) -> Iterator[MaterializedResult]:
        for table in self.run_iter_tables(relation, results_buffer_size):
            yield MaterializedResult(table)

    def run_iter_tables(self, relation: Any, results_buffer_size: int | None = None) -> Iterator["pa.Table"]:
        import pyarrow as pa

        # With an explicit repartition hint, honor the requested partition count
        # (row-slice the materialized table). Otherwise stream the result batch
        # by batch — bounded memory, and the consumer can start work on the first
        # batch before the last is produced.
        n = max(1, int(getattr(relation, "num_partitions", 1) or 1))
        if n > 1:
            for part in self._partition(relation):
                yield part
            return
        try:
            stream = relation.record_batch_stream()
        except Exception:
            # A plan that can't lower to SQL (e.g. a pending UDF) — materialize.
            for part in self._partition(relation):
                yield part
            return
        schema = stream.schema
        emitted = False
        for batch in stream:
            emitted = True
            yield pa.Table.from_batches([batch], schema=schema)
        if not emitted:
            yield schema.empty_table()

    def run_write(self, relation: Any) -> dict[str, Any]:
        # Writes are expressed as COPY on the relation (to_parquet/to_csv).
        raise NotImplementedError("run_write: express writes via Relation.to_parquet/to_csv")


# ---------------------------------------------------------------------------
# Runner selection / lifecycle
# ---------------------------------------------------------------------------

_RUNNER: Runner | None = None
_FORCED_TYPE: str | None = None


def _resolve_runner_type() -> str:
    if _FORCED_TYPE is not None:
        return _FORCED_TYPE
    raw = os.environ.get("JUDE_RUNNER") or os.environ.get("VANE_RUNNER") or "ray"
    raw = raw.strip().lower()
    if raw in ("local", "local-fast"):
        return "local"
    return "ray"


def get_or_infer_runner_type() -> str:
    return _resolve_runner_type()


def set_runner_local(num_workers: int = 1, max_running_tasks: int | None = None) -> None:
    global _RUNNER, _FORCED_TYPE
    _FORCED_TYPE = "local"
    _RUNNER = LocalRunner(num_workers=num_workers)


def set_runner_ray(**kwargs: Any) -> None:
    global _RUNNER, _FORCED_TYPE
    _FORCED_TYPE = "ray"
    _RUNNER = None  # created lazily


def get_or_create_runner() -> Runner:
    global _RUNNER
    if _RUNNER is not None:
        return _RUNNER
    rtype = _resolve_runner_type()
    if rtype == "ray":
        try:
            from jude.runners.ray import RayRunner  # noqa: WPS433

            _RUNNER = RayRunner()
            return _RUNNER
        except Exception:
            # Ray unavailable / not yet implemented: fall back to local so
            # pipelines still run (single-node).
            pass
    _RUNNER = LocalRunner()
    return _RUNNER


def _reset_runner() -> None:
    """Test helper: drop the cached runner."""
    global _RUNNER, _FORCED_TYPE
    _RUNNER = None
    _FORCED_TYPE = None
