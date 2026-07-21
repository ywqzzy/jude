"""jude.incremental_dedup — dedup NEW data against an EXISTING corpus.

Pretraining corpora grow dump by dump (new CommonCrawl snapshots). Re-running
global dedup over everything each time is wasteful; you want to dedup only the
new dump against what's already kept. This persists a content-hash index (a
Lance dataset of hashes, versioned) so an incremental run is O(new docs), not
O(whole corpus).

    idx = HashIndex("dedup_index.lance")   # loads prior hashes if present
    novel = idx.add(new_dump, column="text")   # rows whose content is NOT already seen
    idx.save()                                  # persist the grown index for next time

Exact (content-hash) incremental dedup. Fuzzy/MinHash incremental (band-key
index) is a documented follow-up.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from jude.jude import _curate


class HashIndex:
    """A persistent set of normalized content hashes for incremental exact dedup.
    Backed by a Lance dataset (column ``_hash``) so it survives across runs and is
    version-tracked. In memory it's a Python set for O(1) membership."""

    def __init__(self, path: str | None = None, *, normalize: bool = True):
        self.path = path
        self.normalize = normalize
        self._seen: set[str] = set()
        if path is not None:
            self.load()

    def load(self) -> "HashIndex":
        """Load hashes from ``path`` if the dataset exists (else start empty)."""
        from jude import _lance

        try:
            tbl = _lance.read_table(self.path, columns=["_hash"])
            self._seen = set(tbl.column("_hash").to_pylist())
        except (FileNotFoundError, ValueError, OSError):
            self._seen = set()
        return self

    def save(self) -> dict:
        """Persist the current hash set to ``path`` (overwrite snapshot)."""
        if self.path is None:
            raise ValueError("HashIndex has no path to save to")
        from jude import _lance

        tbl = pa.table({"_hash": pa.array(sorted(self._seen), type=pa.string())})
        _lance.write(tbl, self.path, mode="overwrite")
        return {"path": self.path, "hashes": len(self._seen)}

    def __len__(self) -> int:
        return len(self._seen)

    def add(self, table: pa.Table, *, column: str = "text", keep_hash: bool = False) -> pa.Table:
        """Return the rows of ``table`` whose content is NOT already in the index
        (and within this batch, first occurrence wins), then add them to the
        index. Call ``save()`` afterwards to persist. Idempotent: re-adding the
        same table returns zero rows the second time."""
        hashes = _curate.content_hash_batch(table.column(column).to_pylist(), self.normalize)
        keep: list[int] = []
        for i, h in enumerate(hashes):
            if h is None or h in self._seen:
                continue
            self._seen.add(h)
            keep.append(i)
        out = table.take(pa.array(keep, type=pa.int64()))
        if keep_hash:
            kh = [hashes[i] for i in keep]
            out = out.append_column("_hash", pa.array(kh, type=pa.string()))
        return out.combine_chunks()

    def contains(self, table: pa.Table, *, column: str = "text") -> list[bool]:
        """Per-row: is this document's content already in the index? (No mutation.)"""
        hashes = _curate.content_hash_batch(table.column(column).to_pylist(), self.normalize)
        return [h is not None and h in self._seen for h in hashes]


def incremental_dedup(
    new_table: pa.Table,
    index_path: str,
    *,
    column: str = "text",
    normalize: bool = True,
) -> pa.Table:
    """Convenience: load the index at ``index_path``, keep only ``new_table``'s
    novel documents (not seen in prior runs), persist the grown index, return the
    novel rows."""
    idx = HashIndex(index_path, normalize=normalize)
    novel = idx.add(new_table, column=column)
    idx.save()
    return novel
