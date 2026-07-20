"""jude.catalog — storage governance across formats.

Not just a hand-maintained registry: it *discovers* what storage and tables live
underneath a warehouse root (Lance / Iceberg / Hive / Parquet), and carries rich
metadata — schema, row count, on-disk size, file count, partition columns,
created/updated times, and git-like version history — for everything it knows
about. The catalog itself is queryable as a jude relation (an INFORMATION_SCHEMA
you can run SQL over), and can validate that registered paths still exist.

    jude.catalog.discover("/warehouse")     # auto-find + register all tables
    jude.catalog.information_schema()        # -> jude relation you can SQL over
    jude.catalog.describe("db.docs")         # schema, size, files, partitions, versions
    jude.catalog.column_stats("db.docs")     # per-column min/max/nulls/distinct
    jude.catalog.validate()                  # drift: which registered paths vanished
"""

from __future__ import annotations

import glob as _glob
import json
import os
import threading
from typing import Any

_FORMATS = ("lance", "iceberg", "hive", "parquet", "csv")


def _default_store() -> str:
    return os.environ.get("JUDE_CATALOG") or os.path.join(os.path.expanduser("~"), ".jude", "catalog.json")


# --- format detection (on-disk layout sniffing) -----------------------------


def _is_lance(d: str) -> bool:
    return d.endswith(".lance") or (
        os.path.isdir(os.path.join(d, "_versions")) and os.path.isdir(os.path.join(d, "data"))
    )


def _is_iceberg(d: str) -> bool:
    meta = os.path.join(d, "metadata")
    if not os.path.isdir(meta):
        return False
    entries = os.listdir(meta)
    return any(e.endswith(".metadata.json") for e in entries) or "version-hint.text" in entries


def _hive_partition_keys(d: str) -> list[str]:
    """Ordered partition keys if `d` is the root of a key=value/ Hive layout."""
    keys: list[str] = []
    cur = d
    for _ in range(16):  # bounded descent
        subs = [s for s in os.listdir(cur) if os.path.isdir(os.path.join(cur, s)) and "=" in s]
        if not subs:
            break
        key = subs[0].split("=", 1)[0]
        keys.append(key)
        cur = os.path.join(cur, subs[0])
    return keys


def detect_format(path: str) -> str | None:
    """Best-effort format of a path: lance / iceberg / hive / parquet / csv / None."""
    if os.path.isdir(path):
        if _is_lance(path):
            return "lance"
        if _is_iceberg(path):
            return "iceberg"
        if _hive_partition_keys(path):
            return "hive"
        if _glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True):
            return "parquet"
        return None
    if path.endswith(".parquet"):
        return "parquet"
    if path.endswith((".csv", ".tsv")):
        return "csv"
    return None


def _dir_stats(paths: list[str]) -> dict:
    size = 0
    n = 0
    mtimes: list[float] = []
    for p in paths:
        try:
            st = os.stat(p)
            size += st.st_size
            n += 1
            mtimes.append(st.st_mtime)
        except OSError:
            pass
    out = {"size_bytes": size, "num_files": n}
    if mtimes:
        import datetime as _dt

        out["created_at"] = _dt.datetime.fromtimestamp(min(mtimes)).isoformat(timespec="seconds")
        out["updated_at"] = _dt.datetime.fromtimestamp(max(mtimes)).isoformat(timespec="seconds")
    return out


def _files_of(entry: dict) -> list[str]:
    path, fmt = entry["path"], entry["format"]
    if fmt in ("parquet", "csv") and not os.path.isdir(path):
        return [path]
    root = path
    if fmt == "hive":
        # `path` may be a glob; take its non-glob prefix.
        root = path.split("*", 1)[0].rstrip("/")
    if os.path.isdir(root):
        return [os.path.join(dp, f) for dp, _, fs in os.walk(root) for f in fs]
    return _glob.glob(path, recursive=True)


class Catalog:
    """A discovering, metadata-rich registry of datasets, persisted to JSON."""

    def __init__(self, store: str | None = None):
        self._store = store or _default_store()
        self._lock = threading.Lock()

    # --- persistence --------------------------------------------------------
    def _load(self) -> dict:
        try:
            with open(self._store, encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save(self, data: dict) -> None:
        os.makedirs(os.path.dirname(self._store) or ".", exist_ok=True)
        tmp = self._store + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp, self._store)

    # --- registration -------------------------------------------------------
    def register(self, name: str, path: str, format: str | None = None, **props: Any) -> dict:  # noqa: A002
        fmt = (format or detect_format(path) or "").lower()
        if fmt not in _FORMATS:
            raise ValueError(f"unknown/undetectable format for {name!r} at {path!r}; pass format= one of {_FORMATS}")
        entry = {"name": name, "path": path, "format": fmt, **props}
        if fmt == "hive":
            entry.setdefault("partition_columns", _hive_partition_keys(path.split("*", 1)[0].rstrip("/")))
        with self._lock:
            data = self._load()
            data[name] = entry
            self._save(data)
        return entry

    def drop(self, name: str) -> bool:
        with self._lock:
            data = self._load()
            existed = data.pop(name, None) is not None
            if existed:
                self._save(data)
        return existed

    unregister = drop

    # --- discovery ----------------------------------------------------------
    def discover(self, root: str, namespace: str = "", register: bool = True) -> list[dict]:
        """Walk `root`, detect tables by on-disk layout (Lance / Iceberg / Hive /
        Parquet), and (by default) register them under `namespace.<dirname>`.
        Prunes into detected table dirs so their internals aren't re-scanned."""
        found: list[dict] = []
        root = os.path.abspath(root)
        for dirpath, dirnames, filenames in os.walk(root):
            fmt = detect_format(dirpath)
            if fmt in ("lance", "iceberg", "hive"):
                base = os.path.relpath(dirpath, root).replace(os.sep, ".").strip(".")
                nm = f"{namespace}.{base}" if namespace else (base or os.path.basename(dirpath))
                path = os.path.join(dirpath, "**", "*.parquet") if fmt == "hive" else dirpath
                found.append({"name": nm, "path": path, "format": fmt})
                dirnames[:] = []  # don't descend into a detected table
                continue
            # Loose parquet files at this level (not inside a detected table).
            for f in filenames:
                if f.endswith(".parquet"):
                    fp = os.path.join(dirpath, f)
                    base = os.path.relpath(fp, root).replace(os.sep, ".").rsplit(".parquet", 1)[0]
                    nm = f"{namespace}.{base}" if namespace else base
                    found.append({"name": nm, "path": fp, "format": "parquet"})
        if register:
            for e in found:
                self.register(e["name"], e["path"], e["format"])
        return found

    # --- listing / inspection ----------------------------------------------
    def list(self) -> list[str]:
        return sorted(self._load().keys())

    def tables(self) -> list[dict]:
        d = self._load()
        return [d[n] for n in sorted(d)]

    def get(self, name: str) -> dict:
        data = self._load()
        if name not in data:
            raise KeyError(f"no table {name!r} in catalog")
        return data[name]

    def read(self, name: str):
        import jude

        e = self.get(name)
        fmt, path = e["format"], e["path"]
        if fmt == "lance":
            return jude.read_lance(path)
        if fmt == "iceberg":
            return jude.read_iceberg(path)
        if fmt == "hive":
            return jude.connect().read_hive(path)
        if fmt == "parquet":
            return jude.read_parquet(path)
        if fmt == "csv":
            return jude.read_csv(path)
        raise ValueError(f"cannot read format {fmt!r}")

    def versions(self, name: str) -> list[dict]:
        e = self.get(name)
        fmt, path = e["format"], e["path"]
        if fmt == "lance":
            from jude import _lance

            return _lance.list_versions(path).to_pylist()
        if fmt == "iceberg":
            import jude

            return jude.connect().iceberg_snapshots(path).to_arrow().to_pylist()
        return []

    def describe(self, name: str) -> dict:
        """Full metadata: format, path, columns+types, row count, on-disk size,
        file count, partition columns, created/updated times, version count."""
        e = self.get(name)
        out = dict(e)
        out.update(_dir_stats(_files_of(e)))
        try:
            rel = self.read(name)
            out["columns"] = list(rel.columns)
            out["types"] = list(rel.types)
            out["num_columns"] = len(rel.columns)
            out["num_rows"] = rel.aggregate("count(*) AS n").fetchone()[0]
        except Exception as ex:  # noqa: BLE001 - describe is best-effort
            out["error"] = str(ex)
        if e["format"] in ("lance", "iceberg"):
            try:
                out["num_versions"] = len(self.versions(name))
            except Exception:  # noqa: BLE001
                pass
        return out

    def column_stats(self, name: str):
        """Per-column statistics (min / max / null / distinct / …) as a jude
        relation, via DuckDB SUMMARIZE over the table."""
        import jude

        e = self.get(name)
        rel = self.read(name)
        conn = jude.connect()
        conn.register("_cat_t", rel.to_arrow())
        try:
            return conn.sql("SUMMARIZE SELECT * FROM _cat_t")
        finally:
            pass

    def information_schema(self):
        """The catalog as a queryable jude relation: one row per table with
        format, path, row/column counts, size, versions. Run SQL over it."""
        import jude

        rows = []
        for e in self.tables():
            d = self.describe(e["name"])
            rows.append({
                "name": d["name"],
                "format": d["format"],
                "path": d["path"],
                "num_rows": d.get("num_rows"),
                "num_columns": d.get("num_columns"),
                "size_bytes": d.get("size_bytes"),
                "num_files": d.get("num_files"),
                "num_versions": d.get("num_versions"),
                "updated_at": d.get("updated_at"),
            })
        import pyarrow as pa

        cols = ["name", "format", "path", "num_rows", "num_columns", "size_bytes", "num_files", "num_versions", "updated_at"]
        table = pa.table({c: [r.get(c) for r in rows] for c in cols}) if rows else pa.table({c: [] for c in cols})
        return jude.connect().from_arrow(table)

    def validate(self) -> list[dict]:
        """Drift check: registered tables whose path no longer exists on disk."""
        missing = []
        for e in self.tables():
            p = e["path"].split("*", 1)[0].rstrip("/") if e["format"] == "hive" else e["path"]
            if not os.path.exists(p):
                missing.append({"name": e["name"], "path": e["path"], "status": "missing"})
        return missing


# Process-wide default catalog (JUDE_CATALOG or ~/.jude/catalog.json).
default_catalog = Catalog()

register = default_catalog.register
drop = default_catalog.drop
unregister = default_catalog.drop
discover = default_catalog.discover
list_tables = default_catalog.list
tables = default_catalog.tables
get = default_catalog.get
read = default_catalog.read
versions = default_catalog.versions
describe = default_catalog.describe
column_stats = default_catalog.column_stats
information_schema = default_catalog.information_schema
validate = default_catalog.validate
