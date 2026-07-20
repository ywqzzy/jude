"""jude console — an interactive REPL for querying and governing jude.

Run with ``python -m jude`` (or ``jude.console.main()``). It gives you a SQL
prompt over a live jude connection plus dot-commands to browse the storage
catalog (what tables/storages live underneath), inspect schema and git-like
version history, and see the distributed runner / resource status.

    jude> .tables                 -- catalog: registered/discovered tables
    jude> .discover /warehouse    -- auto-detect + register tables under a root
    jude> .describe db.docs       -- schema, rows, size, versions
    jude> .versions db.docs       -- git-like version history
    jude> .status                 -- runner, workers, resource admission
    jude> SELECT count(*) FROM read_lance('…')
"""

from __future__ import annotations

import sys
from typing import Any

_BANNER = "jude console — SQL + storage governance. Type .help for commands, .quit to exit."


def _print_table(rel: Any, limit: int = 50) -> None:
    """Print a relation/arrow result as a simple aligned table."""
    try:
        tbl = rel.limit(limit).to_arrow() if hasattr(rel, "limit") else rel
    except Exception:
        tbl = rel
    cols = tbl.column_names
    rows = list(zip(*[tbl.column(c).to_pylist() for c in cols])) if cols else []
    widths = [len(c) for c in cols]
    for r in rows:
        for i, v in enumerate(r):
            widths[i] = max(widths[i], len(str(v)))
    if cols:
        print(" | ".join(c.ljust(widths[i]) for i, c in enumerate(cols)))
        print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(str(v).ljust(widths[i]) for i, v in enumerate(r)))
    print(f"({len(rows)} row{'s' if len(rows) != 1 else ''}{' shown' if len(rows) >= limit else ''})")


def _cmd(conn: Any, line: str) -> bool:
    """Handle a dot-command. Returns False to quit, True to continue."""
    import jude

    parts = line.split()
    cmd = parts[0].lower()
    arg = line[len(parts[0]):].strip()
    if cmd in (".quit", ".exit", ".q"):
        return False
    if cmd in (".help", ".h", ".?"):
        print(
            "  .tables                list catalog tables\n"
            "  .discover <root>       auto-detect + register tables under a path\n"
            "  .describe <name>       schema / rows / size / versions of a table\n"
            "  .versions <name>       git-like version history\n"
            "  .read <name>           SELECT * from a catalog table (first rows)\n"
            "  .status                runner / workers / resource admission\n"
            "  .quit                  exit\n"
            "  <sql>                  run SQL"
        )
    elif cmd == ".tables":
        rows = jude.catalog.tables()
        if not rows:
            print("(catalog empty — use .discover <root> or jude.catalog.register)")
        for e in rows:
            print(f"  {e['name']:<30} {e['format']:<8} {e['path']}")
    elif cmd == ".discover":
        if not arg:
            print("usage: .discover <root>")
        else:
            found = jude.catalog.discover(arg)
            print(f"discovered {len(found)} table(s):")
            for e in found:
                print(f"  {e['name']:<30} {e['format']:<8} {e['path']}")
    elif cmd == ".describe":
        import json

        print(json.dumps(jude.catalog.describe(arg), indent=2, default=str))
    elif cmd == ".versions":
        for v in jude.catalog.versions(arg):
            print(f"  {v}")
    elif cmd == ".read":
        _print_table(jude.catalog.read(arg))
    elif cmd == ".status":
        _status()
    else:
        print(f"unknown command {cmd!r}; .help for commands")
    return True


def _status() -> None:
    import jude

    try:
        r = jude.runners.get_or_create_runner()
        print(f"  runner:  {type(r).__name__}")
        if hasattr(r, "num_workers"):
            print(f"  workers: {r.num_workers}  gpus/worker: {getattr(r, 'num_gpus_per_worker', 0)}")
        res = getattr(r, "resources", None)
        if res is not None:
            avail = res.available()
            print(f"  resources available (cpu, gpu, mem, obj): {avail}  in-flight: {res.inflight}")
    except Exception as ex:  # noqa: BLE001
        print(f"  (runner unavailable: {ex})")


def main(argv: list[str] | None = None) -> int:
    import jude

    conn = jude.connect()
    print(_BANNER)
    while True:
        try:
            line = input("jude> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        try:
            if line.startswith("."):
                if not _cmd(conn, line):
                    break
            else:
                _print_table(conn.sql(line.rstrip(";")))
        except Exception as ex:  # noqa: BLE001 - REPL: report, keep going
            print(f"error: {ex}", file=sys.stderr)
    return 0
