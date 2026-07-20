"""Two-phase aggregate decomposition for distributed GROUP BY.

Turns a set of aggregate expressions into (partial_sql, final_sql) so a GROUP BY
can run as: partial-aggregate per partition -> union -> final merge. Exact for
decomposable aggregates.

Supported aggregate functions: COUNT(*), COUNT(x), SUM(x), MIN(x), MAX(x),
AVG(x) (decomposed as SUM/COUNT).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Agg:
    func: str  # COUNT / SUM / MIN / MAX / AVG
    arg: str  # column/expr or "*"
    out: str  # output column name


_AGG_RE = re.compile(r"^\s*(COUNT|SUM|MIN|MAX|AVG)\s*\((.*)\)\s*(?:AS\s+(\w+))?\s*$", re.IGNORECASE)


def parse_agg(expr: str) -> Agg:
    m = _AGG_RE.match(expr)
    if not m:
        raise ValueError(f"unsupported aggregate expression for distribution: {expr!r}")
    func = m.group(1).upper()
    arg = m.group(2).strip()
    out = m.group(3) or f"{func.lower()}_{re.sub(r'[^a-zA-Z0-9]', '_', arg)}"
    return Agg(func=func, arg=arg, out=out)


def build_two_phase(
    group_by: list[str],
    aggs: list[str],
    source_table: str = "part",
    partial_table: str = "partials",
) -> tuple[str, str]:
    """Return (partial_sql, final_sql).

    partial_sql runs over ``source_table``; final_sql runs over ``partial_table``.
    """
    parsed = [parse_agg(a) for a in aggs]
    group_cols = list(group_by)

    partial_selects = list(group_cols)
    final_selects = list(group_cols)

    for a in parsed:
        if a.func == "COUNT":
            # partial: COUNT(arg) as _cnt_out ; final: SUM(_cnt_out) as out
            p = f"COUNT({a.arg}) AS _cnt_{a.out}"
            f = f"SUM(_cnt_{a.out}) AS {a.out}"
            partial_selects.append(p)
            final_selects.append(f)
        elif a.func == "SUM":
            partial_selects.append(f"SUM({a.arg}) AS _sum_{a.out}")
            final_selects.append(f"SUM(_sum_{a.out}) AS {a.out}")
        elif a.func == "MIN":
            partial_selects.append(f"MIN({a.arg}) AS _min_{a.out}")
            final_selects.append(f"MIN(_min_{a.out}) AS {a.out}")
        elif a.func == "MAX":
            partial_selects.append(f"MAX({a.arg}) AS _max_{a.out}")
            final_selects.append(f"MAX(_max_{a.out}) AS {a.out}")
        elif a.func == "AVG":
            # decompose into SUM + COUNT, recombine as SUM/COUNT
            partial_selects.append(f"SUM({a.arg}) AS _avgsum_{a.out}")
            partial_selects.append(f"COUNT({a.arg}) AS _avgcnt_{a.out}")
            final_selects.append(
                f"SUM(_avgsum_{a.out}) / NULLIF(SUM(_avgcnt_{a.out}), 0) AS {a.out}"
            )
        else:  # pragma: no cover
            raise ValueError(f"unhandled aggregate {a.func}")

    group_clause = f" GROUP BY {', '.join(group_cols)}" if group_cols else ""
    partial_sql = f"SELECT {', '.join(partial_selects)} FROM {source_table}{group_clause}"
    final_sql = f"SELECT {', '.join(final_selects)} FROM {partial_table}{group_clause}"
    return partial_sql, final_sql
