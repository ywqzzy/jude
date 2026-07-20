"""Two-phase aggregate decomposition for distributed GROUP BY.

Turns a set of aggregate expressions into (partial_sql, final_sql) so a GROUP BY
can run as: partial-aggregate per partition -> union -> final merge. Exact for
decomposable aggregates.

Decomposable here: COUNT(*), COUNT(x), SUM(x), MIN(x), MAX(x), AVG(x) (as
SUM/COUNT), and STDDEV/VARIANCE (pop+samp) via (count, sum, sum-of-squares).
Non-decomposable aggregates (MEDIAN/QUANTILE/PERCENTILE, COUNT(DISTINCT),
STRING_AGG, CORR, ...) raise NotDecomposable so the caller can fall back to a
single-node aggregate instead of failing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


class NotDecomposable(ValueError):
    """Raised when an aggregate can't be computed as partial->merge two-phase
    (so the distributed executor should fall back to a single-node aggregate)."""


@dataclass
class Agg:
    func: str  # COUNT / SUM / MIN / MAX / AVG / STDDEV_SAMP / STDDEV_POP / VAR_SAMP / VAR_POP
    arg: str  # column/expr or "*"
    out: str  # output column name


_DECOMPOSABLE = "COUNT|SUM|MIN|MAX|AVG|STDDEV_SAMP|STDDEV_POP|STDDEV|VAR_SAMP|VAR_POP|VARIANCE|VAR"
_AGG_RE = re.compile(rf"^\s*({_DECOMPOSABLE})\s*\((.*)\)\s*(?:AS\s+(\w+))?\s*$", re.IGNORECASE)
# canonicalize DuckDB aliases
_CANON = {"STDDEV": "STDDEV_SAMP", "VARIANCE": "VAR_SAMP", "VAR": "VAR_SAMP"}


def parse_agg(expr: str) -> Agg:
    m = _AGG_RE.match(expr)
    if not m:
        raise NotDecomposable(f"aggregate not two-phase decomposable: {expr!r}")
    func = m.group(1).upper()
    func = _CANON.get(func, func)
    arg = m.group(2).strip()
    # COUNT(DISTINCT x) / any DISTINCT aggregate is NOT decomposable by summing
    # partials (partials overlap) — needs a shuffle-by-value; fall back.
    if arg.upper().startswith("DISTINCT"):
        raise NotDecomposable(f"DISTINCT aggregate needs a shuffle, not two-phase: {expr!r}")
    out = m.group(3) or f"{func.lower()}_{re.sub(r'[^a-zA-Z0-9]', '_', arg)}"
    return Agg(func=func, arg=arg, out=out)


def is_decomposable(aggs: list[str]) -> bool:
    """True if every aggregate can run as two-phase (so distribution is exact)."""
    try:
        for a in aggs:
            parse_agg(a)
        return True
    except NotDecomposable:
        return False


def build_two_phase(
    group_by: list[str],
    aggs: list[str],
    source_table: str = "part",
    partial_table: str = "partials",
) -> tuple[str, str]:
    """Return (partial_sql, final_sql).

    partial_sql runs over ``source_table``; final_sql runs over ``partial_table``.
    Raises NotDecomposable if any aggregate can't be two-phased.
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
        elif a.func in ("STDDEV_SAMP", "STDDEV_POP", "VAR_SAMP", "VAR_POP"):
            # variance via (n, Σx, Σx²): Var = (Σx² - (Σx)²/n) / d, d = n (pop) or n-1 (samp)
            e = f"({a.arg})"
            partial_selects.append(f"SUM({e}) AS _vs_{a.out}")
            partial_selects.append(f"SUM(({e})*({e})) AS _vq_{a.out}")
            partial_selects.append(f"COUNT({e}) AS _vc_{a.out}")
            n = f"SUM(_vc_{a.out})"
            sx = f"SUM(_vs_{a.out})"
            sq = f"SUM(_vq_{a.out})"
            num = f"({sq} - ({sx})*({sx})/NULLIF({n},0))"  # Σx² - (Σx)²/n
            denom = f"NULLIF({n},0)" if a.func == "VAR_POP" else f"NULLIF({n}-1,0)"
            if a.func == "STDDEV_POP":
                denom = f"NULLIF({n},0)"
            var = f"{num} / {denom}"
            if a.func.startswith("STDDEV"):
                final_selects.append(f"sqrt({var}) AS {a.out}")
            else:
                final_selects.append(f"{var} AS {a.out}")
        else:  # pragma: no cover
            raise NotDecomposable(f"unhandled aggregate {a.func}")

    group_clause = f" GROUP BY {', '.join(group_cols)}" if group_cols else ""
    partial_sql = f"SELECT {', '.join(partial_selects)} FROM {source_table}{group_clause}"
    final_sql = f"SELECT {', '.join(final_selects)} FROM {partial_table}{group_clause}"
    return partial_sql, final_sql
