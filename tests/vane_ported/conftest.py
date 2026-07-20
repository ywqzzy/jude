"""Harness to run Vane's own test suite against jude.

Vane tests do ``import duckdb`` and use fixtures like ``duckdb_cursor``. This
conftest makes ``duckdb`` resolve to jude (via sys.modules aliasing) and
re-provides Vane's fixtures, so we can drop Vane test files in here and measure
how much of Vane's behavior jude actually matches.

Only behavioral (user-facing) suites are ported — the C++/FTE-internal suites
that test Vane's specific engine internals are intentionally excluded.
"""

import sys
from pathlib import Path

import pytest

import jude as _jude

# Make `import duckdb` inside ported Vane tests resolve to jude.
sys.modules.setdefault("duckdb", _jude)


# --- CI gating: xfail the still-unaligned Vane tests -------------------------
# The full Vane suite is checked in so CI runs it on every commit. Tests that
# jude does not yet match (deep numpy/type-fidelity, a few DuckDB-internal
# features) are listed in ``known_gaps.txt`` and marked xfail here, so a green
# CI run means "no regression in the aligned set". When a fix makes one pass it
# surfaces as XPASS — remove that line from known_gaps.txt to lock the win in.
_KNOWN_GAPS_FILE = Path(__file__).parent / "known_gaps.txt"


def _load_known_gaps() -> set[str]:
    if not _KNOWN_GAPS_FILE.exists():
        return set()
    gaps = set()
    for line in _KNOWN_GAPS_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            gaps.add(line)
    return gaps


def pytest_collection_modifyitems(config, items):
    gaps = _load_known_gaps()
    if not gaps:
        return
    for item in items:
        # item.nodeid is e.g. "tests/vane_ported/test_x.py::TestC::test_y[param]"
        if item.nodeid in gaps:
            item.add_marker(
                pytest.mark.xfail(
                    reason="Known unaligned Vane behavior (tracked in known_gaps.txt)",
                    strict=False,
                )
            )



@pytest.fixture
def duckdb_cursor():
    connection = _jude.connect("")
    yield connection
    try:
        connection.close()
    except Exception:
        pass


@pytest.fixture
def integers(duckdb_cursor):
    cursor = duckdb_cursor
    cursor.execute("CREATE TABLE integers (i integer)")
    cursor.execute(
        "INSERT INTO integers VALUES (0),(1),(2),(3),(4),(5),(6),(7),(8),(9),(NULL)"
    )
    yield
    cursor.execute("drop table integers")


@pytest.fixture
def timestamps(duckdb_cursor):
    cursor = duckdb_cursor
    cursor.execute("CREATE TABLE timestamps (t timestamp)")
    cursor.execute(
        "INSERT INTO timestamps VALUES ('1992-09-20 11:30:00'),('1992-09-20 12:30:00'),(NULL)"
    )
    yield
    cursor.execute("drop table timestamps")


@pytest.fixture
def duckdb_empty_cursor(duckdb_cursor):
    return duckdb_cursor
