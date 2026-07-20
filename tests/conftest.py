"""Shared pytest fixtures / hygiene for the jude test suite."""

import sys

import pytest

import jude

# Make `import duckdb` resolve to jude everywhere (needed by tests/vane_ported/,
# which run Vane's own suite against jude). Set at conftest import so it's in
# place before any test module is collected.
sys.modules.setdefault("duckdb", jude)


@pytest.fixture(autouse=True)
def _teardown_udf_pools():
    """Tear down cached subprocess UDF pools after each test so worker
    processes don't accumulate across the suite (keeps the full run fast).

    Note: we do NOT reset the Ray runner here — its actor pool is intentionally
    reused across a module's tests (resetting would force costly re-creation).
    """
    yield
    try:
        jude.shutdown_udf_pools()
    except Exception:
        pass
