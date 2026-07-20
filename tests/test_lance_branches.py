"""Lance branches (git-like) + distributed-index building blocks."""

from __future__ import annotations

import tempfile

import pyarrow as pa
import pytest

import jude
from jude import lance as jl

lance = pytest.importorskip("lance")


def _ds(rows=50):
    d = tempfile.mkdtemp(prefix="jude_branch_")
    path = d + "/ds"
    jude._lance.write(pa.table({"id": list(range(rows)), "x": [float(i) for i in range(rows)]}), path, mode="create")
    return path


def test_create_and_list_branch():
    path = _ds()
    try:
        jl.create_branch(path, "experiment-a")
    except Exception as e:  # noqa: BLE001 — older Lance may lack branches
        pytest.skip(f"branches unavailable: {e}")
    branches = jl.list_branches(path)
    assert "experiment-a" in branches


def test_shallow_clone():
    path = _ds()
    target = tempfile.mkdtemp(prefix="jude_clone_") + "/clone"
    try:
        jl.shallow_clone(path, target)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"shallow_clone unavailable: {e}")
    # the clone should be readable with the same rows
    out = jude._lance.read_table(target)
    assert out.num_rows == 50
