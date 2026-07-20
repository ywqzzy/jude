"""jude.lineage — reproducibility: bind a pipeline config to the data it made.

For research you must answer "exactly which data + which config produced this
checkpoint". This computes a stable signature of a curation/pipeline config and
records it alongside the output (a JSON sidecar next to the Lance/shard path, on
top of Lance's own version history), linking:

    pipeline config (hashed)  ->  input dataset versions  ->  output version

    sig = pipeline_signature(cfg)
    lin = dataset_lineage(cfg, inputs={"web": 3, "code": 7}, output_version=12)
    write_lineage("train.lance", lin)   # -> train.lance.lineage.json
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def pipeline_signature(config: Any) -> str:
    """Deterministic 128-bit hex signature of a pipeline/curation config (any
    JSON-able value — dict of params, list of stage specs, ...). Same config →
    same signature, so a data version is reproducible from its config."""
    canon = json.dumps(config, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.blake2b(canon.encode("utf-8"), digest_size=16).hexdigest()


def dataset_lineage(
    config: Any,
    *,
    inputs: dict | None = None,
    output_version: Any = None,
    label: str = "curation",
    extra: dict | None = None,
) -> dict:
    """Build a lineage record binding a config signature to input dataset
    versions and the output version. ``inputs`` maps a source name to its version
    (int / tag). Pure data; persist with ``write_lineage``."""
    rec = {
        "label": label,
        "pipeline_signature": pipeline_signature(config),
        "config": config,
        "inputs": inputs or {},
        "output_version": output_version,
    }
    if extra:
        rec["extra"] = extra
    return rec


def _sidecar(path: str) -> str:
    return path.rstrip("/") + ".lineage.json"


def write_lineage(path: str, lineage: dict) -> str:
    """Write a lineage record as a JSON sidecar next to ``path`` (the output
    dataset). Returns the sidecar path. The lineage travels with the data."""
    p = _sidecar(path)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(lineage, f, indent=2, ensure_ascii=False, default=str)
    return p


def read_lineage(path: str) -> dict | None:
    """Read the lineage sidecar for ``path`` (None if absent)."""
    import os

    p = _sidecar(path)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)
