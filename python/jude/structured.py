"""jude.structured — structured LLM output (C6): guided JSON extraction.

Get an LLM to return strict, schema-conforming JSON per row — for data
annotation, synthetic-data generation, and information extraction pipelines.
Built on ``jude.ai.prompt`` (which maps a prompt column -> a response column):
we inject a JSON-schema instruction into the system message so the model emits
JSON, then parse + validate each response against the schema, adding one column
per field (invalid rows -> nulls + an ``_extract_error`` column).

Works with a plain JSON-schema dict OR a Pydantic model (its schema is used and
each row validated through it).

    schema = {"type": "object", "properties": {
        "sentiment": {"type": "string"}, "score": {"type": "number"}}}
    out = jude.structured.extract(rel, "review_text", schema)
    # out has columns: sentiment, score (+ _extract_error where parsing failed)
"""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = ["extract", "build_system_message"]


def _schema_of(return_format: Any) -> tuple[dict, Any]:
    """Return (json_schema_dict, pydantic_model_or_None)."""
    # Pydantic model?
    if hasattr(return_format, "model_json_schema"):
        return return_format.model_json_schema(), return_format
    if hasattr(return_format, "schema") and callable(getattr(return_format, "schema")):
        try:
            return return_format.schema(), return_format  # pydantic v1
        except Exception:  # noqa: BLE001
            pass
    if isinstance(return_format, dict):
        return return_format, None
    raise TypeError("return_format must be a JSON-schema dict or a Pydantic model")


def build_system_message(schema: dict, extra: str | None = None) -> str:
    """The instruction that steers the model to emit only conforming JSON."""
    msg = (
        "You are a precise information-extraction engine. Respond with ONLY a "
        "single JSON object that conforms to this JSON schema — no prose, no "
        "markdown fences:\n" + json.dumps(schema, ensure_ascii=False)
    )
    if extra:
        msg = extra + "\n\n" + msg
    return msg


_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_json(text: str) -> dict | None:
    if text is None:
        return None
    t = text.strip()
    # strip markdown fences if present
    m = _FENCE.search(t)
    if m:
        t = m.group(1)
    else:
        # else take the first {...} balanced-ish span
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end > start:
            t = t[start : end + 1]
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:  # noqa: BLE001
        return None


def extract(
    rel: Any,
    column: str,
    return_format: Any,
    *,
    provider: str | None = None,
    model: str | None = None,
    system_message: str | None = None,
    raw_column: str = "_raw_response",
) -> Any:
    """Extract structured fields from ``column`` per row via an LLM, validated
    against ``return_format`` (JSON-schema dict or Pydantic model).

    Returns a jude Relation with one column per top-level schema field, plus
    ``_extract_error`` (None when the row parsed+validated, else the reason).
    """
    import pyarrow as pa

    import jude

    schema, pyd = _schema_of(return_format)
    sys_msg = build_system_message(schema, extra=system_message)

    # run the LLM: prompt column -> raw response column
    responded = rel.prompt(column, provider=provider, model=model, system_message=sys_msg, output_column=raw_column)
    tbl = responded.to_arrow()
    raws = tbl.column(raw_column).to_pylist()

    fields = list((schema.get("properties") or {}).keys())
    cols: dict[str, list] = {f: [] for f in fields}
    errors: list = []
    for raw in raws:
        obj = _parse_json(raw)
        err = None
        if obj is None:
            err = "unparseable_json"
        elif pyd is not None:
            try:
                validated = pyd(**obj)
                obj = validated.model_dump() if hasattr(validated, "model_dump") else validated.dict()
            except Exception as e:  # noqa: BLE001
                err = f"validation_error: {type(e).__name__}"
                obj = None
        for f in fields:
            cols[f].append(obj.get(f) if obj else None)
        errors.append(err)

    out = tbl
    for f in fields:
        # infer a coarse Arrow type per field from the schema
        out = out.append_column(f, pa.array(cols[f]))
    out = out.append_column("_extract_error", pa.array(errors, type=pa.string()))
    con = jude.connect()
    return con.from_arrow(out)
