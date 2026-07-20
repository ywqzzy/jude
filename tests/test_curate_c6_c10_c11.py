"""C10 PII, C11 decontamination, C6 structured-output parsing."""

from __future__ import annotations

import pyarrow as pa

from jude import curate, structured


# --- C10 PII -----------------------------------------------------------------


def test_redact_pii_replaces_matches():
    t = pa.table({"text": ["reach me at a@b.com or 5551234567", "no pii here at all"]})
    out = curate.redact_pii(t)
    vals = out.column("text").to_pylist()
    assert "[EMAIL]" in vals[0] and "[PHONE]" in vals[0]
    assert "a@b.com" not in vals[0]
    assert vals[1] == "no pii here at all"


def test_redact_pii_out_column():
    t = pa.table({"text": ["ip 10.0.0.1"]})
    out = curate.redact_pii(t, out_column="clean")
    assert "text" in out.column_names and "clean" in out.column_names
    assert "[IPV4]" in out.column("clean").to_pylist()[0]


def test_detect_pii_count():
    t = pa.table({"text": ["a@b.com and c@d.com and 1.2.3.4", "clean"]})
    out = curate.detect_pii(t)
    counts = out.column("pii_count").to_pylist()
    assert counts[0] >= 3
    assert counts[1] == 0


# --- C11 decontamination -----------------------------------------------------


def test_decontaminate_drops_contaminated():
    bench = ["what is the capital of france the answer is paris"]
    docs = pa.table({"id": [1, 2], "text": [
        "trivia question what is the capital of france the answer is paris for sure",
        "an essay about distributed systems and consensus protocols in databases",
    ]})
    out = curate.decontaminate(docs, bench, ngram=5, threshold=0.15)
    assert out.column("id").to_pylist() == [2]


def test_decontaminate_annotate_mode():
    bench = ["the mitochondria is the powerhouse of the cell"]
    docs = pa.table({"text": [
        "recall that the mitochondria is the powerhouse of the cell in biology",
        "rust ownership and borrowing prevent data races at compile time",
    ]})
    out = curate.decontaminate(docs, bench, ngram=4, reason_column="contam")
    ratios = out.column("contam").to_pylist()
    assert ratios[0] > ratios[1]
    assert out.num_rows == 2  # nothing dropped in annotate mode


# --- C6 structured output (parsing/schema, no live LLM) ----------------------


def test_build_system_message_has_schema():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    msg = structured.build_system_message(schema)
    assert "JSON" in msg and '"a"' in msg


def test_parse_plain_json():
    assert structured._parse_json('{"x": 1, "y": "z"}') == {"x": 1, "y": "z"}


def test_parse_fenced_json():
    raw = 'Here you go:\n```json\n{"x": 2}\n```\nthanks'
    assert structured._parse_json(raw) == {"x": 2}


def test_parse_embedded_json():
    raw = 'The answer is {"sentiment": "positive", "score": 0.9} based on the text'
    assert structured._parse_json(raw) == {"sentiment": "positive", "score": 0.9}


def test_parse_bad_json_returns_none():
    assert structured._parse_json("not json at all") is None
    assert structured._parse_json("[1,2,3]") is None  # not an object
