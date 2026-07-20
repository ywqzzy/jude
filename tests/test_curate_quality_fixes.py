"""C4/C6 curation-quality fixes + edge cases (all deterministic, no model).

- credit_card PII now requires a valid Luhn checksum (no false positives).
- language confidence is margin-based (clearly-English text scores high, not ~0.08).
- decontamination is dilution-resistant (a benchmark question buried in a long
  doc still scores ~1.0 — benchmark-side coverage, not doc-side ratio).
"""

from __future__ import annotations

import pyarrow as pa

from jude import curate
from jude.jude import _curate


# --- C6: PII / Luhn ----------------------------------------------------------

def test_credit_card_requires_luhn():
    valid = _curate.detect_pii("pay with 4111111111111111 today")   # Luhn-valid Visa
    invalid = _curate.detect_pii("order number 1234567812345678")   # Luhn-invalid
    assert any(k == "credit_card" for k, _, _ in valid)
    assert not any(k == "credit_card" for k, _, _ in invalid)


def test_credit_card_with_separators():
    spans = _curate.detect_pii("card 4111 1111 1111 1111 charged")
    assert any(k == "credit_card" for k, _, _ in spans)


def test_pii_email_url_ip_ssn_still_detected():
    spans = _curate.detect_pii("mail a@b.com site http://x.io ip 8.8.8.8 ssn 123456789")
    kinds = {k for k, _, _ in spans}
    assert {"email", "url", "ipv4", "ssn"} <= kinds


def test_redact_pii_replaces_and_counts():
    t = pa.table({"text": ["reach me at bob@corp.com or 4111111111111111"]})
    out = curate.redact_pii(t)
    red = out.column("text")[0].as_py()
    assert "[EMAIL]" in red and "[CREDIT_CARD]" in red
    assert "bob@corp.com" not in red


def test_detect_pii_count_column():
    t = pa.table({"text": ["a@b.com and 8.8.8.8", "clean text no pii here"]})
    out = curate.detect_pii(t)
    counts = out.column("pii_count").to_pylist()
    assert counts[0] >= 2 and counts[1] == 0


# --- C4: language ------------------------------------------------------------

def test_english_confidence_is_high():
    t = pa.table({"text": ["The quick brown fox jumps over the lazy dog every morning"]})
    out = curate.detect_language(t)
    assert out.column("lang")[0].as_py() == "en"
    assert out.column("lang_conf")[0].as_py() >= 0.5   # not the old ~0.08


def test_language_filter_keeps_clear_english():
    t = pa.table({"text": ["the cat and the dog are in the house",
                           "这是一段完全的中文文本没有英文"]})
    out = curate.language_filter(t, keep="en", min_confidence=0.5)
    assert out.num_rows == 1
    assert "cat" in out.column("text")[0].as_py()


def test_japanese_with_kana_detected():
    t = pa.table({"text": ["これは日本語のテストです"]})
    out = curate.detect_language(t)
    assert out.column("lang")[0].as_py() == "ja"


# --- C11: decontamination (dilution-resistant) -------------------------------

def test_decontamination_catches_buried_benchmark():
    bench = ["what is the capital of france"]
    padding = "some unrelated filler sentence here " * 100
    docs = pa.table({"text": [
        "what is the capital of france",                    # exact
        f"intro text {' '.join(['what is the capital of france'])} {padding}",  # buried
        "a totally different document about ocean biology",  # clean
    ]})
    out = curate.decontaminate(docs, bench, ngram=5, threshold=0.5)
    kept = out.column("text").to_pylist()
    # both contaminated docs dropped (even the long buried one), clean kept
    assert len(kept) == 1
    assert "ocean biology" in kept[0]


def test_decontamination_reason_column_scores():
    bench = ["the mitochondria is the powerhouse of the cell"]
    docs = pa.table({"text": [
        "the mitochondria is the powerhouse of the cell " + ("noise word " * 200),
        "an unrelated document about medieval european history and castles",
    ]})
    out = curate.decontaminate(docs, bench, ngram=5, reason_column="contam")
    scores = out.column("contam").to_pylist()
    assert scores[0] > 0.9    # buried benchmark -> high coverage despite length
    assert scores[1] < 0.1    # unrelated -> low
