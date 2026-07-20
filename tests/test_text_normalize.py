"""L1.2: text normalization — mojibake repair + unicode NFC. Dependency-free."""

from __future__ import annotations

import unicodedata

import pyarrow as pa

from jude import curate


def test_fix_encoding_repairs_mojibake():
    # 'café' encoded UTF-8 then mis-decoded as latin-1 => 'cafÃ©'
    bad = "café".encode("utf-8").decode("latin-1")
    out = curate.fix_encoding(pa.table({"text": [bad]}))
    assert out.column("text")[0].as_py() == "café"


def test_fix_encoding_leaves_clean_text():
    clean = "a perfectly normal english sentence."
    out = curate.fix_encoding(pa.table({"text": [clean]}))
    assert out.column("text")[0].as_py() == clean          # never corrupts clean text


def test_fix_encoding_repairs_cyrillic():
    bad = "русский".encode("utf-8").decode("latin-1")
    out = curate.fix_encoding(pa.table({"text": [bad]}))
    assert out.column("text")[0].as_py() == "русский"


def test_normalize_unicode_nfc():
    decomposed = "é"                                  # e + combining acute
    out = curate.normalize_unicode(pa.table({"text": [decomposed]}))
    got = out.column("text")[0].as_py()
    assert got == unicodedata.normalize("NFC", "é")
    assert len(got) == 1                                    # composed to single char


def test_normalize_idempotent():
    t = pa.table({"text": ["already normal café"]})
    once = curate.normalize_unicode(t).column("text")[0].as_py()
    twice = curate.normalize_unicode(curate.normalize_unicode(t)).column("text")[0].as_py()
    assert once == twice


def test_out_column():
    out = curate.fix_encoding(pa.table({"text": ["hi"]}), out_column="clean")
    assert "clean" in out.column_names and "text" in out.column_names
