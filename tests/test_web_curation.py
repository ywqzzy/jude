"""C1: web-corpus curation — C4-style line cleaning + cross-document line dedup."""

from __future__ import annotations

import pyarrow as pa

from jude import curate


# --- c4_line_filter ----------------------------------------------------------

def test_c4_drops_nav_and_keeps_prose():
    doc = "\n".join([
        "Home | About | Contact",                       # nav, no terminal punct
        "This is a proper sentence that ends well.",     # prose -> keep
        "javascript:void(0)",                            # marker -> drop
        "Read more",                                     # marker + short -> drop
        "Another complete thought, stated clearly here.",  # prose -> keep
        "ok",                                            # too short -> drop
    ])
    out = curate.c4_line_filter(pa.table({"text": [doc]}))
    lines = out.column("text")[0].as_py().splitlines()
    assert "This is a proper sentence that ends well." in lines
    assert "Another complete thought, stated clearly here." in lines
    assert all("javascript" not in l.lower() for l in lines)
    assert "Home | About | Contact" not in lines
    assert len(lines) == 2


def test_c4_terminal_punct_optional():
    doc = "a line without terminal punctuation but enough words here"
    keep = curate.c4_line_filter(pa.table({"text": [doc]}), require_terminal_punct=False)
    assert keep.column("text")[0].as_py() == doc
    drop = curate.c4_line_filter(pa.table({"text": [doc]}), require_terminal_punct=True)
    assert drop.column("text")[0].as_py() == ""


def test_c4_out_column():
    out = curate.c4_line_filter(pa.table({"text": ["This sentence is clean and complete."]}),
                                out_column="clean")
    assert "clean" in out.column_names and "text" in out.column_names


# --- corpus_line_dedup -------------------------------------------------------

def test_corpus_line_dedup_removes_boilerplate():
    footer = "Copyright 2026 Example Corp"
    docs = [
        f"Unique intro one.\n{footer}\nBody of page one.",
        f"Unique intro two.\n{footer}\nBody of page two.",
        f"Unique intro three.\n{footer}\nBody of page three.",
    ]
    out = curate.corpus_line_dedup(pa.table({"text": docs}), min_docs=2)
    cleaned = out.column("text").to_pylist()
    # the footer (in all 3 docs) is gone; unique lines remain
    assert all(footer.lower() not in c.lower() for c in cleaned)
    assert "Unique intro one." in cleaned[0]
    assert "Body of page two." in cleaned[1]


def test_corpus_line_dedup_keeps_rare_lines():
    docs = ["shared line\nrare one", "shared line\nrare two"]
    out = curate.corpus_line_dedup(pa.table({"text": docs}), min_docs=2)
    cleaned = out.column("text").to_pylist()
    assert "shared line" not in cleaned[0]      # in 2 docs -> dropped
    assert "rare one" in cleaned[0]             # in 1 doc -> kept
    assert "rare two" in cleaned[1]


def test_corpus_line_dedup_threshold():
    docs = ["x\ncommon", "y\ncommon", "z\nunique"]
    # min_docs=3 -> "common" appears in only 2 -> kept
    out = curate.corpus_line_dedup(pa.table({"text": docs}), min_docs=3)
    assert "common" in out.column("text")[0].as_py()


def test_corpus_line_dedup_normalize():
    docs = ["Header Text", "header text", "HEADER TEXT"]
    out = curate.corpus_line_dedup(pa.table({"text": docs}), min_docs=2, normalize=True)
    # case-insensitive: all three are the "same" line, in 3 docs -> dropped
    assert all(c == "" for c in out.column("text").to_pylist())
