"""L0.2 + L1.1: WARC reading (CommonCrawl form) + HTML->text extraction."""

from __future__ import annotations

import io
import tempfile

import pyarrow as pa
import pytest

from jude import curate


# --- L1.1 HTML -> text (zero-dep) --------------------------------------------

def test_html_to_text_strips_tags_and_scripts():
    html = ("<html><head><style>.x{color:red}</style>"
            "<script>var a=1;</script></head><body>"
            "<h1>Title</h1><p>Hello &amp; welcome to the &lt;page&gt;.</p>"
            "</body></html>")
    out = curate.html_to_text(pa.table({"html": [html]}))
    text = out.column("text")[0].as_py()
    assert "Title" in text and "Hello & welcome" in text
    assert "color:red" not in text and "var a=1" not in text   # script/style gone
    assert "<h1>" not in text and "<p>" not in text and "<script>" not in text  # tags gone
    assert "<page>" in text                                     # &lt;page&gt; unescaped to content


def test_html_to_text_out_column():
    out = curate.html_to_text(pa.table({"html": ["<p>hi</p>"]}), out_column="body")
    assert out.column("body")[0].as_py() == "hi"


# --- L0.2 WARC ---------------------------------------------------------------

def _make_warc() -> str:
    warcio = pytest.importorskip("warcio")
    from warcio.warcwriter import WARCWriter
    from warcio.statusandheaders import StatusAndHeaders

    path = tempfile.mkdtemp() + "/test.warc.gz"
    with open(path, "wb") as fh:
        w = WARCWriter(fh, gzip=True)
        for i, body in enumerate([b"<html><body><p>doc one</p></body></html>",
                                   b"<html><body><p>doc two</p></body></html>"]):
            http = StatusAndHeaders("200 OK", [("Content-Type", "text/html")],
                                    protocol="HTTP/1.0")
            rec = w.create_warc_record(f"http://example.com/{i}", "response",
                                       payload=io.BytesIO(body), http_headers=http)
            w.write_record(rec)
    return path


def test_read_warc_records():
    from jude import warc

    pytest.importorskip("warcio")
    path = _make_warc()
    recs = list(warc.read_warc_records(f"file://{path}"))
    assert len(recs) == 2
    assert recs[0]["url"] == "http://example.com/0"
    assert b"doc one" in recs[0]["content"]


def test_warc_to_table_extract_text():
    from jude import warc

    pytest.importorskip("warcio")
    path = _make_warc()
    tbl = warc.warc_to_table(f"file://{path}", extract_text=True)
    assert tbl.num_rows == 2
    assert "text" in tbl.column_names and "url" in tbl.column_names
    texts = tbl.column("text").to_pylist()
    assert any("doc one" in t for t in texts)
    assert all("<" not in t for t in texts)                    # HTML stripped
