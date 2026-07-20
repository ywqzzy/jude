"""jude.warc — read WARC/WET archives (CommonCrawl) into Arrow.

The front of a pretraining pipeline: CommonCrawl ships as WARC (raw HTTP
responses) and WET (extracted text). This streams records into Arrow so the
curation pipeline (extract → clean → dedup → quality → tokenize) can start from
raw crawl data. Reads any fsspec URL (local / s3:// MinIO / gs://) — the payload
is decompressed by warcio. ``warcio`` is an optional dependency.

    for rec in read_warc_records("crawl.warc.gz"):
        rec["url"], rec["content"]           # bytes payload (HTML for a response)
    tbl = warc_to_table("s3://cc/*.warc.gz", extract_text=True)   # url + text
"""

from __future__ import annotations

from typing import Any, Iterator

import pyarrow as pa


def _iter_stream(url: str, **storage_options: Any):
    """Open a WARC URL as a binary stream via fsspec (local/s3/gs/memory)."""
    from jude.storage import resolve

    fs, path = resolve(url, **storage_options)
    return fs.open(path, "rb")


def read_warc_records(
    url: str,
    *,
    record_type: str = "response",
    limit: int | None = None,
    **storage_options: Any,
) -> Iterator[dict]:
    """Yield ``{"url", "content_type", "content"}`` per WARC record of
    ``record_type`` (``response`` for HTML, ``conversion`` for WET text). Streams
    with bounded memory (one record at a time)."""
    try:
        from warcio.archiveiterator import ArchiveIterator
    except ImportError as e:  # pragma: no cover
        raise ImportError("read_warc_records needs `warcio` (pip install warcio)") from e

    n = 0
    with _iter_stream(url, **storage_options) as stream:
        for rec in ArchiveIterator(stream):
            if rec.rec_type != record_type:
                continue
            payload = rec.content_stream().read()
            yield {
                "url": rec.rec_headers.get_header("WARC-Target-URI") or "",
                "content_type": rec.http_headers.get_header("Content-Type") if rec.http_headers else "",
                "content": payload,
            }
            n += 1
            if limit is not None and n >= limit:
                return


def warc_to_table(
    url: str,
    *,
    record_type: str = "response",
    extract_text: bool = False,
    limit: int | None = None,
    **storage_options: Any,
) -> pa.Table:
    """Read WARC records into an Arrow table (url, content_type, and ``content``
    bytes — or ``text`` if ``extract_text``, running the built-in HTML→text).
    Accepts a glob URL across shards."""
    from jude.storage import glob as _glob

    urls = _glob(url, **storage_options) if any(c in url for c in "*?[") else [url]
    rows_url: list = []
    rows_ct: list = []
    rows_payload: list = []
    for u in urls:
        for rec in read_warc_records(u, record_type=record_type, limit=limit, **storage_options):
            rows_url.append(rec["url"])
            rows_ct.append(rec["content_type"])
            rows_payload.append(rec["content"])
    if extract_text:
        from jude.curate import _html_to_text

        texts = [_html_to_text(p.decode("utf-8", "ignore")) for p in rows_payload]
        return pa.table({"url": pa.array(rows_url, type=pa.string()),
                         "text": pa.array(texts, type=pa.string())})
    return pa.table({
        "url": pa.array(rows_url, type=pa.string()),
        "content_type": pa.array(rows_ct, type=pa.string()),
        "content": pa.array(rows_payload, type=pa.binary()),
    })
