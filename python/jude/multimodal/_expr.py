"""jude.multimodal._expr — apply multimodal expression ops that fall back to the
Python decoders (audio/video/document), for ops with no pure-Rust codec.

The Rust kernel (`src/multimodal`) handles image/url ops directly. Ops whose
codec lives only in Python (audio via soundfile, video via PyAV, document via
pypdf) route here: `Relation.with_column` calls `apply_expr` with the
materialized Arrow table, the input column, the output column, and the op specs.

Audio decode is 1:1 (row-preserving) and yields a struct column
`{samples: list<float32>, sample_rate, num_frames, num_channels}` — queryable
like the image struct. (Video frames / document pages are 1:many and go through
a separate explode entry, not `with_column`.)
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from jude.multimodal.decoders import decode_audio_batch, decode_document_batch, decode_video_batch

# Op names handled here rather than in the Rust kernel.
FALLBACK_OPS = {"audio_decode"}


def apply_expr(
    table: pa.Table,
    input_column: str,
    output_column: str,
    op_specs: list[tuple[str, dict]],
) -> pa.Table:
    """Apply a fallback op chain, returning a new table with `output_column`.

    Only single-op fallback chains are supported for now (a decode). Mixed
    Rust+fallback chains and multi-fallback chains raise.
    """
    fb = [(name, kw) for (name, kw) in op_specs if name in FALLBACK_OPS]
    if len(op_specs) != 1 or len(fb) != 1:
        raise NotImplementedError(
            "only a single fallback multimodal op (e.g. audio.decode()) is supported per with_column"
        )
    name, kw = fb[0]
    if name == "audio_decode":
        return _audio_decode(table, input_column, output_column, kw)
    raise NotImplementedError(f"unsupported fallback multimodal op: {name}")


def _audio_decode(table: pa.Table, input_column: str, output_column: str, kw: dict[str, Any]) -> pa.Table:
    mono = bool(kw.get("mono", True))
    target_sr = kw.get("sample_rate")
    decoded = decode_audio_batch(
        table,
        audio_column=input_column,
        out_column="__samples",
        mono=mono,
        target_sample_rate=int(target_sr) if target_sr else None,
    )
    # Pack the decoder's flat outputs into one queryable struct column.
    struct = pa.StructArray.from_arrays(
        [
            decoded.column("__samples").combine_chunks(),
            decoded.column("sample_rate").combine_chunks(),
            decoded.column("num_frames").combine_chunks(),
            decoded.column("num_channels").combine_chunks(),
        ],
        names=["samples", "sample_rate", "num_frames", "num_channels"],
    )
    # Return the original table + the struct column (drop the decoder's temp cols).
    out = table.append_column(output_column, struct)
    return out


__all__ = ["apply_expr", "explode", "FALLBACK_OPS"]


def explode(table: pa.Table, kind: str, input_column: str, **kwargs: Any) -> pa.Table:
    """1:many multimodal decode — one input row fans out to many output rows
    (video → frames, document → pages). Returns a NEW table (row count changes),
    reusing the tested PyAV / pypdf decoders.
    """
    if kind == "video":
        size = kwargs.get("size")
        return decode_video_batch(
            table,
            video_column=input_column,
            out_column=kwargs.get("out_column", "frame"),
            size=tuple(size) if size else None,
            max_frames=int(kwargs.get("max_frames", 8)),
            stride=int(kwargs.get("stride", 1)),
        )
    if kind == "document":
        max_pages = kwargs.get("max_pages")
        return decode_document_batch(
            table,
            document_column=input_column,
            out_column=kwargs.get("out_column", "text"),
            max_pages=int(max_pages) if max_pages else None,
        )
    raise ValueError(f"explode: unsupported kind {kind!r} (expected 'video' or 'document')")
