"""jude.multimodal.decoders — decode encoded multimodal bytes into tensors/text.

These are jude UDF-style *batch ops*: each is a callable ``(pa.Table) -> pa.Table``
following the same batch-in / batch-out contract as ``Relation.map_batches`` and
``jude.pipeline`` stages. They turn the binary columns produced by
``jude.sources`` (Image / Audio / Video / Document bytes) into queryable columns:

- ``decode_image_batch``   image bytes  -> fixed_shape_tensor (H,W,C) or list<uint8>
- ``decode_audio_batch``   audio bytes  -> samples list<float32> + sample_rate
- ``decode_video_batch``   video bytes  -> one row per frame (frame tensor)   [1:many]
- ``decode_document_batch`` doc bytes   -> one row per page (text)            [1:many]

Decoding uses Python codec libraries (Pillow / soundfile / PyAV / pypdf); this is
I/O + codec work, not the orchestration hot loop, so Python here is expected.
The 1:1 decoders are row-preserving (they append columns to the input table); the
1:many decoders (video/document) emit a fresh table keyed by the source path so
the result still joins back to the origin relation.
"""

from __future__ import annotations

import io
from typing import Any, Sequence

import pyarrow as pa

from jude.types import tensor_array

__all__ = [
    "decode_image_batch",
    "decode_audio_batch",
    "decode_video_batch",
    "decode_document_batch",
]


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------


def decode_image_batch(
    table: pa.Table,
    *,
    image_column: str = "image",
    out_column: str = "tensor",
    size: Sequence[int] | None = None,
    mode: str = "RGB",
) -> pa.Table:
    """Decode an encoded-image binary column into a tensor column.

    When ``size=(H, W)`` is given (or every image already shares a shape) the
    output is an Arrow ``fixed_shape_tensor`` ``uint8`` column of shape
    ``(H, W, C)`` — the shape multimodal batch-inference wants. When images vary
    in size and no ``size`` is given, the output is a var-length ``list<uint8>``
    column plus a ``{out_column}_shape`` column so the row is still reconstructable.

    Row-preserving: the input columns are kept and the tensor column is appended.
    Also appends ``height`` / ``width`` / ``channels`` columns.
    """
    import numpy as np

    try:
        from PIL import Image as _PILImage
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("decode_image_batch requires Pillow (pip install pillow)") from e

    target = tuple(int(x) for x in size) if size is not None else None
    arrays: list[np.ndarray] = []
    heights: list[int] = []
    widths: list[int] = []
    channels: list[int] = []
    for blob in table.column(image_column).to_pylist():
        if blob is None:
            raise ValueError("decode_image_batch: null image bytes")
        im = _PILImage.open(io.BytesIO(bytes(blob)))
        if mode:
            im = im.convert(mode)
        if target is not None:
            # PIL resize takes (width, height); target is (H, W).
            im = im.resize((target[1], target[0]))
        arr = np.asarray(im)
        if arr.ndim == 2:  # grayscale -> add channel axis
            arr = arr[:, :, None]
        arrays.append(arr)
        heights.append(int(arr.shape[0]))
        widths.append(int(arr.shape[1]))
        channels.append(int(arr.shape[2]))

    out = table
    out = out.append_column("height", pa.array(heights, type=pa.int32()))
    out = out.append_column("width", pa.array(widths, type=pa.int32()))
    out = out.append_column("channels", pa.array(channels, type=pa.int32()))

    uniform = len({a.shape for a in arrays}) == 1 if arrays else True
    if arrays and (target is not None or uniform):
        shape = arrays[0].shape
        stacked = np.stack(arrays).astype("uint8")
        tcol = tensor_array(stacked, dtype="uint8", shape=list(shape))
        out = out.append_column(out_column, tcol)
    else:
        # variable-shape images: flattened list<uint8> + a shape column
        flat = pa.array([a.reshape(-1).astype("uint8").tolist() for a in arrays], type=pa.list_(pa.uint8()))
        shapes = pa.array([list(a.shape) for a in arrays], type=pa.list_(pa.int32()))
        out = out.append_column(out_column, flat)
        out = out.append_column(f"{out_column}_shape", shapes)
    return out


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


def decode_audio_batch(
    table: pa.Table,
    *,
    audio_column: str = "audio",
    out_column: str = "samples",
    mono: bool = True,
    target_sample_rate: int | None = None,
) -> pa.Table:
    """Decode an encoded-audio binary column into a samples column.

    Audio clips vary in length, so samples land as a var-length
    ``list<float32>`` column (a jude unshaped tensor). Also appends
    ``sample_rate`` / ``num_frames`` / ``num_channels``. Row-preserving.
    """
    import numpy as np

    try:
        import soundfile as sf
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("decode_audio_batch requires soundfile (pip install soundfile)") from e

    samples_list: list[list[float]] = []
    rates: list[int] = []
    n_frames: list[int] = []
    n_channels: list[int] = []
    for blob in table.column(audio_column).to_pylist():
        if blob is None:
            raise ValueError("decode_audio_batch: null audio bytes")
        data, sr = sf.read(io.BytesIO(bytes(blob)), dtype="float32", always_2d=True)
        ch = int(data.shape[1])
        if mono and ch > 1:
            data = data.mean(axis=1, keepdims=True)
        if target_sample_rate and target_sample_rate != sr:
            data = _resample_linear(data, sr, target_sample_rate)
            sr = target_sample_rate
        flat = np.ascontiguousarray(data.reshape(-1), dtype="float32")
        samples_list.append(flat.tolist())
        rates.append(int(sr))
        n_frames.append(int(data.shape[0]))
        n_channels.append(1 if mono else ch)

    out = table
    out = out.append_column(out_column, pa.array(samples_list, type=pa.list_(pa.float32())))
    out = out.append_column("sample_rate", pa.array(rates, type=pa.int32()))
    out = out.append_column("num_frames", pa.array(n_frames, type=pa.int32()))
    out = out.append_column("num_channels", pa.array(n_channels, type=pa.int32()))
    return out


def _resample_linear(data: Any, sr: int, target: int) -> Any:
    """Cheap linear resample (no scipy dependency). data is (frames, channels)."""
    import numpy as np

    n = data.shape[0]
    if n == 0:
        return data
    new_n = max(1, int(round(n * target / sr)))
    x_old = np.linspace(0.0, 1.0, num=n, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=new_n, endpoint=False)
    out = np.empty((new_n, data.shape[1]), dtype="float32")
    for c in range(data.shape[1]):
        out[:, c] = np.interp(x_new, x_old, data[:, c])
    return out


# ---------------------------------------------------------------------------
# Video (1:many — one output row per sampled frame)
# ---------------------------------------------------------------------------


def decode_video_batch(
    table: pa.Table,
    *,
    video_column: str = "video",
    path_column: str = "path",
    out_column: str = "frame",
    size: Sequence[int] | None = None,
    max_frames: int = 8,
    stride: int = 1,
) -> pa.Table:
    """Decode a video binary column into one row per sampled frame.

    Emits columns: ``{path_column}``, ``frame_index`` and ``{out_column}`` (an RGB
    frame tensor). When ``size=(H, W)`` is given all frames are resized so the
    output is a ``fixed_shape_tensor`` ``uint8`` ``(H, W, 3)`` column; otherwise a
    var-length ``list<uint8>`` column + ``{out_column}_shape`` is produced.
    """
    import numpy as np

    try:
        import av
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("decode_video_batch requires PyAV (pip install av)") from e

    target = tuple(int(x) for x in size) if size is not None else None
    paths: list[Any] = []
    idxs: list[int] = []
    frames: list[np.ndarray] = []

    has_path = path_column in table.column_names
    path_vals = table.column(path_column).to_pylist() if has_path else [None] * table.num_rows
    for path_val, blob in zip(path_vals, table.column(video_column).to_pylist()):
        if blob is None:
            raise ValueError("decode_video_batch: null video bytes")
        container = av.open(io.BytesIO(bytes(blob)))
        try:
            taken = 0
            seen = 0
            for frame in container.decode(video=0):
                if seen % max(1, stride) == 0:
                    arr = frame.to_ndarray(format="rgb24")  # (H, W, 3)
                    if target is not None:
                        arr = _resize_frame(arr, target)
                    frames.append(arr.astype("uint8"))
                    paths.append(path_val)
                    idxs.append(taken)
                    taken += 1
                    if taken >= max_frames:
                        break
                seen += 1
        finally:
            container.close()

    out = {path_column: pa.array(paths), "frame_index": pa.array(idxs, type=pa.int32())}
    tbl = pa.table(out)
    uniform = len({a.shape for a in frames}) == 1 if frames else True
    if frames and (target is not None or uniform):
        shape = frames[0].shape
        stacked = np.stack(frames).astype("uint8")
        tbl = tbl.append_column(out_column, tensor_array(stacked, dtype="uint8", shape=list(shape)))
    else:
        flat = pa.array([a.reshape(-1).astype("uint8").tolist() for a in frames], type=pa.list_(pa.uint8()))
        shapes = pa.array([list(a.shape) for a in frames], type=pa.list_(pa.int32()))
        tbl = tbl.append_column(out_column, flat)
        tbl = tbl.append_column(f"{out_column}_shape", shapes)
    return tbl


def _resize_frame(arr: Any, target: tuple[int, int]) -> Any:
    from PIL import Image as _PILImage

    im = _PILImage.fromarray(arr).resize((target[1], target[0]))
    import numpy as np

    return np.asarray(im)


# ---------------------------------------------------------------------------
# Document (1:many — one output row per page)
# ---------------------------------------------------------------------------


def decode_document_batch(
    table: pa.Table,
    *,
    document_column: str = "document",
    path_column: str = "path",
    out_column: str = "text",
    max_pages: int | None = None,
) -> pa.Table:
    """Decode a document binary column into one row per page.

    PDFs (magic ``%PDF``) are split into pages via pypdf; anything else is
    treated as a UTF-8 text document (a single page). Emits ``{path_column}``,
    ``page_number`` and ``{out_column}`` (the extracted text).
    """
    paths: list[Any] = []
    pages: list[int] = []
    texts: list[str] = []

    has_path = path_column in table.column_names
    path_vals = table.column(path_column).to_pylist() if has_path else [None] * table.num_rows
    for path_val, blob in zip(path_vals, table.column(document_column).to_pylist()):
        if blob is None:
            raise ValueError("decode_document_batch: null document bytes")
        raw = bytes(blob)
        if raw[:4] == b"%PDF":
            for pno, text in _pdf_pages(raw, max_pages):
                paths.append(path_val)
                pages.append(pno)
                texts.append(text)
        else:
            paths.append(path_val)
            pages.append(0)
            texts.append(raw.decode("utf-8", errors="replace"))

    return pa.table(
        {
            path_column: pa.array(paths),
            "page_number": pa.array(pages, type=pa.int32()),
            out_column: pa.array(texts, type=pa.string()),
        }
    )


def _pdf_pages(raw: bytes, max_pages: int | None):
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("decode_document_batch requires pypdf (pip install pypdf)") from e

    reader = PdfReader(io.BytesIO(raw))
    for pno, page in enumerate(reader.pages):
        if max_pages is not None and pno >= max_pages:
            break
        yield pno, page.extract_text() or ""
