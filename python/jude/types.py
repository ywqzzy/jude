"""jude.types — multimodal type system.

Vane extends DuckDB with a native ``TensorType`` and treats images / audio /
video / documents as first-class multimodal columns. jude replicates this
*without forking DuckDB* by layering on Arrow:

- **Tensor / embedding** columns are Arrow ``fixed_shape_tensor`` (dtype + shape)
  while flowing between UDF workers and pipeline stages; they degrade to
  ``fixed_size_list`` when stored through DuckDB SQL (shape recoverable from the
  declared jude type).
- **Image / Audio / Video / Document** columns are Arrow ``binary`` (the encoded
  bytes) tagged with a jude logical type, plus decode helpers that turn bytes
  into numpy arrays / PIL images / tensors inside a UDF or pipeline stage.

This is the data-plane type system used by ``map_batches``, ``jude.pipeline``
stages, and the AI functions — the multimodal surface Vane exposes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pyarrow as pa

__all__ = [
    "JudeType",
    "TensorType",
    "ImageType",
    "AudioType",
    "VideoType",
    "DocumentType",
    "Tensor",
    "Image",
    "Audio",
    "Video",
    "tensor_array",
    "tensor_to_numpy",
    "image_array",
    "decode_image",
    "decode_audio",
    "arrow_type_of",
]


# ---------------------------------------------------------------------------
# Logical types
# ---------------------------------------------------------------------------


class JudeType:
    """Base for jude multimodal logical types."""

    #: the Arrow storage type this logical type maps to
    arrow_type: pa.DataType


@dataclass(frozen=True)
class TensorType(JudeType):
    """A fixed-shape tensor (e.g. an embedding [768] or an image [H,W,3])."""

    dtype: str = "float32"
    shape: tuple[int, ...] = ()

    @property
    def arrow_type(self) -> pa.DataType:
        pa_dtype = _pa_scalar(self.dtype)
        if self.shape:
            return pa.fixed_shape_tensor(pa_dtype, list(self.shape))
        # unshaped / variable: a var-length list
        return pa.list_(pa_dtype)

    def __repr__(self) -> str:
        return f"TensorType({self.dtype}, shape={self.shape})"


@dataclass(frozen=True)
class ImageType(JudeType):
    """An encoded image column (PNG/JPEG/... bytes)."""

    arrow_type: pa.DataType = pa.binary()

    def __repr__(self) -> str:
        return "ImageType()"


@dataclass(frozen=True)
class AudioType(JudeType):
    arrow_type: pa.DataType = pa.binary()

    def __repr__(self) -> str:
        return "AudioType()"


@dataclass(frozen=True)
class VideoType(JudeType):
    arrow_type: pa.DataType = pa.binary()

    def __repr__(self) -> str:
        return "VideoType()"


@dataclass(frozen=True)
class DocumentType(JudeType):
    """An encoded document (PDF/... bytes)."""

    arrow_type: pa.DataType = pa.binary()

    def __repr__(self) -> str:
        return "DocumentType()"


# Convenience singletons
Image = ImageType()
Audio = AudioType()
Video = VideoType()
Document = DocumentType()


def Tensor(dtype: str = "float32", shape: Sequence[int] = ()) -> TensorType:
    return TensorType(dtype=dtype, shape=tuple(shape))


def _pa_scalar(dtype: str) -> pa.DataType:
    return {
        "float32": pa.float32(),
        "float64": pa.float64(),
        "float16": pa.float16(),
        "int8": pa.int8(),
        "int16": pa.int16(),
        "int32": pa.int32(),
        "int64": pa.int64(),
        "uint8": pa.uint8(),
        "bool": pa.bool_(),
    }.get(dtype, pa.float32())


def arrow_type_of(t: Any) -> pa.DataType:
    """Resolve a jude type (or a pyarrow type) to its Arrow storage type."""
    if isinstance(t, JudeType):
        return t.arrow_type
    if isinstance(t, pa.DataType):
        return t
    raise TypeError(f"not a jude/arrow type: {t!r}")


# ---------------------------------------------------------------------------
# Tensor <-> Arrow helpers
# ---------------------------------------------------------------------------


def tensor_array(values: Any, dtype: str = "float32", shape: Sequence[int] | None = None) -> pa.Array:
    """Build an Arrow fixed_shape_tensor array from a numpy array or nested list.

    ``values`` may be an (N, *shape) numpy array or a list of arrays. When
    ``shape`` is omitted it is inferred from the first element.
    """
    import numpy as np

    arr = np.asarray(values, dtype=np.dtype(dtype if dtype != "float16" else "float16"))
    if arr.ndim < 2:
        raise ValueError("tensor_array expects a batch dimension: shape (N, *tensor_shape)")
    per_shape = tuple(shape) if shape is not None else tuple(arr.shape[1:])
    n = arr.shape[0]
    flat = arr.reshape(n, -1)
    tt = pa.fixed_shape_tensor(_pa_scalar(dtype), list(per_shape))
    storage = pa.array(list(flat), type=pa.list_(_pa_scalar(dtype), flat.shape[1]))
    return pa.ExtensionArray.from_storage(tt, storage)


def tensor_to_numpy(array: pa.Array, shape: Sequence[int] | None = None) -> Any:
    """Convert a tensor / fixed_size_list Arrow array back to an (N, *shape)
    numpy array. ``shape`` is required if the array has degraded to a plain
    fixed_size_list (e.g. after a DuckDB round-trip)."""
    import numpy as np

    # A Table column is a ChunkedArray; combine to a single Array first.
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    if isinstance(array.type, pa.FixedShapeTensorType):
        return array.to_numpy_ndarray()
    # fixed_size_list or list: stack the rows
    rows = array.to_pylist()
    out = np.array(rows)
    if shape is not None:
        out = out.reshape((len(rows), *shape))
    return out


# ---------------------------------------------------------------------------
# Image / Audio helpers (decode encoded bytes inside UDFs / stages)
# ---------------------------------------------------------------------------


def image_array(images: Sequence[Any]) -> pa.Array:
    """Build a binary image column from a list of encoded-bytes / PIL images."""
    encoded = []
    for im in images:
        if isinstance(im, (bytes, bytearray)):
            encoded.append(bytes(im))
        elif hasattr(im, "save"):  # PIL.Image
            import io

            buf = io.BytesIO()
            im.save(buf, format="PNG")
            encoded.append(buf.getvalue())
        else:
            raise TypeError(f"cannot encode image of type {type(im)!r}")
    return pa.array(encoded, type=pa.binary())


def decode_image(data: bytes) -> Any:
    """Decode encoded image bytes -> numpy HxWxC array (needs Pillow)."""
    import io

    import numpy as np

    try:
        from PIL import Image as _PILImage
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("decode_image requires Pillow (pip install pillow)") from e
    return np.asarray(_PILImage.open(io.BytesIO(bytes(data))).convert("RGB"))


def decode_audio(data: bytes) -> tuple[Any, int]:
    """Decode encoded audio bytes -> (samples numpy array, sample_rate)
    (needs soundfile)."""
    import io

    try:
        import soundfile as sf
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("decode_audio requires soundfile (pip install soundfile)") from e
    samples, sr = sf.read(io.BytesIO(bytes(data)))
    return samples, sr
