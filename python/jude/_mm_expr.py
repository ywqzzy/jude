"""jude._mm_expr — multimodal expression accessors (Daft-parity surface).

A ``MultimodalExpr`` names an input column and a fused chain of multimodal ops
(image decode / resize / encode / to_tensor). The chain is a list of
``(op_name, kwargs)`` specs — no logic here; the Rust kernel
(``jude.multimodal``, via ``Relation.with_column``) does the work.

    import jude
    from jude.sources import ImageFileSource

    con = jude.connect()
    rel = ImageFileSource("/imgs/*.png").to_relation(con)
    decoded = rel.with_column("img", jude.mm("data").image.decode().image.resize(64, 64))
    decoded.aggregate("avg(img.height) AS h").fetchall()

The decoded image is an Arrow struct ``{height, width, channels, data:list<uint8>}``
so SQL can read the dimensions and aggregate over them.
"""

from __future__ import annotations

from typing import Any


class MultimodalExpr:
    """An immutable builder: an input column plus a fused op chain.

    Each accessor method returns a *new* MultimodalExpr with one more op
    appended, so chains compose without mutating shared state.
    """

    __slots__ = ("input_column", "ops")

    def __init__(self, input_column: str, ops: list[tuple[str, dict]] | None = None):
        self.input_column = input_column
        self.ops: list[tuple[str, dict]] = list(ops or [])

    def _append(self, op_name: str, **kwargs: Any) -> "MultimodalExpr":
        return MultimodalExpr(self.input_column, [*self.ops, (op_name, dict(kwargs))])

    @property
    def image(self) -> "ImageNamespace":
        return ImageNamespace(self)

    @property
    def url(self) -> "UrlNamespace":
        return UrlNamespace(self)

    @property
    def audio(self) -> "AudioNamespace":
        return AudioNamespace(self)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        chain = " -> ".join(name for name, _ in self.ops) or "(none)"
        return f"MultimodalExpr(column={self.input_column!r}, ops={chain})"


class AudioNamespace:
    """The ``.audio`` op namespace. Decode runs the soundfile-backed decoder
    (a Python-fallback codec) and yields a queryable struct column
    ``{samples: list<float32>, sample_rate, num_frames, num_channels}``."""

    __slots__ = ("_expr",)

    def __init__(self, expr: MultimodalExpr):
        self._expr = expr

    def decode(self, sample_rate: int | None = None, mono: bool = True) -> MultimodalExpr:
        kw: dict = {"mono": bool(mono)}
        if sample_rate is not None:
            kw["sample_rate"] = int(sample_rate)
        return self._expr._append("audio_decode", **kw)


class UrlNamespace:
    """The ``.url`` op namespace (Daft's ``url.download()``)."""

    __slots__ = ("_expr",)

    def __init__(self, expr: MultimodalExpr):
        self._expr = expr

    def download(self) -> MultimodalExpr:
        """Read bytes from a column of local file paths / ``file://`` URLs."""
        return self._expr._append("url_download")


class ImageNamespace:
    """The ``.image`` op namespace, mirroring Daft's image expressions."""

    __slots__ = ("_expr",)

    def __init__(self, expr: MultimodalExpr):
        self._expr = expr

    def decode(self) -> MultimodalExpr:
        """Decode encoded image bytes (PNG/JPEG/…) into an RGB image struct."""
        return self._expr._append("image_decode")

    def resize(self, width: int, height: int) -> MultimodalExpr:
        """Resize a decoded image to (width, height)."""
        return self._expr._append("image_resize", width=int(width), height=int(height))

    def crop(self, x: int, y: int, width: int, height: int) -> MultimodalExpr:
        """Crop a decoded image to the (x, y, width, height) region."""
        return self._expr._append("image_crop", x=int(x), y=int(y), width=int(width), height=int(height))

    def to_mode(self, mode: str) -> MultimodalExpr:
        """Convert color mode: 'RGB', 'L' (grayscale), or 'RGBA'."""
        return self._expr._append("image_to_mode", mode=str(mode))

    def to_tensor(self) -> MultimodalExpr:
        """Normalize to the tensor representation (the decoded struct)."""
        return self._expr._append("image_to_tensor")

    def encode(self, image_format: str = "PNG") -> MultimodalExpr:
        """Re-encode a decoded image struct back to bytes."""
        return self._expr._append("image_encode", format=str(image_format))


def mm(column: str) -> MultimodalExpr:
    """Start a multimodal expression on a column: ``jude.mm("data").image.decode()``."""
    return MultimodalExpr(column)


__all__ = ["MultimodalExpr", "ImageNamespace", "UrlNamespace", "AudioNamespace", "mm"]
