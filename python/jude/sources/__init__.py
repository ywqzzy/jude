"""jude.sources — multimodal DataSources.

Ingest files (local paths / globs / directories) into a jude relation whose
columns are jude multimodal types. This mirrors Daft's ``from_glob_path`` +
``url.download()`` shape: a source lists file paths and reads their bytes into a
binary column tagged with a jude logical type (Image / Audio / Video /
Document). The resulting Arrow table / jude Relation is queryable with the normal
relation & SQL API, and the bytes column feeds the ``jude.multimodal`` decoders.

    import jude
    from jude.sources import ImageFileSource

    con = jude.connect()
    rel = ImageFileSource("/data/imgs/*.png").to_relation(con)
    rel.filter("size_bytes > 1000").project("path").show()

Sources are also usable as pipeline sources — see ``jude.pipeline.relation_to_samples``
and ``SourceStage``.
"""

from __future__ import annotations

import glob as _glob
import os
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from jude.types import AudioType, DocumentType, ImageType, JudeType, VideoType

__all__ = [
    "FileSource",
    "ImageFileSource",
    "AudioFileSource",
    "VideoFrameSource",
    "DocumentSource",
    "list_files",
]


# Default extension filters per modality (lower-cased, includes the dot).
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"}
_AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
_DOC_EXTS = {".pdf", ".txt", ".md", ".text"}


def _resolve_paths(inputs: Any, exts: set[str] | None) -> list[str]:
    """Expand a glob string / dir / path / iterable into a sorted file list."""
    if isinstance(inputs, (str, os.PathLike)):
        inputs = [os.fspath(inputs)]
    resolved: list[str] = []
    for item in inputs:
        item = os.fspath(item)
        if any(ch in item for ch in "*?[]"):
            resolved.extend(_glob.glob(item, recursive=True))
        elif os.path.isdir(item):
            for root, _dirs, files in os.walk(item):
                for f in files:
                    resolved.append(os.path.join(root, f))
        else:
            resolved.append(item)
    # de-dup, filter to files, apply extension filter, sort for determinism
    seen: set[str] = set()
    out: list[str] = []
    for p in sorted(resolved):
        if p in seen or not os.path.isfile(p):
            continue
        seen.add(p)
        if exts is not None and os.path.splitext(p)[1].lower() not in exts:
            continue
        out.append(p)
    return out


def list_files(inputs: Any, *, exts: set[str] | None = None) -> pa.Table:
    """List files matching ``inputs`` as a table of ``path`` + ``size_bytes``.

    Daft's ``from_glob_path`` analogue — metadata only, no bytes read.
    """
    paths = _resolve_paths(inputs, exts)
    sizes = [os.path.getsize(p) for p in paths]
    return pa.table(
        {
            "path": pa.array(paths, type=pa.string()),
            "size_bytes": pa.array(sizes, type=pa.int64()),
        }
    )


@dataclass
class FileSource:
    """Base multimodal file source.

    Resolves ``inputs`` (a glob string, directory, path, or iterable of those)
    into files and reads their bytes into a binary column named ``column`` with
    logical type ``jude_type``. Subclasses set the modality defaults.
    """

    inputs: Any
    column: str = "data"
    jude_type: JudeType | None = None
    exts: set[str] | None = None
    limit: int | None = None
    #: extra scalar columns to attach (name -> constant / callable(path))
    extra_columns: dict[str, Any] = field(default_factory=dict)

    def paths(self) -> list[str]:
        paths = _resolve_paths(self.inputs, self.exts)
        if self.limit is not None:
            paths = paths[: self.limit]
        return paths

    def _read_bytes(self, path: str) -> bytes:
        with open(path, "rb") as fh:
            return fh.read()

    def to_arrow(self, *, read_bytes: bool = True) -> pa.Table:
        """Build an Arrow table: ``path`` + ``size_bytes`` (+ bytes column).

        With ``read_bytes=False`` only metadata is produced (Daft
        ``from_glob_path`` shape); the bytes column is added by a later download
        step. With ``read_bytes=True`` (default) the encoded bytes are read into
        ``self.column`` as a binary column.
        """
        paths = self.paths()
        cols: dict[str, pa.Array] = {
            "path": pa.array(paths, type=pa.string()),
            "size_bytes": pa.array([os.path.getsize(p) for p in paths], type=pa.int64()),
        }
        if read_bytes:
            blobs = [self._read_bytes(p) for p in paths]
            cols[self.column] = pa.array(blobs, type=pa.binary())
        for name, spec in self.extra_columns.items():
            if callable(spec):
                cols[name] = pa.array([spec(p) for p in paths])
            else:
                cols[name] = pa.array([spec] * len(paths))
        return pa.table(cols)

    def to_relation(self, con: Any = None, *, read_bytes: bool = True) -> Any:
        """Materialize this source as a queryable jude Relation.

        A fresh in-memory connection is created if ``con`` is None.
        """
        if con is None:
            import jude

            con = jude.connect()
        return con.from_arrow(self.to_arrow(read_bytes=read_bytes))

    def arrow_schema(self, *, read_bytes: bool = True) -> pa.Schema:
        """The Arrow schema this source produces (bytes column typed binary)."""
        fields = [pa.field("path", pa.string()), pa.field("size_bytes", pa.int64())]
        if read_bytes:
            fields.append(pa.field(self.column, pa.binary()))
        return pa.schema(fields)


@dataclass
class ImageFileSource(FileSource):
    """Glob/list of image files -> relation with ``path`` + ``image`` binary col."""

    column: str = "image"
    jude_type: JudeType | None = field(default_factory=ImageType)
    exts: set[str] | None = field(default_factory=lambda: set(_IMAGE_EXTS))


@dataclass
class AudioFileSource(FileSource):
    """Glob/list of audio files -> relation with ``path`` + ``audio`` binary col."""

    column: str = "audio"
    jude_type: JudeType | None = field(default_factory=AudioType)
    exts: set[str] | None = field(default_factory=lambda: set(_AUDIO_EXTS))


@dataclass
class VideoFrameSource(FileSource):
    """Glob/list of video files -> relation with ``path`` + ``video`` binary col.

    Decoding to frames is a separate op (``jude.multimodal.decode_video_batch`` /
    ``DecodeStage``); this source just ingests the encoded video bytes so decode
    can run as its own independently-scaled pipeline stage.
    """

    column: str = "video"
    jude_type: JudeType | None = field(default_factory=VideoType)
    exts: set[str] | None = field(default_factory=lambda: set(_VIDEO_EXTS))


@dataclass
class DocumentSource(FileSource):
    """Glob/list of PDF/text files -> relation with ``path`` + ``document`` binary col."""

    column: str = "document"
    jude_type: JudeType | None = field(default_factory=DocumentType)
    exts: set[str] | None = field(default_factory=lambda: set(_DOC_EXTS))
