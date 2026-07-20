"""jude.sources.video_stream — streaming video-frame DataSource.

Gap 2b: jude's ``decode_video_batch`` decodes an in-memory video-bytes column
(all sampled frames materialized at once). For large videos / many files that
doesn't scale. This source *streams* frames off disk as fixed-shape tensor
batches with a soft byte budget per emitted chunk, so peak memory is O(one chunk)
regardless of total video size — the scalable frame reader Vane has
(``video_reader.VideoFrameSource``) but built on jude's pluggable
``jude.datasource`` API, so it fans out across Ray workers unchanged.

    from jude.sources.video_stream import VideoFrameSource
    import jude

    src = VideoFrameSource("/data/clips/*.mp4", size=(640, 640), fps_stride=5,
                           chunk_bytes=128 * 2**20)
    rel = jude.datasource.read(src, distributed=True)   # one task per video file
    # columns: path, frame_index, frame  (fixed_shape_tensor uint8 (H,W,3))
"""

from __future__ import annotations

from typing import Iterator

import pyarrow as pa

from jude.datasource import DataSource, DataSourceTask
from jude.sources import _resolve_paths, _VIDEO_EXTS

__all__ = ["VideoFrameSource", "VideoFileTask"]


class VideoFileTask(DataSourceTask):
    """Stream frames from ONE video file as bounded tensor chunks."""

    def __init__(self, path: str, size, fps_stride: int, chunk_bytes: int, max_frames: int | None):
        self.path = path
        self.size = tuple(size) if size else None
        self.fps_stride = max(1, int(fps_stride))
        self.chunk_bytes = int(chunk_bytes)
        self.max_frames = max_frames

    def execute(self) -> Iterator[pa.RecordBatch]:
        import numpy as np

        try:
            import av
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("VideoFrameSource requires PyAV (pip install av)") from e

        from jude.multimodal.decoders import _resize_frame
        from jude.types import tensor_array

        container = av.open(self.path)
        buf_frames: list = []
        buf_idx: list = []
        buf_bytes = 0
        taken = 0
        seen = 0

        def flush():
            nonlocal buf_frames, buf_idx, buf_bytes
            if not buf_frames:
                return None
            stacked = np.stack(buf_frames).astype("uint8")
            shape = list(buf_frames[0].shape)
            tbl = pa.table(
                {
                    "path": pa.array([self.path] * len(buf_frames), type=pa.string()),
                    "frame_index": pa.array(buf_idx, type=pa.int32()),
                }
            )
            tbl = tbl.append_column("frame", tensor_array(stacked, dtype="uint8", shape=shape))
            buf_frames, buf_idx, buf_bytes = [], [], 0
            return tbl

        try:
            for frame in container.decode(video=0):
                if seen % self.fps_stride == 0:
                    arr = frame.to_ndarray(format="rgb24")
                    if self.size is not None:
                        arr = _resize_frame(arr, self.size)
                    arr = arr.astype("uint8")
                    buf_frames.append(arr)
                    buf_idx.append(taken)
                    buf_bytes += arr.nbytes
                    taken += 1
                    if buf_bytes >= self.chunk_bytes:
                        out = flush()
                        if out is not None:
                            yield out
                    if self.max_frames is not None and taken >= self.max_frames:
                        break
                seen += 1
        finally:
            container.close()
        out = flush()
        if out is not None:
            yield out


class VideoFrameSource(DataSource):
    """A streaming DataSource over video files (one task per file).

    ``size=(H, W)`` fixes the output tensor shape (recommended for stacking).
    ``fps_stride`` samples every Nth decoded frame; ``chunk_bytes`` bounds the
    per-emitted-chunk memory (default 128 MiB); ``max_frames`` optionally caps
    frames per video.
    """

    def __init__(
        self,
        inputs,
        *,
        size=None,
        fps_stride: int = 1,
        chunk_bytes: int = 128 * 2**20,
        max_frames: int | None = None,
    ):
        self.paths = _resolve_paths(inputs, _VIDEO_EXTS)
        self.size = tuple(size) if size else None
        self.fps_stride = fps_stride
        self.chunk_bytes = chunk_bytes
        self.max_frames = max_frames

    def schema(self) -> pa.Schema:
        # Frame column is a fixed_shape_tensor when size is set; declare a
        # permissive schema (the tasks produce the concrete tensor type, and
        # datasource normalization aligns by column name).
        return pa.schema(
            [
                pa.field("path", pa.string()),
                pa.field("frame_index", pa.int32()),
            ]
        )

    def tasks(self):
        return [
            VideoFileTask(p, self.size, self.fps_stride, self.chunk_bytes, self.max_frames)
            for p in self.paths
        ]
