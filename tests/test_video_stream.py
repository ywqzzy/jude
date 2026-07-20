"""Streaming video-frame DataSource: bounded-memory frame streaming off disk."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pyarrow as pa
import pytest

import jude

av = pytest.importorskip("av")


def _make_video(path: str, n_frames: int = 12, w: int = 32, h: int = 24, fps: int = 6):
    container = av.open(path, mode="w")
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    rng = np.random.default_rng(0)
    for _ in range(n_frames):
        arr = rng.integers(0, 256, (h, w, 3), dtype="uint8")
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


@pytest.fixture(scope="module")
def clip():
    d = tempfile.mkdtemp(prefix="jude_vid_")
    p = os.path.join(d, "clip.mp4")
    _make_video(p, n_frames=12)
    yield p


def test_stream_frames_local(clip):
    from jude.sources.video_stream import VideoFrameSource

    src = VideoFrameSource(clip, size=(24, 32), fps_stride=1, chunk_bytes=1)
    # chunk_bytes=1 forces one chunk per frame -> proves streaming granularity
    batches = list(jude.datasource.read_stream(src))
    assert len(batches) >= 2  # multiple chunks, not one giant table
    tbl = pa.Table.from_batches(batches)
    assert "frame" in tbl.column_names
    assert "frame_index" in tbl.column_names
    assert tbl.num_rows >= 8  # ~all frames (mpeg4 may drop a couple)


def test_stream_frames_stride(clip):
    from jude.sources.video_stream import VideoFrameSource

    src = VideoFrameSource(clip, size=(24, 32), fps_stride=3, chunk_bytes=1 << 20)
    rel = jude.datasource.read(src)
    n = rel.num_rows
    # every 3rd frame of ~12 -> ~4
    assert 2 <= n <= 6


def test_stream_frames_max_frames(clip):
    from jude.sources.video_stream import VideoFrameSource

    src = VideoFrameSource(clip, size=(24, 32), fps_stride=1, max_frames=3)
    rel = jude.datasource.read(src)
    assert rel.num_rows == 3


def test_distributed_video_read(clip):
    ray = pytest.importorskip("ray")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=4)
    from jude.sources.video_stream import VideoFrameSource

    src = VideoFrameSource([clip, clip], size=(24, 32), fps_stride=2)  # 2 files -> 2 tasks
    rel = jude.datasource.read(src, distributed=True)
    # two copies of the same clip
    assert rel.num_rows >= 4
    assert "frame" in rel.to_arrow().column_names
