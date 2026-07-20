"""Multimodal ingestion + decode + pipeline tests.

Synthetic fixtures (tiny PNG / WAV / PDF / MP4 / text) are generated into a tmp
dir — no binaries are committed. Covers:

- jude.sources DataSources (Image/Audio/Video/Document + generic FileSource)
- jude.multimodal decoders (image/audio -> tensors, video/document -> 1:many)
- jude.pipeline.RelationPipeline: relation/source in -> multi-stage pipeline ->
  queryable jude relation out (local engine always; cosmos engine when installed)
"""

from __future__ import annotations

import io
import os

import numpy as np
import pyarrow as pa
import pytest

import jude
from jude.multimodal import (
    decode_audio_batch,
    decode_document_batch,
    decode_image_batch,
    decode_video_batch,
)
from jude.pipeline import RelationPipeline
from jude.sources import (
    AudioFileSource,
    DocumentSource,
    ImageFileSource,
    VideoFrameSource,
    list_files,
)
from jude.types import tensor_to_numpy


# ---------------------------------------------------------------------------
# Fixtures: generate tiny synthetic multimodal files in a tmp dir.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def media_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("media")
    _make_images(d)
    _make_audio(d)
    _make_pdf(d)
    _make_text(d)
    _make_video(d)
    return d


def _make_images(d, n=3, h=6, w=8):
    from PIL import Image

    rng = np.random.default_rng(0)
    for i in range(n):
        arr = rng.integers(0, 255, (h, w, 3), dtype="uint8")
        Image.fromarray(arr, "RGB").save(d / f"img_{i}.png")


def _make_audio(d, n=2, sr=16000, frames=1600):
    import soundfile as sf

    rng = np.random.default_rng(1)
    for i in range(n):
        sf.write(d / f"clip_{i}.wav", rng.standard_normal(frames).astype("float32"), sr, format="WAV")


def _make_pdf(d):
    from pypdf import PdfWriter

    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    w.add_blank_page(width=200, height=200)
    with open(d / "doc_0.pdf", "wb") as fh:
        w.write(fh)


def _make_text(d):
    (d / "note_0.txt").write_text("hello jude multimodal\nsecond line", encoding="utf-8")


def _make_video(d, frames=6, size=16):
    import av

    rng = np.random.default_rng(2)
    container = av.open(str(d / "vid_0.mp4"), mode="w", format="mp4")
    stream = container.add_stream("mpeg4", rate=5)
    stream.width = size
    stream.height = size
    stream.pix_fmt = "yuv420p"
    for _ in range(frames):
        arr = rng.integers(0, 255, (size, size, 3), dtype="uint8")
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


# ---------------------------------------------------------------------------
# DataSources
# ---------------------------------------------------------------------------


class TestDataSources:
    def test_list_files(self, media_dir):
        tbl = list_files(str(media_dir / "*.png"))
        assert tbl.num_rows == 3
        assert tbl.column_names == ["path", "size_bytes"]
        assert all(p.endswith(".png") for p in tbl.column("path").to_pylist())

    def test_image_source_to_relation(self, media_dir):
        con = jude.connect()
        rel = ImageFileSource(str(media_dir / "*.png")).to_relation(con)
        assert rel.num_rows == 3
        assert set(rel.columns) == {"path", "size_bytes", "image"}
        # queryable via the normal relation API
        big = rel.filter("size_bytes > 0").project("path")
        assert big.num_rows == 3
        # bytes column is binary and non-empty
        tbl = rel.to_arrow()
        assert pa.types.is_binary(tbl.schema.field("image").type)
        assert len(tbl.column("image")[0].as_py()) > 0

    def test_image_source_glob_and_dir_and_list(self, media_dir):
        # glob, directory, and explicit list all resolve to the 3 pngs
        by_glob = ImageFileSource(str(media_dir / "*.png")).paths()
        by_dir = ImageFileSource(str(media_dir), exts={".png"}).paths()
        by_list = ImageFileSource([str(media_dir / "img_0.png"), str(media_dir / "img_1.png")]).paths()
        assert len(by_glob) == 3
        assert len(by_dir) == 3
        assert len(by_list) == 2

    def test_audio_source(self, media_dir):
        rel = AudioFileSource(str(media_dir / "*.wav")).to_relation()
        assert rel.num_rows == 2
        assert "audio" in rel.columns

    def test_document_source(self, media_dir):
        rel = DocumentSource(str(media_dir)).to_relation()
        # both the pdf and the txt are picked up
        paths = rel.project("path").fetchall()
        exts = {os.path.splitext(p[0])[1] for p in paths}
        assert ".pdf" in exts and ".txt" in exts

    def test_video_source(self, media_dir):
        rel = VideoFrameSource(str(media_dir / "*.mp4")).to_relation()
        assert rel.num_rows == 1
        assert "video" in rel.columns

    def test_metadata_only_source(self, media_dir):
        # read_bytes=False yields the Daft from_glob_path shape (no bytes column)
        rel = ImageFileSource(str(media_dir / "*.png")).to_relation(read_bytes=False)
        assert set(rel.columns) == {"path", "size_bytes"}

    def test_limit_and_extra_columns(self, media_dir):
        src = ImageFileSource(str(media_dir / "*.png"), limit=2, extra_columns={"label": "cat"})
        tbl = src.to_arrow()
        assert tbl.num_rows == 2
        assert tbl.column("label").to_pylist() == ["cat", "cat"]


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------


class TestDecoders:
    def test_decode_image_uniform_to_tensor(self, media_dir):
        con = jude.connect()
        tbl = ImageFileSource(str(media_dir / "*.png")).to_relation(con).to_arrow()
        out = decode_image_batch(tbl, image_column="image", out_column="pixels")
        assert out.num_rows == 3
        assert {"height", "width", "channels", "pixels"} <= set(out.column_names)
        # all pngs are 6x8x3 -> a fixed_shape_tensor
        assert isinstance(out.schema.field("pixels").type, pa.FixedShapeTensorType)
        arr = tensor_to_numpy(out.column("pixels"))
        assert arr.shape == (3, 6, 8, 3)
        assert arr.dtype == np.uint8

    def test_decode_image_resize(self, media_dir):
        tbl = ImageFileSource(str(media_dir / "*.png")).to_arrow()
        out = decode_image_batch(tbl, image_column="image", out_column="px", size=(4, 4))
        arr = tensor_to_numpy(out.column("px"))
        assert arr.shape == (3, 4, 4, 3)

    def test_decode_audio(self, media_dir):
        tbl = AudioFileSource(str(media_dir / "*.wav")).to_arrow()
        out = decode_audio_batch(tbl, audio_column="audio", out_column="samples")
        assert out.num_rows == 2
        assert {"samples", "sample_rate", "num_frames", "num_channels"} <= set(out.column_names)
        assert out.column("sample_rate").to_pylist() == [16000, 16000]
        assert out.column("num_frames").to_pylist() == [1600, 1600]
        assert pa.types.is_list(out.schema.field("samples").type)

    def test_decode_audio_resample(self, media_dir):
        tbl = AudioFileSource(str(media_dir / "*.wav")).to_arrow()
        out = decode_audio_batch(tbl, audio_column="audio", target_sample_rate=8000)
        assert out.column("sample_rate").to_pylist() == [8000, 8000]
        assert out.column("num_frames").to_pylist() == [800, 800]

    def test_decode_document_pdf_and_text(self, media_dir):
        tbl = DocumentSource(str(media_dir)).to_arrow()
        out = decode_document_batch(tbl, document_column="document")
        assert {"path", "page_number", "text"} <= set(out.column_names)
        # pdf -> 2 pages, txt -> 1 page => 3 rows total
        assert out.num_rows == 3
        # the text file's content is extracted
        texts = out.column("text").to_pylist()
        assert any("hello jude multimodal" in t for t in texts)

    def test_decode_video_frames(self, media_dir):
        tbl = VideoFrameSource(str(media_dir / "*.mp4")).to_arrow()
        out = decode_video_batch(tbl, video_column="video", max_frames=4, size=(8, 8))
        # 1:many — one row per sampled frame (<= max_frames)
        assert out.num_rows >= 1
        assert out.num_rows <= 4
        assert {"path", "frame_index", "frame"} <= set(out.column_names)
        arr = tensor_to_numpy(out.column("frame"))
        assert arr.shape[1:] == (8, 8, 3)


# ---------------------------------------------------------------------------
# Decoders wired through the relation API (map_batches / flat_map)
# ---------------------------------------------------------------------------


def _decode_images_udf(tbl):
    from jude.multimodal import decode_image_batch

    return decode_image_batch(tbl, image_column="image", out_column="pixels", size=(4, 4))


class TestDecodeThroughRelation:
    def test_map_batches_decode(self, media_dir):
        con = jude.connect()
        rel = ImageFileSource(str(media_dir / "*.png")).to_relation(con)
        out = rel.map_batches(_decode_images_udf, batch_size=2)
        assert out.num_rows == 3
        assert "pixels" in out.columns
        # and it's still queryable
        assert out.filter("height = 4").num_rows == 3


# ---------------------------------------------------------------------------
# End-to-end multi-stage pipeline: source in -> stages -> queryable relation out
# ---------------------------------------------------------------------------


def _mean_pixel(tbl):
    from jude.types import tensor_to_numpy

    arr = tensor_to_numpy(tbl.column("pixels")).astype("float32")
    means = arr.reshape(arr.shape[0], -1).mean(axis=1)
    return tbl.append_column("mean_pixel", pa.array(means, type=pa.float32()))


class TestRelationPipeline:
    def test_shard_roundtrip(self, media_dir):
        from jude.pipeline import relation_to_shards, shards_to_table

        rel = ImageFileSource(str(media_dir / "*.png")).to_relation()
        shards = relation_to_shards(rel, rows_per_shard=2)
        assert len(shards) == 2  # 3 rows -> [2, 1]
        back = shards_to_table(shards)
        assert back.num_rows == 3

    def test_end_to_end_local(self, media_dir):
        # DataSource (paths only) -> LOAD stage -> DECODE stage -> TRANSFORM stage
        # -> queryable relation. Three distinct, independently-resourced stages.
        con = jude.connect()
        src = ImageFileSource(str(media_dir / "*.png"))
        pipe = (
            RelationPipeline.from_source(src, read_bytes=False, rows_per_shard=2, engine="local")
            .load_files(out_column="image", cpus=1)
            .decode("image", image_column="image", out_column="pixels", size=(4, 4), cpus=2)
            .map_batches(_mean_pixel, cpus=1)
        )
        rel = pipe.to_relation(con)
        assert rel.num_rows == 3
        assert {"path", "image", "pixels", "mean_pixel"} <= set(rel.columns)
        # queryable via SQL: register + aggregate
        con.register("decoded", rel.to_arrow())
        avg = con.sql("SELECT avg(mean_pixel) AS a, count(*) AS c FROM decoded").fetchone()
        assert avg[1] == 3
        assert 0.0 <= avg[0] <= 255.0

    def test_end_to_end_document_pipeline_1_to_many(self, media_dir):
        # load -> decode(document) is 1:many (pages) and still lands as a relation
        con = jude.connect()
        src = DocumentSource(str(media_dir))
        rel = (
            RelationPipeline.from_source(src, read_bytes=False, engine="local")
            .load_files(out_column="document")
            .decode("document", document_column="document")
            .to_relation(con)
        )
        assert rel.num_rows == 3  # 2 pdf pages + 1 text page
        assert "page_number" in rel.columns

    def test_pipeline_from_relation_source(self, media_dir):
        # source can also be an existing relation (rows fed into the pipeline)
        con = jude.connect()
        base = ImageFileSource(str(media_dir / "*.png")).to_relation(con)
        rel = (
            RelationPipeline.from_relation(base, rows_per_shard=1, engine="local")
            .decode("image", image_column="image", out_column="pixels", size=(2, 2))
            .to_relation(con)
        )
        assert rel.num_rows == 3
        assert "pixels" in rel.columns

    def test_pipeline_records_to_observe(self):
        # A pipeline run shows up on the observability dashboard as a `pipeline`
        # query with one stage per pipeline stage.
        import pyarrow as pa

        from jude import observe

        observe.reset()
        pipe = RelationPipeline.from_table(
            pa.table({"x": [1, 2, 3, 4]}), rows_per_shard=2, engine="local"
        ).map_batches(lambda t: t.append_column("y", pa.array([v * 2 for v in t.column("x").to_pylist()])))
        out = pipe.run()
        assert out.num_rows == 4
        snap = observe.snapshot()
        pipes = [q for q in snap["queries"] if q["kind"] == "pipeline"]
        assert len(pipes) == 1
        assert pipes[0]["status"] == "done"
        assert pipes[0]["rows"] == 4
        assert any(s["name"] == "MapBatchesStage" for s in snap["stages"])
        observe.reset()

    def test_pipeline_from_datasource_streaming(self):
        # from_datasource bridges the streaming jude.datasource API into a pipeline.
        import pyarrow as pa

        from jude import datasource as ds

        schema = pa.schema([("x", pa.int64())])

        def shard():
            yield pa.record_batch({"x": [1, 2]}, schema=schema)
            yield pa.record_batch({"x": [3]}, schema=schema)

        src = ds.GeneratorSource(schema, [shard])
        rel = (
            RelationPipeline.from_datasource(src, engine="local")
            .map_batches(lambda t: t.append_column("y", pa.array([v + 10 for v in t.column("x").to_pylist()])))
            .to_relation()
        )
        assert rel.num_rows == 3
        assert sorted(rel.to_arrow().column("y").to_pylist()) == [11, 12, 13]

    @pytest.mark.slow
    def test_end_to_end_cosmos(self, media_dir):
        # Same pipeline, real cosmos-xenna multi-stage engine (Ray). Slow: spins a
        # local Ray cluster + per-stage actor pools.
        pytest.importorskip("cosmos_xenna")
        import jude.pipeline as jp

        if not jp.is_cosmos_backed():
            pytest.skip("cosmos-xenna not backing jude.pipeline")
        con = jude.connect()
        src = ImageFileSource(str(media_dir / "*.png"))
        pipe = (
            RelationPipeline.from_source(src, read_bytes=False, rows_per_shard=2, engine="cosmos")
            .load_files(out_column="image")
            .decode("image", image_column="image", out_column="pixels", size=(4, 4))
        )
        rel = pipe.to_relation(con)
        assert rel.num_rows == 3
        assert "pixels" in rel.columns
