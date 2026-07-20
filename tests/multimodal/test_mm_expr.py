"""Multimodal expressions (jude.mm) — decode / resize / encode as query columns."""

import os
import tempfile

import pytest

import jude

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


@pytest.fixture
def image_dir():
    d = tempfile.mkdtemp()
    # deterministic, distinct sizes + colors
    specs = [(8, 4, (255, 0, 0)), (6, 6, (0, 128, 0)), (10, 2, (0, 0, 255))]
    for i, (w, h, color) in enumerate(specs):
        Image.new("RGB", (w, h), color).save(os.path.join(d, f"img{i}.png"))
    return d


def _rel(image_dir):
    from jude.sources import ImageFileSource

    con = jude.connect()
    rel = ImageFileSource(os.path.join(image_dir, "*.png")).to_relation(con)
    return con, rel


class TestMultimodalExpressions:
    def test_decode_exposes_dimensions_to_sql(self, image_dir):
        con, rel = _rel(image_dir)
        dec = rel.with_column("img", jude.mm("image").image.decode())
        con.register("decoded", dec.to_arrow())
        dims = con.sql("SELECT img.width AS w, img.height AS h FROM decoded ORDER BY w").fetchall()
        assert dims == [(6, 6), (8, 4), (10, 2)]
        # channels + pixel-count are queryable too
        px = con.sql("SELECT img.width * img.height * img.channels AS n FROM decoded ORDER BY n").fetchall()
        assert px == [(20 * 3,), (32 * 3,), (36 * 3,)]

    def test_resize(self, image_dir):
        con, rel = _rel(image_dir)
        rz = rel.with_column("t", jude.mm("image").image.decode().image.resize(4, 4))
        con.register("rz", rz.to_arrow())
        assert con.sql("SELECT DISTINCT t.width, t.height FROM rz").fetchall() == [(4, 4)]

    def test_crop(self, image_dir):
        con, rel = _rel(image_dir)
        cr = rel.with_column("t", jude.mm("image").image.decode().image.crop(0, 0, 3, 2))
        con.register("cr", cr.to_arrow())
        assert con.sql("SELECT DISTINCT t.width, t.height FROM cr").fetchall() == [(3, 2)]

    def test_to_mode_grayscale(self, image_dir):
        con, rel = _rel(image_dir)
        g = rel.with_column("t", jude.mm("image").image.decode().image.to_mode("L"))
        con.register("g", g.to_arrow())
        assert con.sql("SELECT DISTINCT t.channels FROM g").fetchall() == [(1,)]

    def test_encode_round_trip(self, image_dir):
        con, rel = _rel(image_dir)
        enc = rel.with_column("png2", jude.mm("image").image.decode().image.resize(5, 5).image.encode("PNG"))
        red = enc.with_column("back", jude.mm("png2").image.decode())
        con.register("rt", red.to_arrow())
        assert con.sql("SELECT DISTINCT back.width, back.height FROM rt").fetchall() == [(5, 5)]

    def test_aggregate_over_decoded(self, image_dir):
        con, rel = _rel(image_dir)
        dec = rel.with_column("img", jude.mm("image").image.decode())
        con.register("d", dec.to_arrow())
        # multimodal column composes with a normal SQL aggregate
        total = con.sql("SELECT sum(img.width) AS w, count(*) AS n FROM d").fetchone()
        assert total == (8 + 6 + 10, 3)

    def test_chain_is_immutable(self):
        base = jude.mm("image")
        a = base.image.decode()
        b = a.image.resize(2, 2)
        # building b must not mutate a
        assert len(a.ops) == 1
        assert len(b.ops) == 2
        assert base.ops == []

    def test_url_download_then_decode(self, image_dir):
        import pyarrow as pa

        paths = sorted(
            os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")
        )
        con = jude.connect()
        con.register("urls", pa.table({"u": paths}))
        rel = con.sql("SELECT u FROM urls")
        # relation of file paths -> download bytes -> decode -> queryable dims
        out = rel.with_column("img", jude.mm("u").url.download().image.decode())
        con.register("o", out.to_arrow())
        dims = con.sql("SELECT img.width AS w, img.height AS h FROM o ORDER BY w").fetchall()
        assert dims == [(6, 6), (8, 4), (10, 2)]


class TestAudioExpressions:
    def test_audio_decode(self):
        sf = pytest.importorskip("soundfile")
        np = pytest.importorskip("numpy")
        d = tempfile.mkdtemp()
        for i, sr in enumerate([8000, 16000]):
            sig = (0.1 * np.sin(np.arange(sr) * 0.1)).astype("float32")  # 1 second
            sf.write(os.path.join(d, f"a{i}.wav"), sig, sr)
        from jude.sources import AudioFileSource

        con = jude.connect()
        rel = AudioFileSource(os.path.join(d, "*.wav")).to_relation(con)
        dec = rel.with_column("a", jude.mm("audio").audio.decode())
        con.register("d", dec.to_arrow())
        rows = con.sql(
            "SELECT a.sample_rate AS sr, a.num_frames AS n, a.num_channels AS c FROM d ORDER BY sr"
        ).fetchall()
        assert rows == [(8000, 8000, 1), (16000, 16000, 1)]


class TestExplodeExpressions:
    def test_document_explode_pages(self):
        pytest.importorskip("pypdf")  # decoder import guard (text path needs no pypdf, but keep parity)
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "a.txt"), "w") as f:
            f.write("hello world")
        with open(os.path.join(d, "b.txt"), "w") as f:
            f.write("page one")
        from jude.sources import DocumentSource

        con = jude.connect()
        rel = DocumentSource(os.path.join(d, "*.txt")).to_relation(con)
        # 1:many — each text doc is a single page here
        pages = rel.explode_multimodal("document", "document")
        con.register("p", pages.to_arrow())
        rows = con.sql("SELECT page_number, text FROM p ORDER BY text").fetchall()
        assert rows == [(0, "hello world"), (0, "page one")]
