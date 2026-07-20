"""Tests for jude.types — the multimodal type system (tensors, images, etc.)."""

import numpy as np
import pyarrow as pa
import pytest

import jude
from jude.types import (
    Audio,
    Document,
    Image,
    Tensor,
    TensorType,
    Video,
    arrow_type_of,
    image_array,
    tensor_array,
    tensor_to_numpy,
)


class TestTensorType:
    def test_embedding_roundtrip(self):
        embs = np.random.rand(4, 768).astype("float32")
        arr = tensor_array(embs, dtype="float32", shape=[768])
        assert isinstance(arr.type, pa.FixedShapeTensorType)
        back = tensor_to_numpy(arr)
        assert back.shape == (4, 768)
        assert np.allclose(back, embs)

    def test_image_tensor_hwc(self):
        imgs = np.random.randint(0, 255, (2, 8, 8, 3)).astype("uint8")
        arr = tensor_array(imgs, dtype="uint8", shape=[8, 8, 3])
        back = tensor_to_numpy(arr)
        assert back.shape == (2, 8, 8, 3)
        assert np.array_equal(back, imgs)

    def test_tensor_type_arrow_mapping(self):
        t = Tensor("float32", [768])
        assert isinstance(t, TensorType)
        assert isinstance(t.arrow_type, pa.FixedShapeTensorType)
        assert list(t.arrow_type.shape) == [768]

    def test_store_through_duckdb_recovers_shape(self):
        # A tensor degrades to fixed_size_list through DuckDB SQL; shape is
        # recoverable from the declared jude type.
        embs = np.random.rand(3, 128).astype("float32")
        arr = tensor_array(embs, dtype="float32", shape=[128])
        con = jude.connect()
        con.register("e", pa.table({"id": [0, 1, 2], "emb": arr}))
        r = con.sql("SELECT * FROM e ORDER BY id").to_arrow()
        # degraded storage type
        assert pa.types.is_fixed_size_list(r.schema.field("emb").type)
        recovered = tensor_to_numpy(r.column("emb"), shape=[128])
        assert recovered.shape == (3, 128)
        assert np.allclose(recovered, embs)


class TestMultimodalTypes:
    def test_binary_backed_types(self):
        assert Image.arrow_type == pa.binary()
        assert Audio.arrow_type == pa.binary()
        assert Video.arrow_type == pa.binary()
        assert Document.arrow_type == pa.binary()

    def test_image_array_from_bytes(self):
        arr = image_array([b"\x89PNG...", b"\xff\xd8\xff..."])
        assert arr.type == pa.binary()
        assert arr.to_pylist()[0] == b"\x89PNG..."

    def test_arrow_type_of(self):
        assert arrow_type_of(Tensor("float64", [10])) == pa.fixed_shape_tensor(pa.float64(), [10])
        assert arrow_type_of(pa.int32()) == pa.int32()
        with pytest.raises(TypeError):
            arrow_type_of("not a type")


class TestTensorInPipeline:
    def test_map_batches_produces_embedding_tensor(self):
        # A UDF that turns each row into a fixed-shape embedding tensor column —
        # the multimodal batch-inference shape.
        con = jude.connect()
        con.execute("CREATE TABLE t AS SELECT * FROM range(10) t(n)")

        def embed(tbl):
            import numpy as np

            from jude.types import tensor_array

            ns = tbl["n"].to_pylist()
            vecs = np.stack([np.full(4, v, dtype="float32") for v in ns])
            return tbl.append_column("emb", tensor_array(vecs, dtype="float32", shape=[4]))

        out = con.sql("SELECT * FROM t").map_batches(embed, batch_size=4)
        assert out.num_rows == 10
        assert "emb" in out.columns
        tbl = out.to_arrow()
        rec = tensor_to_numpy(tbl.column("emb"), shape=[4])
        assert rec.shape == (10, 4)
        # row n has embedding [n,n,n,n]
        assert np.allclose(rec[5], [5, 5, 5, 5])
