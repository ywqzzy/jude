"""L0.1: real S3 write+read round-trip against an in-process S3 (moto server),
which exercises the exact same s3:// code path as a local MinIO. Skips cleanly
if s3fs/moto aren't installed."""

from __future__ import annotations

import pyarrow as pa
import pytest

s3fs = pytest.importorskip("s3fs")
moto_server = pytest.importorskip("moto.server")


@pytest.fixture()
def s3_endpoint():
    """A local S3 endpoint (moto) — stand-in for MinIO."""
    s3fs.S3FileSystem.clear_instance_cache()          # avoid cross-test cached fs
    server = moto_server.ThreadedMotoServer(port=0)
    server.start()
    host, port = server.get_host_and_port()
    endpoint = f"http://{host}:{port}"
    fs = s3fs.S3FileSystem(key="test", secret="test",
                           client_kwargs={"endpoint_url": endpoint})
    try:
        fs.mkdir("judebucket")
    except FileExistsError:
        pass
    yield endpoint
    server.stop()
    s3fs.S3FileSystem.clear_instance_cache()


def _opts(endpoint):
    # fsspec storage_options for s3fs -> the same you'd pass for MinIO
    return {"key": "test", "secret": "test", "client_kwargs": {"endpoint_url": endpoint}}


def test_write_parquet_to_s3(s3_endpoint):
    from jude import storage

    t = pa.table({"id": [1, 2, 3], "text": ["a", "b", "c"]})
    storage.write_parquet(t, "s3://judebucket/data/x.parquet", **_opts(s3_endpoint))
    assert storage.exists("s3://judebucket/data/x.parquet", **_opts(s3_endpoint))
    back = storage.read_arrow("s3://judebucket/data/x.parquet", fmt="parquet",
                             **_opts(s3_endpoint))
    assert back.num_rows == 3 and back.column("text").to_pylist() == ["a", "b", "c"]


def test_write_bytes_and_glob_s3(s3_endpoint):
    from jude import storage

    for i in range(3):
        storage.write_bytes(f"s3://judebucket/blobs/f{i}.bin", b"x" * i, **_opts(s3_endpoint))
    got = storage.glob("s3://judebucket/blobs/*.bin", **_opts(s3_endpoint))
    assert len(got) == 3


def test_pipeline_write_streaming_to_s3(s3_endpoint):
    from jude.datasource import GeneratorSource
    from jude.pipeline._multimodal import RelationPipeline

    src = GeneratorSource(schema=pa.schema([("x", pa.int64())]),
                          task_fns=[(lambda: (yield pa.record_batch({"x": [1, 2, 3]})))])
    p = RelationPipeline.from_datasource(src).map_batches(lambda t: t)
    manifest = p.write_streaming("s3://judebucket/out", fmt="parquet", **_opts(s3_endpoint))
    assert manifest["rows"] == 3                       # streamed straight to S3
