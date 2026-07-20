"""jude.execution — out-of-process UDF worker.

A persistent subprocess that runs a user UDF against Arrow batches. The Rust
subprocess pool spawns one of these per worker; each worker unpickles the UDF
*once* (so stateful ``vane.cls`` actors keep state across batches) and then
loops:

    read one framed Arrow IPC stream  (a batch, or a control message)
    -> run the UDF                    (map_batches / flat_map / map)
    -> write one framed Arrow IPC stream back

Framing on stdin/stdout is a single little-endian u32 length prefix followed by
that many bytes of Arrow IPC stream (or, for control, a length of 0xFFFFFFFF and
a JSON control payload). Keeping data as Arrow IPC over a pipe avoids per-value
Python conversion; the GIL only matters inside this process, so N workers give
N-way real parallelism.
"""

from __future__ import annotations

import io
import json
import struct
import sys
from typing import Any, Callable

import pyarrow as pa

try:
    import cloudpickle
except ImportError:  # pragma: no cover - cloudpickle is a hard dep for UDFs
    cloudpickle = None

_CTRL = 0xFFFFFFFF
_LEN = struct.Struct("<I")


def _read_exact(stream: io.BufferedReader, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError
        buf.extend(chunk)
    return bytes(buf)


def _read_frame(stream: io.BufferedReader) -> tuple[int, bytes]:
    header = _read_exact(stream, 4)
    (length,) = _LEN.unpack(header)
    if length == _CTRL:
        clen = _LEN.unpack(_read_exact(stream, 4))[0]
        return _CTRL, _read_exact(stream, clen)
    return length, _read_exact(stream, length)


def _write_frame(stream: io.BufferedWriter, payload: bytes) -> None:
    stream.write(_LEN.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def _write_ctrl(stream: io.BufferedWriter, obj: dict) -> None:
    data = json.dumps(obj).encode("utf-8")
    stream.write(_LEN.pack(_CTRL))
    stream.write(_LEN.pack(len(data)))
    stream.write(data)
    stream.flush()


def _table_from_ipc(data: bytes) -> pa.Table:
    reader = pa.ipc.open_stream(pa.BufferReader(data))
    return reader.read_all()


def _table_to_ipc(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _to_table(result: Any) -> pa.Table:
    if isinstance(result, pa.Table):
        return result
    if isinstance(result, pa.RecordBatch):
        return pa.Table.from_batches([result])
    if isinstance(result, dict):
        return pa.table(result)
    # Iterator of tables/batches (flat_map style).
    if hasattr(result, "__iter__"):
        parts = [_to_table(r) for r in result]
        if parts:
            return pa.concat_tables(parts)
        return pa.table({})
    raise TypeError(f"UDF returned unsupported type {type(result)!r}")


def _load_callable(payload: dict) -> Callable[[pa.Table], Any]:
    if cloudpickle is None:
        raise RuntimeError("cloudpickle is required for out-of-process UDFs")
    fn = cloudpickle.loads(bytes.fromhex(payload["fn_hex"]))
    is_class = payload.get("is_class", False)
    if is_class:
        # Actor: instantiate once so per-batch state persists.
        instance = fn() if isinstance(fn, type) else fn
        return instance
    return fn


def _apply(fn: Callable, table: pa.Table, call_mode: str) -> pa.Table:
    if call_mode in ("map_batches", "map_batches_rows", "flat_map"):
        return _to_table(fn(table))
    if call_mode == "map":
        # Scalar per-row over the first column; return a single-column table.
        col = table.column(0).to_pylist()
        out = [fn(v) for v in col]
        return pa.table({"result": pa.array(out)})
    raise ValueError(f"unknown call_mode {call_mode!r}")


def main() -> None:
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    # First frame is the control message with the pickled UDF + config.
    kind, payload = _read_frame(stdin)
    if kind != _CTRL:
        raise RuntimeError("expected control init frame first")
    config = json.loads(payload.decode("utf-8"))
    fn = _load_callable(config)
    call_mode = config.get("call_mode", "map_batches")
    _write_ctrl(stdout, {"status": "ready"})

    while True:
        try:
            kind, data = _read_frame(stdin)
        except EOFError:
            break
        if kind == _CTRL:
            ctrl = json.loads(data.decode("utf-8"))
            if ctrl.get("cmd") == "shutdown":
                break
            continue
        table = _table_from_ipc(data)
        try:
            result = _apply(fn, table, call_mode)
            _write_frame(stdout, _table_to_ipc(result))
        except Exception as exc:  # report error, keep worker alive
            _write_ctrl(stdout, {"status": "error", "message": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
