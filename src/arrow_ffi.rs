//! Zero-copy Arrow interchange between Python and DuckDB via the Arrow C Data /
//! C Stream Interface.
//!
//! Ingest (DuckDB / Python -> Rust): pyarrow exports a `RecordBatchReader` into an
//! `FFI_ArrowArrayStream` through its `_export_to_c` capsule pointer, which we wrap
//! with `arrow`'s `ArrowArrayStreamReader`.
//!
//! Egress (Rust -> Python): we build an `FFI_ArrowArrayStream` over our batches and
//! hand its pointer to `pyarrow.RecordBatchReader._import_from_c`.
//!
//! This avoids both the old `/tmp` parquet round-trip and per-value Python loops,
//! and needs no `arrow-pyarrow` crate (which would pin an incompatible pyo3).

use crate::connection::quote_ident;
use crate::error::Error;
use duckdb::arrow::datatypes::SchemaRef;
use duckdb::arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use duckdb::arrow::record_batch::{RecordBatch, RecordBatchIterator, RecordBatchReader};
use duckdb::vtab::arrow_recordbatch_to_query_params;
use duckdb::Connection as DuckConnection;
use pyo3::prelude::*;

/// Coerce a pandas DataFrame / polars (Lazy)Frame into a pyarrow Table; leave
/// objects that are already Arrow-like (Table / RecordBatch / reader) untouched.
fn coerce_to_arrow<'py>(
    _py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    pyarrow: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    // Already Arrow-like: a reader, Table, or RecordBatch.
    if (obj.hasattr("read_all")? && obj.hasattr("schema")?)
        || obj.hasattr("combine_chunks")?
        || obj.hasattr("to_batches")?
    {
        return Ok(obj.clone());
    }
    let module = obj
        .get_type()
        .getattr("__module__")
        .ok()
        .and_then(|m| m.extract::<String>().ok())
        .unwrap_or_default();
    let tyname = obj
        .get_type()
        .getattr("__name__")
        .ok()
        .and_then(|m| m.extract::<String>().ok())
        .unwrap_or_default();
    if module.starts_with("pandas") && tyname == "DataFrame" {
        return pyarrow
            .getattr("Table")?
            .call_method1("from_pandas", (obj,));
    }
    if module.starts_with("polars") {
        // polars DataFrame / LazyFrame
        let df = if tyname == "LazyFrame" {
            obj.call_method0("collect")?
        } else {
            obj.clone()
        };
        return df.call_method0("to_arrow");
    }
    // Not a recognized DataFrame — return as-is and let the export path try.
    Ok(obj.clone())
}

/// Convert a Python Arrow-like object (pyarrow Table, RecordBatch, RecordBatchReader,
/// or anything exposing `__arrow_c_stream__`) into a `Vec<RecordBatch>` zero-copy.
pub fn py_arrow_to_batches(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Vec<RecordBatch>> {
    // Normalize to a pyarrow RecordBatchReader so we have a single export path.
    let pyarrow = py.import("pyarrow")?;
    let reader_cls = pyarrow.getattr("RecordBatchReader")?;

    // Coerce non-Arrow inputs (pandas / polars) to a pyarrow Table first, so
    // callers can register/ingest a DataFrame directly (DuckDB semantics).
    let obj: Bound<'_, PyAny> = coerce_to_arrow(py, obj, &pyarrow)?;
    let obj = &obj;

    let reader_obj: Bound<'_, PyAny> = if obj.hasattr("read_all")? && obj.hasattr("schema")? {
        // Already a RecordBatchReader.
        obj.clone()
    } else {
        // Normalize anything else (Table / RecordBatch / dict-like) to a Table,
        // then combine_chunks() so the exported buffers are contiguous and
        // aligned — the arrow-rs C-stream importer panics on unaligned buffers
        // (which concat_tables / slice can produce).
        let table: Bound<'_, PyAny> = if obj.hasattr("combine_chunks")? {
            obj.clone()
        } else if obj.hasattr("to_batches")? {
            obj.clone()
        } else {
            // A lone RecordBatch -> one-batch Table.
            let schema = obj.getattr("schema")?;
            let batches = pyo3::types::PyList::new(py, [obj])?;
            reader_cls.call_method1("from_batches", (schema, batches))?
        };
        let table = if table.hasattr("combine_chunks")? {
            table.call_method0("combine_chunks")?
        } else {
            table
        };
        table.call_method0("to_reader")?
    };

    // Export the reader into an FFI stream via a capsule pointer.
    let mut stream = FFI_ArrowArrayStream::empty();
    let stream_ptr = &mut stream as *mut FFI_ArrowArrayStream as usize;
    reader_obj.call_method1("_export_to_c", (stream_ptr,))?;

    let reader = ArrowArrayStreamReader::try_new(stream)
        .map_err(|e| Error::Other(format!("failed to import Arrow stream: {e}")))?;
    let schema = reader.schema();
    let mut out = Vec::new();
    for batch in reader {
        out.push(batch.map_err(Error::Arrow)?);
    }
    // A 0-row Arrow object yields no batches through the C stream; emit one empty
    // batch carrying the schema so downstream registration keeps the columns
    // (otherwise an empty table would round-trip to a schemaless dummy).
    if out.is_empty() {
        out.push(RecordBatch::new_empty(schema));
    }
    Ok(out)
}

/// Export a single owned `RecordBatch` to a Python `pyarrow.RecordBatch` via the
/// C Stream Interface. The batch is owned (its Arrow arrays are `Send`), so —
/// unlike the live DuckDB stream — it can go through `FFI_ArrowArrayStream`.
pub fn batch_to_pyarrow(
    py: Python<'_>,
    batch: &RecordBatch,
    schema: &SchemaRef,
) -> PyResult<Py<PyAny>> {
    let pyarrow = py.import("pyarrow")?;
    let reader_cls = pyarrow.getattr("RecordBatchReader")?;
    let owned: Vec<Result<RecordBatch, duckdb::arrow::error::ArrowError>> = vec![Ok(batch.clone())];
    let iter = RecordBatchIterator::new(owned.into_iter(), schema.clone());
    let stream = Box::new(FFI_ArrowArrayStream::new(Box::new(iter)));
    let stream_ptr = Box::into_raw(stream) as usize;
    let reader = reader_cls.call_method1("_import_from_c", (stream_ptr,))?;
    let batch = reader.call_method0("read_next_batch")?;
    Ok(batch.into())
}

/// Convert a slice of RecordBatches into a Python `pyarrow.Table` zero-copy via
/// the C Stream Interface.
pub fn batches_to_pyarrow_table(
    py: Python<'_>,
    batches: &[RecordBatch],
    schema: &SchemaRef,
) -> PyResult<Py<PyAny>> {
    let pyarrow = py.import("pyarrow")?;
    let reader_cls = pyarrow.getattr("RecordBatchReader")?;

    let owned: Vec<Result<RecordBatch, duckdb::arrow::error::ArrowError>> =
        batches.iter().cloned().map(Ok).collect();
    let iter = RecordBatchIterator::new(owned.into_iter(), schema.clone());
    let stream = FFI_ArrowArrayStream::new(Box::new(iter));
    let stream = Box::new(stream);
    let stream_ptr = Box::into_raw(stream) as usize;

    // pyarrow takes ownership of the stream via the pointer.
    let reader = reader_cls.call_method1("_import_from_c", (stream_ptr,))?;
    let table = reader.call_method0("read_all")?;
    Ok(table.into())
}

/// Materialize `batches` into DuckDB under a persistent view/table named `view_name`.
pub fn register_batches_as_view(
    conn: &DuckConnection,
    view_name: &str,
    batches: &[RecordBatch],
) -> Result<(), Error> {
    let ident = quote_ident(view_name);
    // Drop any existing object of either type independently — DuckDB's
    // `DROP VIEW IF EXISTS x` still errors if `x` exists as a TABLE (and vice
    // versa), so we can't batch them; swallow the type-mismatch error.
    let _ = conn.execute_batch(&format!("DROP TABLE IF EXISTS {ident};"));
    let _ = conn.execute_batch(&format!("DROP VIEW IF EXISTS {ident};"));
    // Register into the connection-local TEMP schema (DuckDB `register`
    // semantics): this keeps replacement-scan / registered objects out of the
    // durable catalog, so a later `CREATE TABLE <same name>` in the main schema
    // does not collide, and unqualified lookups still resolve to the temp object.
    ingest_batches(conn, &ident, batches, /*temp=*/ true)
}

/// Materialize `batches` into a fresh TEMP table and return its name, giving a
/// map_batches-produced (SQL-less) relation a SQL identity for further chaining.
pub fn batches_to_temp_table(
    conn: &DuckConnection,
    batches: &[RecordBatch],
) -> Result<String, Error> {
    let name = format!("_jude_mat_{}", uuid::Uuid::new_v4().simple());
    ingest_batches(conn, &name, batches, /*temp=*/ true)?;
    Ok(name)
}

fn ingest_batches(
    conn: &DuckConnection,
    ident: &str,
    batches: &[RecordBatch],
    temp: bool,
) -> Result<(), Error> {
    let create_kw = if temp {
        "CREATE TEMP TABLE"
    } else {
        "CREATE TABLE"
    };
    if batches.is_empty() {
        conn.execute_batch(&format!("{create_kw} {ident} AS SELECT 1 WHERE FALSE"))
            .map_err(Error::DuckDb)?;
        return Ok(());
    }
    let mut created = false;
    for batch in batches {
        let params = arrow_recordbatch_to_query_params(batch.clone());
        let sql = if !created {
            created = true;
            format!("{create_kw} {ident} AS SELECT * FROM arrow(?, ?)")
        } else {
            format!("INSERT INTO {ident} SELECT * FROM arrow(?, ?)")
        };
        let mut stmt = conn.prepare(&sql).map_err(Error::DuckDb)?;
        stmt.execute(duckdb::params![params[0] as u64, params[1] as u64])
            .map_err(Error::DuckDb)?;
    }
    Ok(())
}
