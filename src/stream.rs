//! Streaming execution — pull a query's results one Arrow `RecordBatch` at a
//! time instead of collecting the whole result into a `Vec` (and then one giant
//! Python table). This bounds jude-side memory and lets consumers (the local
//! runner, `map_batches`) pipeline: process a batch while the next is produced.
//!
//! DuckDB delivers batches lazily via `Statement::step()`, but that borrows the
//! `Connection`. We keep the connection alive through the shared `Arc` and hold
//! the borrowing `Statement` next to it in a small self-referential struct.

use std::sync::Arc;

use duckdb::arrow::datatypes::SchemaRef;
use duckdb::arrow::record_batch::RecordBatch;
use duckdb::{Connection as DuckConnection, Statement};
use pyo3::prelude::*;

use crate::error::Error;

/// A prepared, executed DuckDB statement that yields result batches lazily.
///
/// # Safety
/// `stmt` borrows the `DuckConnection` owned (behind the `Arc`) by this struct.
/// Soundness rests on three facts:
/// 1. The `Arc` keeps that `DuckConnection` allocation alive for as long as the
///    `BatchCursor` lives; moving the `Arc` into the struct does not move the
///    heap allocation it points at, so the borrow stays valid.
/// 2. Fields drop in declaration order, so `stmt` (declared first) is dropped
///    before `conn`; the explicit `Drop` makes that ordering intent-clear.
/// 3. All access happens under the Python GIL (the pyclass is `unsendable`), so
///    there is no concurrent use of the connection.
pub struct BatchCursor {
    stmt: Option<Statement<'static>>,
    schema: SchemaRef,
    conn: Arc<DuckConnection>,
}

impl BatchCursor {
    pub fn new(conn: Arc<DuckConnection>, sql: &str) -> Result<Self, Error> {
        // SAFETY: see the struct doc. We extend the borrow of the heap-stable
        // connection to 'static; the Arc field keeps it alive and `stmt` drops
        // first.
        let conn_ref: &'static DuckConnection =
            unsafe { std::mem::transmute::<&DuckConnection, &'static DuckConnection>(&*conn) };
        let mut stmt = conn_ref.prepare(sql).map_err(Error::DuckDb)?;
        // Execute the query and drop the returned Arrow handle immediately; the
        // result state persists on the statement, and `step()` pulls batches.
        {
            let _ = stmt.query_arrow([]).map_err(Error::DuckDb)?;
        }
        let schema = stmt.schema();
        Ok(Self {
            stmt: Some(stmt),
            schema,
            conn,
        })
    }

    pub fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    /// The next result batch, or `None` when the result is exhausted.
    pub fn next_batch(&self) -> Option<RecordBatch> {
        let stmt = self.stmt.as_ref()?;
        stmt.step().map(|sa| RecordBatch::from(&sa))
    }
}

impl Drop for BatchCursor {
    fn drop(&mut self) {
        // Drop the connection-borrowing statement before the connection Arc.
        self.stmt = None;
        let _ = &self.conn;
    }
}

/// A Python iterator over a query's result batches. Each `next()` yields one
/// `pyarrow.RecordBatch`; iteration ends (StopIteration) when the result is
/// drained. `unsendable` because it holds a live DuckDB statement.
#[pyclass(unsendable)]
pub struct RecordBatchStream {
    cursor: BatchCursor,
    done: bool,
}

impl RecordBatchStream {
    pub fn new(conn: Arc<DuckConnection>, sql: &str) -> Result<Self, Error> {
        Ok(Self {
            cursor: BatchCursor::new(conn, sql)?,
            done: false,
        })
    }
}

#[pymethods]
impl RecordBatchStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        if self.done {
            return Ok(None);
        }
        match self.cursor.next_batch() {
            Some(batch) => Ok(Some(crate::arrow_ffi::batch_to_pyarrow(
                py,
                &batch,
                &self.cursor.schema(),
            )?)),
            None => {
                self.done = true;
                Ok(None)
            }
        }
    }

    /// The pyarrow `Schema` of the stream (available before the first batch).
    #[getter]
    fn schema(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let empty = RecordBatch::new_empty(self.cursor.schema());
        let b = crate::arrow_ffi::batch_to_pyarrow(py, &empty, &self.cursor.schema())?;
        Ok(b.bind(py).getattr("schema")?.into())
    }

    /// Read the remaining batches into a single `pyarrow.Table` (drains the
    /// stream). Mirrors `pyarrow.RecordBatchReader.read_all`.
    fn read_all(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mut batches: Vec<RecordBatch> = Vec::new();
        while let Some(b) = self.cursor.next_batch() {
            batches.push(b);
        }
        self.done = true;
        crate::arrow_ffi::batches_to_pyarrow_table(py, &batches, &self.cursor.schema())
    }
}
