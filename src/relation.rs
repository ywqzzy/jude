use crate::connection::{quote_ident, quote_qualified_name, Connection};
use crate::error::Error;
use crate::plan::{JoinType, LogicalPlan, SetOpKind};
use duckdb::arrow::array::*;
use duckdb::arrow::datatypes::{DataType, Schema, SchemaRef};
use duckdb::arrow::record_batch::RecordBatch;
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::cell::RefCell;
use std::sync::Arc;

/// A lazy relation: a **logical plan tree** plus a back-reference to its
/// connection.
///
/// Relational-algebra methods build `LogicalPlan` nodes (a real operator DAG),
/// not SQL strings. The plan runs only when the relation is materialized
/// (fetch/num_rows/to_arrow/show/…); local execution lowers the tree to SQL via
/// `LogicalPlan::to_sql`, while the distributed runner walks the tree directly.
#[pyclass(name = "Relation", unsendable)]
pub struct Relation {
    pub conn: Py<Connection>,
    pub plan: Arc<LogicalPlan>,
    /// Cached materialization (batches + schema), lazily computed.
    cache: RefCell<Option<(Vec<RecordBatch>, SchemaRef)>>,
    /// DBAPI cursor position: index of the next unread row across the cached
    /// batches. Advanced by fetchone/fetchmany/fetchall; reset by execute().
    read_pos: RefCell<usize>,
}

impl Relation {
    /// Build a relation from a logical plan.
    pub fn from_plan(py: Python<'_>, conn: &Connection, plan: LogicalPlan) -> PyResult<Self> {
        let py_conn = Py::new(py, Connection::from_arc(conn.inner.clone()))?;
        Ok(Self {
            conn: py_conn,
            plan: Arc::new(plan),
            cache: RefCell::new(None),
            read_pos: RefCell::new(0),
        })
    }

    /// A shallow clone sharing the same connection and plan (fresh cache).
    pub fn clone_ref_relation(&self, py: Python<'_>) -> Self {
        Self {
            conn: self.conn.clone_ref(py),
            plan: self.plan.clone(),
            cache: RefCell::new(None),
            read_pos: RefCell::new(0),
        }
    }

    /// Build a relation from a raw SQL leaf (sql(), read_csv(), values(), …).
    pub fn new_lazy_sql(py: Python<'_>, conn: &Connection, sql: String) -> PyResult<Self> {
        // DuckDB's relational API binds at construction: `con.sql(bad_query)`
        // Preparing the statement binds it (resolving tables/columns) without
        // executing, so we surface those errors eagerly while staying lazy for
        // the actual scan.
        conn.inner
            .prepare(&sql)
            .map_err(|e| PyErr::from(Error::DuckDb(e)))?;
        Self::from_plan(py, conn, LogicalPlan::RawSql { sql })
    }

    /// Build a relation over in-memory Arrow batches (map_batches / from_arrow).
    pub fn new_materialized(
        py: Python<'_>,
        conn: &Connection,
        batches: Vec<RecordBatch>,
    ) -> PyResult<Self> {
        Self::from_plan(
            py,
            conn,
            LogicalPlan::Materialized {
                batches: Arc::new(batches),
            },
        )
    }

    /// Build a derived relation whose plan wraps this relation's plan.
    fn derive_plan(
        &self,
        py: Python<'_>,
        make: impl FnOnce(Arc<LogicalPlan>) -> LogicalPlan,
    ) -> PyResult<Self> {
        let conn_ref = self.conn.borrow(py);
        let py_conn = Py::new(py, Connection::from_arc(conn_ref.inner.clone()))?;
        Ok(Self {
            conn: py_conn,
            plan: Arc::new(make(self.plan.clone())),
            cache: RefCell::new(None),
            read_pos: RefCell::new(0),
        })
    }

    /// Lower this relation's plan to a subquery-embeddable SQL string. Any
    /// Materialized/MapBatches leaves are registered as TEMP tables on demand.
    fn to_subquery_sql(&self, py: Python<'_>) -> PyResult<String> {
        let conn_ref = self.conn.borrow(py);
        let inner = &conn_ref.inner;
        let mut resolve = |node: &LogicalPlan| -> Result<String, Error> {
            match node {
                LogicalPlan::Materialized { batches } => {
                    crate::arrow_ffi::batches_to_temp_table(inner, batches)
                }
                other => Err(Error::Other(format!(
                    "cannot resolve non-materialized leaf {} to a table",
                    other.op_name()
                ))),
            }
        };
        self.plan.to_sql(&mut resolve).map_err(PyErr::from)
    }

    /// Like `to_subquery_sql`, but lowers each Materialized leaf to a
    /// re-scannable `jude_scan(<id>)` table function over the held batches
    /// instead of copying them into a TEMP TABLE. Returns the SQL and the ids to
    /// unregister once the query has run. Used by `materialize` so a boundary's
    /// output streams into downstream SQL without a full DuckDB-side copy.
    fn to_subquery_sql_scanning(&self, _py: Python<'_>) -> PyResult<(String, Vec<u64>)> {
        let mut ids: Vec<u64> = Vec::new();
        let mut resolve = |node: &LogicalPlan| -> Result<String, Error> {
            match node {
                LogicalPlan::Materialized { batches } => {
                    match batches.first() {
                        Some(b) => {
                            let id = crate::mat_scan::register(batches.clone(), b.schema());
                            ids.push(id);
                            Ok(format!("jude_scan({id}::UBIGINT)"))
                        }
                        // Empty (no batches / no schema) — fall back is handled by
                        // returning an empty SELECT with no columns is unsafe, so
                        // signal to the caller to use the temp-table path instead.
                        None => Err(Error::Other("__jude_empty_materialized__".to_string())),
                    }
                }
                other => Err(Error::Other(format!(
                    "cannot resolve non-materialized leaf {} to a table",
                    other.op_name()
                ))),
            }
        };
        let sql = self.plan.to_sql(&mut resolve);
        match sql {
            Ok(s) => Ok((s, ids)),
            Err(e) => {
                // Unregister anything registered before the failure.
                for id in &ids {
                    crate::mat_scan::unregister(*id);
                }
                Err(PyErr::from(e))
            }
        }
    }

    /// Materialize (execute + cache).
    fn materialize(&self, py: Python<'_>) -> PyResult<()> {
        if self.cache.borrow().is_some() {
            return Ok(());
        }
        // Fast path: a Materialized leaf needs no execution.
        if let LogicalPlan::Materialized { batches } = &*self.plan {
            let schema = batches
                .first()
                .map(|b| b.schema())
                .unwrap_or_else(|| Arc::new(Schema::empty()));
            *self.cache.borrow_mut() = Some(((**batches).clone(), schema));
            return Ok(());
        }
        // Prefer the re-scannable jude_scan lowering (no temp-table copy of a
        // boundary's output); fall back to the temp-table path for the degenerate
        // empty-materialized case that carries no schema.
        match self.to_subquery_sql_scanning(py) {
            Ok((sql, ids)) => {
                let conn_ref = self.conn.borrow(py);
                let result = conn_ref.run_sql_with_schema(&sql);
                drop(conn_ref);
                for id in &ids {
                    crate::mat_scan::unregister(*id);
                }
                let (batches, schema) = result.map_err(PyErr::from)?;
                *self.cache.borrow_mut() = Some((batches, schema));
            }
            Err(_) => {
                // Empty-materialized or other non-scannable leaf — temp-table path.
                let sql = self.to_subquery_sql(py)?;
                let conn_ref = self.conn.borrow(py);
                let (batches, schema) = conn_ref.run_sql_with_schema(&sql).map_err(PyErr::from)?;
                *self.cache.borrow_mut() = Some((batches, schema));
            }
        }
        Ok(())
    }

    fn with_batches<R>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&[RecordBatch], &SchemaRef) -> R,
    ) -> PyResult<R> {
        self.materialize(py)?;
        let cache = self.cache.borrow();
        let (batches, schema) = cache.as_ref().unwrap();
        Ok(f(batches, schema))
    }

    /// Fetch the schema without materializing all rows.
    fn probe_schema(&self, py: Python<'_>) -> PyResult<SchemaRef> {
        if let Some((_, schema)) = self.cache.borrow().as_ref() {
            return Ok(schema.clone());
        }
        if let LogicalPlan::Materialized { batches } = &*self.plan {
            return Ok(batches
                .first()
                .map(|b| b.schema())
                .unwrap_or_else(|| Arc::new(Schema::empty())));
        }
        let sql = self.to_subquery_sql(py)?;
        let conn_ref = self.conn.borrow(py);
        conn_ref.schema_of(&sql).map_err(PyErr::from)
    }

    /// Apply an out-of-band batch transform (used by AI functions), producing a
    /// new materialized relation. Kept for compatibility with the AI batch path.
    pub fn map_batches<F>(&self, py: Python<'_>, f: F) -> PyResult<Self>
    where
        F: Fn(&RecordBatch) -> PyResult<RecordBatch>,
    {
        self.materialize(py)?;
        let cache = self.cache.borrow();
        let (batches, _) = cache.as_ref().unwrap();
        let mut out = Vec::with_capacity(batches.len());
        for batch in batches {
            out.push(f(batch)?);
        }
        drop(cache);
        let conn_ref = self.conn.borrow(py);
        Self::new_materialized(py, &conn_ref, out)
    }

    /// Access materialized batches for the local runner.
    pub fn collect_batches(&self, py: Python<'_>) -> PyResult<Vec<RecordBatch>> {
        self.with_batches(py, |b, _| b.to_vec())
    }

    /// Sanitize an aggregate-function argument the way DuckDB's relational API
    /// does: parse it as a single SQL expression; if it parses, pass it through
    /// (so `a + b`, `abs(v)`, `case ...`, `(select 1)`, `v::BOOL`, `*` all work);
    /// if it fails to parse as one expression, quote it as an identifier. This
    /// both supports column names needing quotes (reserved keywords, spaces,
    /// embedded quotes) and neutralizes injections (`v; drop ...`, `v union ...`,
    /// `v) from ...`) — they become an unknown column, i.e. a BinderException.
    fn sanitize_agg_arg(&self, py: Python<'_>, expr: &str) -> String {
        let t = expr.trim();
        // Empty / whitespace-only: leave empty so agg_or_window's guard rejects
        // it as a ParserException (rather than quoting "" into a column ref).
        if t.is_empty() {
            return String::new();
        }
        // Star is only valid unquoted (count(*)) — never quote it.
        if t == "*" || t.ends_with(".*") {
            return expr.to_string();
        }
        // Probe: does `(<expr>)` parse as a single expression? We wrap in parens
        // (so trailing `; drop`, `) from`, `union ...` cause a *parser* error)
        // and append a newline (so a trailing `-- comment` can't eat the close
        // paren). No FROM clause, so column refs raise a *binder* error, which
        // still means the expression itself parsed fine.
        let parses = {
            let conn = self.conn.borrow(py);
            let probe = format!("SELECT ({}\n)", expr);
            match conn.inner.prepare(&probe) {
                Ok(_) => true,
                Err(e) => {
                    let m = e.to_string().to_ascii_lowercase();
                    !(m.contains("parser error") || m.contains("syntax error"))
                }
            }
        };
        if parses {
            // Pass through; keep the trailing newline so a line comment inside
            // the expression can't swallow the aggregate's closing paren.
            format!("{}\n", expr)
        } else {
            // Quote as an identifier (escaping embedded double quotes).
            format!("\"{}\"", expr.replace('"', "\"\""))
        }
    }

    /// Build an aggregate OR a windowed projection, matching Vane's
    /// `(expression, groups="", window_spec="", projected_columns="")` shape.
    ///
    /// - No window_spec: GROUP BY `groups`, SELECT `projected_columns` + the
    ///   aggregate. `projected_columns` (not `groups`) is the passthrough prefix
    ///   — Vane groups by `groups` but projects `projected_columns`.
    /// - With window_spec: `func OVER (window_spec)` projected alongside
    ///   `projected_columns`.
    fn agg_or_window(
        &self,
        py: Python<'_>,
        func_expr: &str,
        alias: &str,
        groups: &str,
        window_spec: &str,
        projected: &str,
    ) -> PyResult<Self> {
        // Reject empty / whitespace-only aggregate arguments (Vane raises
        // ParserException). Detect a call like `FUNC()` or `FUNC( )` — an empty
        // inner argument — and raise the matching jude exception.
        if let Some(open) = func_expr.find('(') {
            if let Some(close) = func_expr.rfind(')') {
                if close > open && func_expr[open + 1..close].trim().is_empty() {
                    return Err(parser_exception(
                        py,
                        "aggregate expression must not be empty",
                    ));
                }
            }
        }
        let inner = self.to_subquery_sql(py)?;
        let sql = if window_spec.is_empty() {
            let agg = format!("{func_expr} AS {alias}");
            let select = if projected.is_empty() {
                agg
            } else {
                format!("{projected}, {agg}")
            };
            // Group by `groups` if given, else by `projected_columns` (Vane
            // semantics: the projected columns are the grouping columns). A
            // `groups` spec may carry a trailing `ORDER BY …` (DuckDB relational
            // syntax) that does not change grouped-aggregate content, so we
            // group by the leading key list only.
            let group_by_raw = if !groups.is_empty() {
                groups
            } else {
                projected
            };
            let group_by = strip_trailing_order_by(group_by_raw);
            if group_by.is_empty() {
                format!("SELECT {select} FROM ({inner}) AS _t")
            } else {
                format!("SELECT {select} FROM ({inner}) AS _t GROUP BY {group_by}")
            }
        } else {
            let win = format!("{func_expr} {} AS {alias}", render_over_clause(window_spec));
            let proj = if projected.is_empty() {
                win
            } else {
                format!("{projected}, {win}")
            };
            format!("SELECT {proj} FROM ({inner}) AS _t")
        };
        self.derive_plan(py, move |_input| LogicalPlan::RawSql { sql })
    }

    /// A pure window function projection: `func OVER (window_spec) AS alias`,
    /// alongside `projected` columns.
    fn window_fn(
        &self,
        py: Python<'_>,
        func_call: &str,
        alias: &str,
        window_spec: &str,
        projected: &str,
    ) -> PyResult<Self> {
        let inner = self.to_subquery_sql(py)?;
        let win = format!("{func_call} {} AS {alias}", render_over_clause(window_spec));
        let proj = if projected.is_empty() {
            win
        } else {
            format!("{projected}, {win}")
        };
        let sql = format!("SELECT {proj} FROM ({inner}) AS _t");
        self.derive_plan(py, move |_input| LogicalPlan::RawSql { sql })
    }

    /// In-process map_batches: call the Python fn once per (optionally rechunked)
    /// batch under the GIL. Simple and correct; used when no out-of-process
    /// backend is requested.
    fn map_batches_inprocess(
        &self,
        py: Python<'_>,
        fn_: &Bound<'_, PyAny>,
        batch_size: Option<usize>,
    ) -> PyResult<Self> {
        // Stream the input batch-by-batch (bounded memory, and the UDF starts on
        // the first batch before the whole input is read) unless the caller
        // asked for a specific batch size, which needs a rebatched table.
        let pa_batches: Vec<Bound<'_, PyAny>> = match batch_size {
            Some(n) if n > 0 => {
                self.materialize(py)?;
                let cache = self.cache.borrow();
                let (batches, sch) = cache.as_ref().unwrap();
                let table = crate::arrow_ffi::batches_to_pyarrow_table(py, batches, sch)?;
                drop(cache);
                table
                    .bind(py)
                    .call_method1("to_batches", (n,))?
                    .try_iter()?
                    .collect::<PyResult<Vec<_>>>()?
            }
            _ => {
                let sql = self.to_subquery_sql(py)?;
                let conn_ref = self.conn.borrow(py);
                let stream = crate::stream::RecordBatchStream::new(conn_ref.inner.clone(), &sql)
                    .map_err(PyErr::from)?;
                drop(conn_ref);
                let stream_obj = Py::new(py, stream)?;
                stream_obj
                    .bind(py)
                    .try_iter()?
                    .collect::<PyResult<Vec<_>>>()?
            }
        };
        let mut out_py = Vec::new();
        for b in pa_batches {
            out_py.push(fn_.call1((b,))?.unbind());
        }
        let out_table = reassemble_table(py, out_py)?;
        let batches = crate::arrow_ffi::py_arrow_to_batches(py, out_table.bind(py))?;
        let conn_ref = self.conn.borrow(py);
        Self::new_materialized(py, &conn_ref, batches)
    }

    /// Out-of-process map_batches: ship the pickled UDF to a pool of worker
    /// subprocesses and dispatch batches with the GIL released. This is the
    /// GIL-free parallel path (N workers = N interpreters running in parallel).
    fn map_batches_subprocess(
        &self,
        py: Python<'_>,
        fn_: &Bound<'_, PyAny>,
        batch_size: Option<usize>,
        max_bytes: Option<usize>,
        num_workers: Option<usize>,
        call_mode: &str,
    ) -> PyResult<Self> {
        // Serialize the UDF via jude.execution.serialize_udf -> control JSON.
        let exec = py.import("jude.execution")?;
        // Treat a class, or an instance/callable carrying jude actor markers, as
        // a stateful actor (instantiated once per worker so state persists).
        let is_class = fn_.is_instance_of::<pyo3::types::PyType>()
            || fn_.hasattr("_jude_is_cls").unwrap_or(false)
            || fn_.hasattr("_jude_is_cls_batch").unwrap_or(false);
        let payload = exec.call_method(
            "serialize_udf",
            (fn_,),
            Some(&{
                let d = pyo3::types::PyDict::new(py);
                d.set_item("call_mode", call_mode)?;
                d.set_item("is_class", is_class)?;
                d
            }),
        )?;
        let init_ctrl: Vec<u8> = py
            .import("json")?
            .call_method1("dumps", (payload,))?
            .extract::<String>()?
            .into_bytes();

        // Determine the interpreter to run workers with.
        let python_exe: String = py.import("sys")?.getattr("executable")?.extract()?;
        let workers = num_workers.unwrap_or_else(default_worker_count);

        // Materialize + optionally rechunk input batches (row and/or byte target).
        self.materialize(py)?;
        let input_batches: Vec<RecordBatch> = {
            let cache = self.cache.borrow();
            let (batches, _) = cache.as_ref().unwrap();
            rechunk_batches_bytes(batches, batch_size, max_bytes)
        };

        // Reuse a cached pool for this UDF (amortizes worker spawn cost), and
        // run with the GIL released so worker pipes overlap and other Python
        // threads keep running.
        let out_batches = py.detach(|| -> Result<Vec<RecordBatch>, Error> {
            let pool = crate::udf::get_or_create_pool(&python_exe, workers, &init_ctrl)?;
            pool.map_batches(&input_batches)
        })?;

        let conn_ref = self.conn.borrow(py);
        Self::new_materialized(py, &conn_ref, out_batches)
    }

    /// Distributed map_batches via the Ray runner: partition the relation and
    /// apply the UDF on Ray actors (each its own interpreter, optional GPU).
    fn map_batches_ray(
        &self,
        py: Python<'_>,
        fn_: &Bound<'_, PyAny>,
        batch_size: Option<usize>,
    ) -> PyResult<Self> {
        // Build the pickled UDF payload (reuse jude.execution.serialize_udf).
        let exec = py.import("jude.execution")?;
        let is_class = fn_.is_instance_of::<pyo3::types::PyType>()
            || fn_.hasattr("_jude_is_cls").unwrap_or(false)
            || fn_.hasattr("_jude_is_cls_batch").unwrap_or(false);
        let payload = exec.call_method(
            "serialize_udf",
            (fn_,),
            Some(&{
                let d = pyo3::types::PyDict::new(py);
                d.set_item("call_mode", "map_batches")?;
                d.set_item("is_class", is_class)?;
                d
            }),
        )?;

        // Get the Ray runner and dispatch: runner.map_relation(self, payload, batch_size).
        let runners = py.import("jude.runners")?;
        let runner = runners.call_method0("get_or_create_runner")?;
        if runner.getattr("name")?.extract::<String>()? != "ray" {
            // Ray unavailable — fall back to the local subprocess pool.
            return self.map_batches_subprocess(py, fn_, batch_size, None, None, "map_batches");
        }
        // Pass self as a Python object; the runner materializes via to_arrow.
        let py_self = self_as_pyobject(py, self)?;
        let out_tables = runner.call_method1("map_relation", (py_self, payload, batch_size))?;
        // out_tables is a list of pyarrow Tables; concat + ingest.
        let pyarrow = py.import("pyarrow")?;
        let merged = pyarrow.call_method1("concat_tables", (out_tables,))?;
        let batches = crate::arrow_ffi::py_arrow_to_batches(py, &merged)?;
        let conn_ref = self.conn.borrow(py);
        Self::new_materialized(py, &conn_ref, batches)
    }

    /// Per-batch Ray execution via jude.execution backends (ray_task / ray_actor),
    /// mirroring Vane's duckdb/execution model. Materializes the input to an
    /// Arrow table, runs the backend, and re-ingests the result.
    fn map_batches_exec(
        &self,
        py: Python<'_>,
        fn_: &Bound<'_, PyAny>,
        batch_size: Option<usize>,
        backend: &str,
        num_workers: Option<usize>,
        num_gpus: Option<f64>,
    ) -> PyResult<Self> {
        let exec = py.import("jude.execution")?;
        let is_class = fn_.is_instance_of::<pyo3::types::PyType>()
            || fn_.hasattr("_jude_is_cls").unwrap_or(false)
            || fn_.hasattr("_jude_is_cls_batch").unwrap_or(false);
        // GPU allocation: explicit arg wins, else read a `jude.cls(gpus=…)` marker.
        let gpus = num_gpus.or_else(|| {
            fn_.getattr("_jude_gpus")
                .ok()
                .and_then(|g| g.extract::<f64>().ok())
        });
        let payload = exec.call_method(
            "serialize_udf",
            (fn_,),
            Some(&{
                let d = pyo3::types::PyDict::new(py);
                d.set_item("call_mode", "map_batches")?;
                d.set_item("is_class", is_class)?;
                d
            }),
        )?;
        // Materialize input to a pyarrow Table.
        let table = self.to_arrow(py)?;
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("execution_backend", backend)?;
        if let Some(bs) = batch_size {
            kwargs.set_item("batch_size", bs)?;
        }
        kwargs.set_item("num_workers", num_workers.unwrap_or(1))?;
        if let Some(g) = gpus {
            kwargs.set_item("num_gpus", g)?;
        }
        let out = exec.call_method("run_ray_map", (payload, table), Some(&kwargs))?;
        let batches = crate::arrow_ffi::py_arrow_to_batches(py, &out)?;
        let conn_ref = self.conn.borrow(py);
        Self::new_materialized(py, &conn_ref, batches)
    }
}

/// Re-wrap a `&Relation` as a Python object so it can be passed to Python-side
/// runners. Clones the connection handle and the current plan.
fn self_as_pyobject(py: Python<'_>, rel: &Relation) -> PyResult<Py<Relation>> {
    let conn_ref = rel.conn.borrow(py);
    let clone = Relation {
        conn: Py::new(py, Connection::from_arc(conn_ref.inner.clone()))?,
        plan: rel.plan.clone(),
        cache: RefCell::new(None),
        read_pos: RefCell::new(0),
    };
    Py::new(py, clone)
}

fn default_worker_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(8)
}

/// Byte-batch target from the environment (VANE_/JUDE_UDF_TARGET_MAX_BATCH_BYTES).
/// None if unset, so byte-batching is opt-in unless configured globally.
fn udf_target_max_batch_bytes_env() -> Option<usize> {
    for key in [
        "JUDE_UDF_TARGET_MAX_BATCH_BYTES",
        "VANE_UDF_TARGET_MAX_BATCH_BYTES",
    ] {
        if let Ok(v) = std::env::var(key) {
            if let Ok(n) = v.trim().parse::<usize>() {
                if n > 0 {
                    return Some(n);
                }
            }
        }
    }
    None
}

/// Rechunk by a row target AND/OR a byte target. A chunk is cut when it reaches
/// either limit — byte-based batching (Vane's VANE_UDF_TARGET_MAX_BATCH_BYTES)
/// keeps GPU/model batches within a memory budget regardless of row width.
fn rechunk_batches_bytes(
    batches: &[RecordBatch],
    batch_size: Option<usize>,
    max_bytes: Option<usize>,
) -> Vec<RecordBatch> {
    if batch_size.filter(|&n| n > 0).is_none() && max_bytes.filter(|&n| n > 0).is_none() {
        return batches.to_vec();
    }
    let row_limit = batch_size.filter(|&n| n > 0);
    let byte_limit = max_bytes.filter(|&n| n > 0);
    let mut out = Vec::new();
    for batch in batches {
        let rows = batch.num_rows();
        if rows == 0 {
            continue;
        }
        // Estimate per-row bytes to translate the byte budget into a row count.
        let batch_bytes = batch.get_array_memory_size().max(1);
        let per_row = (batch_bytes / rows).max(1);
        let effective = match (row_limit, byte_limit) {
            (Some(r), Some(b)) => r.min((b / per_row).max(1)),
            (Some(r), None) => r,
            (None, Some(b)) => (b / per_row).max(1),
            (None, None) => rows,
        };
        let mut start = 0;
        while start < rows {
            let len = effective.min(rows - start);
            out.push(batch.slice(start, len));
            start += len;
        }
    }
    out
}

#[pymethods]
impl Relation {
    fn show(&self, py: Python<'_>) -> PyResult<()> {
        self.with_batches(py, |batches, schema| {
            if batches.iter().map(|b| b.num_rows()).sum::<usize>() == 0 {
                let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
                if names.is_empty() {
                    println!("(empty)");
                } else {
                    println!("{}", names.join(" | "));
                    println!("(0 rows)");
                }
                return;
            }
            for batch in batches {
                print_batch(batch);
            }
        })
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let n = self.probe_schema(py).map(|s| s.fields().len()).unwrap_or(0);
        format!("Relation({}, {n} cols)", self.plan.op_name())
    }

    /// DuckDB-style rendering: column headers + a small preview of rows + a row
    /// footer. Large relations are not fully counted — the footer shows
    /// "? rows" with a ">9999 rows" estimate (matching DuckDB), so printing a
    /// billion-row relation stays cheap.
    fn __str__(&self, py: Python<'_>) -> String {
        const PREVIEW: usize = 10;
        const CAP: usize = 10001; // fetch just past the display estimate threshold
        let schema = match self.probe_schema(py) {
            Ok(s) => s,
            Err(_) => return self.__repr__(py),
        };
        let cols: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
        let types: Vec<String> = schema
            .fields()
            .iter()
            .map(|f| arrow_type_to_duckdb_name(f.data_type()))
            .collect();

        // Bounded fetch: run the plan with a LIMIT so we never count a huge table.
        let (rows, capped): (Vec<Vec<String>>, bool) = (|| {
            let inner = self.to_subquery_sql(py).ok()?;
            let conn = self.conn.borrow(py);
            let sql = format!("SELECT * FROM ({inner}) AS _t LIMIT {CAP}");
            let (batches, _) = conn.run_sql_with_schema(&sql).ok()?;
            let total: usize = batches.iter().map(|b| b.num_rows()).sum();
            let mut out = Vec::new();
            'outer: for b in &batches {
                for r in 0..b.num_rows() {
                    if out.len() >= PREVIEW {
                        break 'outer;
                    }
                    let mut row = Vec::new();
                    for c in 0..b.num_columns() {
                        let col = b.column(c);
                        let v = if col.is_null(r) {
                            "NULL".to_string()
                        } else {
                            array_value_to_py(col, r, py)
                                .ok()
                                .and_then(|o| o.bind(py).str().ok().map(|s| s.to_string()))
                                .unwrap_or_default()
                        };
                        row.push(v);
                    }
                    out.push(row);
                }
            }
            Some((out, total >= CAP))
        })()
        .unwrap_or((Vec::new(), false));

        let mut s = String::new();
        s.push_str(&format!("┌─ Relation ({}) ─\n", self.plan.op_name()));
        s.push_str(&format!("│ {}\n", cols.join(" | ")));
        s.push_str(&format!("│ {}\n", types.join(" | ")));
        s.push_str("├───\n");
        for row in &rows {
            s.push_str(&format!("│ {}\n", row.join(" | ")));
        }
        s.push_str("├───\n");
        if capped {
            // Unknown exact size; DuckDB shows this pair for large relations.
            s.push_str("│ ? rows\n│ >9999 rows\n");
        } else {
            s.push_str(&format!("│ {} rows\n", rows.len().min(PREVIEW).max(0)));
        }
        s.push_str("└───");
        s
    }

    #[getter]
    fn columns(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        let schema = self.probe_schema(py)?;
        Ok(schema.fields().iter().map(|f| f.name().clone()).collect())
    }

    /// The relation's alias (name it can be referenced by). Reflects the most
    /// recent `set_alias`; defaults to DuckDB's "unnamed_relation".
    #[getter]
    fn alias(&self) -> String {
        match &*self.plan {
            LogicalPlan::Alias { name, .. } => name.clone(),
            _ => "unnamed_relation".to_string(),
        }
    }

    /// DuckDB-style relation type name (TABLE_RELATION, PROJECTION_RELATION, …).
    #[getter]
    fn r#type(&self) -> String {
        fn classify(p: &LogicalPlan) -> String {
            match p {
                // Look through an alias to the wrapped relation.
                LogicalPlan::Alias { input, .. } => classify(input),
                LogicalPlan::Table { .. } => "TABLE_RELATION".to_string(),
                LogicalPlan::ScanFunction { .. } => "TABLE_FUNCTION_RELATION".to_string(),
                LogicalPlan::Materialized { .. } => "MATERIALIZED_RELATION".to_string(),
                LogicalPlan::Filter { .. } => "FILTER_RELATION".to_string(),
                LogicalPlan::Project { .. } => "PROJECTION_RELATION".to_string(),
                LogicalPlan::Aggregate { .. } => "AGGREGATE_RELATION".to_string(),
                LogicalPlan::Join { .. } => "JOIN_RELATION".to_string(),
                LogicalPlan::SetOp { .. } => "SET_OPERATION_RELATION".to_string(),
                LogicalPlan::Order { .. } => "ORDER_RELATION".to_string(),
                LogicalPlan::Limit { .. } => "LIMIT_RELATION".to_string(),
                LogicalPlan::Distinct { .. } => "PROJECTION_RELATION".to_string(),
                LogicalPlan::RawSql { sql } => {
                    // A plain single-table scan (`SELECT * FROM <ident>`) is a
                    // TABLE_RELATION in DuckDB; anything else is a query.
                    let s = sql.trim();
                    let low = s.to_ascii_lowercase();
                    let body = low.strip_prefix("select * from ").unwrap_or("");
                    let is_table = !body.is_empty()
                        && !body.contains(' ')
                        && !body.contains('(')
                        && !body.contains(';');
                    if is_table {
                        "TABLE_RELATION".to_string()
                    } else {
                        "QUERY_RELATION".to_string()
                    }
                }
                _ => "QUERY_RELATION".to_string(),
            }
        }
        classify(&self.plan)
    }

    #[getter]
    fn types(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        let schema = self.probe_schema(py)?;
        Ok(schema
            .fields()
            .iter()
            .map(|f| arrow_type_to_duckdb_name(f.data_type()))
            .collect())
    }

    #[getter]
    fn dtypes(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        self.types(py)
    }

    /// Project only the columns whose type is in `types` (DuckDB `select_dtypes`
    /// / `select_types`). `types` is a list of SQL type strings (e.g.
    /// `[VARCHAR]`); matching is case-insensitive on the type name. Column names
    /// are quoted so names with spaces/keywords survive.
    #[pyo3(name = "select_dtypes")]
    fn select_dtypes(&self, py: Python<'_>, types: Vec<String>) -> PyResult<Self> {
        if types.is_empty() {
            return Err(parser_exception(
                py,
                "select_types requires at least one type",
            ));
        }
        let want: Vec<String> = types
            .iter()
            .map(|t| t.trim().to_ascii_uppercase())
            .collect();
        let schema = self.probe_schema(py)?;
        let kept: Vec<String> = schema
            .fields()
            .iter()
            .filter(|f| {
                let ty = arrow_type_to_duckdb_name(f.data_type()).to_ascii_uppercase();
                let base = ty
                    .split(['(', '['])
                    .next()
                    .unwrap_or(&ty)
                    .trim()
                    .to_string();
                want.iter().any(|w| {
                    // Parameterized types (STRUCT(...), DECIMAL(p,s), T[n]) must
                    // match exactly; base types (VARCHAR, BIGINT) match the base.
                    if w.contains('(') || w.contains('[') {
                        *w == ty
                    } else {
                        *w == base
                    }
                })
            })
            .map(|f| quote_ident(f.name()))
            .collect();
        if kept.is_empty() {
            return Err(parser_exception(py, "select_types matched no columns"));
        }
        self.derive_plan(py, move |input| LogicalPlan::Project { input, exprs: kept })
    }

    /// Alias for `select_dtypes` (DuckDB exposes both).
    #[pyo3(name = "select_types")]
    fn select_types(&self, py: Python<'_>, types: Vec<String>) -> PyResult<Self> {
        self.select_dtypes(py, types)
    }

    #[getter]
    fn num_rows(&self, py: Python<'_>) -> PyResult<usize> {
        self.with_batches(py, |batches, _| batches.iter().map(|b| b.num_rows()).sum())
    }

    #[getter]
    fn shape(&self, py: Python<'_>) -> PyResult<(usize, usize)> {
        let rows = self.num_rows(py)?;
        let cols = self.probe_schema(py)?.fields().len();
        Ok((rows, cols))
    }

    // ---- Relational algebra (logical-plan composition) ----

    pub fn filter(&self, py: Python<'_>, condition: &Bound<'_, PyAny>) -> PyResult<Self> {
        let predicate = expr_to_sql(condition)?;
        self.derive_plan(py, |input| LogicalPlan::Filter { input, predicate })
    }

    fn where_(&self, py: Python<'_>, condition: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.filter(py, condition)
    }

    pub fn project(&self, py: Python<'_>, exprs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let items = exprs_to_sql_list(exprs)?;
        self.derive_plan(py, |input| LogicalPlan::Project {
            input,
            exprs: items,
        })
    }

    fn select(&self, py: Python<'_>, columns: &Bound<'_, PyAny>) -> PyResult<Self> {
        // `select` in DuckDB's relational API is column projection; accept either
        // column names, expressions, or a list thereof.
        self.project(py, columns)
    }

    #[pyo3(signature = (aggr_expr, group_expr=""))]
    fn aggregate(&self, py: Python<'_>, aggr_expr: &str, group_expr: &str) -> PyResult<Self> {
        let aggs = vec![aggr_expr.to_string()];
        let group: Vec<String> = if group_expr.is_empty() {
            Vec::new()
        } else {
            group_expr
                .split(',')
                .map(|s| s.trim().to_string())
                .collect()
        };
        self.derive_plan(py, |input| LogicalPlan::Aggregate { input, group, aggs })
    }

    /// Add a column computed by a multimodal expression chain (jude.mm).
    ///
    /// `mm_expr` is a `jude._mm_expr.MultimodalExpr`: it names an input column
    /// and a fused op chain (image.decode / resize / encode / to_tensor). The
    /// chain runs as a Rust kernel over the materialized Arrow batches (a
    /// materialization boundary — see docs/multimodal_design.md), producing a
    /// new relation with the added column, over which ordinary SQL composes.
    fn with_column(
        &self,
        py: Python<'_>,
        name: &str,
        mm_expr: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let input_column: String = mm_expr.getattr("input_column")?.extract()?;
        let ops_obj = mm_expr.getattr("ops")?;

        // Ops whose codec lives only in Python (audio/video/document) route to
        // jude.multimodal._expr rather than the Rust kernel. Detect them by name
        // before parsing into MmOp (which only knows the Rust-native ops).
        let mut op_names: Vec<String> = Vec::new();
        for item in ops_obj.try_iter()? {
            op_names.push(item?.get_item(0)?.extract::<String>()?);
        }
        const FALLBACK_OPS: &[&str] = &["audio_decode"];
        let is_fallback = op_names.iter().any(|n| FALLBACK_OPS.contains(&n.as_str()));

        if is_fallback {
            // Materialize, hand the whole Arrow table to the Python fallback,
            // and re-ingest its result as a materialized relation.
            self.materialize(py)?;
            let cache = self.cache.borrow();
            let (batches, schema) = cache.as_ref().unwrap();
            let table = crate::arrow_ffi::batches_to_pyarrow_table(py, batches, schema)?;
            drop(cache);
            let module = py.import("jude.multimodal._expr")?;
            let out_table = module
                .call_method1("apply_expr", (table.bind(py), &input_column, name, ops_obj))?;
            let out_batches = crate::arrow_ffi::py_arrow_to_batches(py, &out_table)?;
            let conn_ref = self.conn.borrow(py);
            return Self::new_materialized(py, &conn_ref, out_batches);
        }

        // Rust-native op chain (image/url): parse specs into Vec<MmOp>.
        let mut ops: Vec<crate::multimodal::MmOp> = Vec::new();
        for item in ops_obj.try_iter()? {
            let item = item?;
            let op_name: String = item.get_item(0)?.extract()?;
            let kwargs = item.get_item(1)?;
            let get_u32 = |k: &str| -> Option<u32> {
                kwargs
                    .get_item(k)
                    .ok()
                    .and_then(|v| v.extract::<u32>().ok())
            };
            let get_str = |k: &str| -> Option<String> {
                kwargs
                    .get_item(k)
                    .ok()
                    .and_then(|v| v.extract::<String>().ok())
            };
            ops.push(
                crate::multimodal::MmOp::from_spec(&op_name, get_u32, get_str)
                    .map_err(PyErr::from)?,
            );
        }

        self.materialize(py)?;
        let cache = self.cache.borrow();
        let (batches, _) = cache.as_ref().unwrap();
        let mut out_batches = Vec::with_capacity(batches.len());
        for batch in batches {
            let idx = batch.schema().index_of(&input_column).map_err(|_| {
                crate::error::Error::Other(format!("column '{input_column}' not found"))
            })?;
            let input_col = batch.column(idx).clone();
            let out_col = crate::multimodal::apply_chain(&input_col, &ops).map_err(PyErr::from)?;
            // Append the computed column to the batch.
            let mut fields: Vec<arrow::datatypes::Field> = batch
                .schema()
                .fields()
                .iter()
                .map(|f| (**f).clone())
                .collect();
            fields.push(arrow::datatypes::Field::new(
                name,
                out_col.data_type().clone(),
                true,
            ));
            let mut cols: Vec<arrow::array::ArrayRef> = batch.columns().to_vec();
            cols.push(out_col);
            let schema = Arc::new(arrow::datatypes::Schema::new(fields));
            out_batches.push(
                RecordBatch::try_new(schema, cols).map_err(|e| crate::error::Error::Arrow(e))?,
            );
        }
        drop(cache);
        let conn_ref = self.conn.borrow(py);
        Self::new_materialized(py, &conn_ref, out_batches)
    }

    /// 1:many multimodal decode — one input row fans out to many output rows
    /// (video → frames, document → pages). Routes to jude.multimodal._expr.explode
    /// (the tested PyAV / pypdf decoders) and re-ingests the result. Returns a
    /// new relation with a different row count, queryable with ordinary SQL.
    #[pyo3(signature = (kind, input_column, **kwargs))]
    fn explode_multimodal(
        &self,
        py: Python<'_>,
        kind: &str,
        input_column: &str,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Self> {
        self.materialize(py)?;
        let cache = self.cache.borrow();
        let (batches, schema) = cache.as_ref().unwrap();
        let table = crate::arrow_ffi::batches_to_pyarrow_table(py, batches, schema)?;
        drop(cache);
        let module = py.import("jude.multimodal._expr")?;
        let args = (table.bind(py), kind, input_column);
        let out_table = module.call_method("explode", args, kwargs)?;
        let out_batches = crate::arrow_ffi::py_arrow_to_batches(py, &out_table)?;
        let conn_ref = self.conn.borrow(py);
        Self::new_materialized(py, &conn_ref, out_batches)
    }

    #[pyo3(signature = (other, condition, how="inner"))]
    fn join(&self, py: Python<'_>, other: &Self, condition: &str, how: &str) -> PyResult<Self> {
        let right = other.plan.clone();
        let cond = condition.to_string();
        let how = JoinType::parse(how);
        self.derive_plan(py, |left| LogicalPlan::Join {
            left,
            right,
            condition: cond,
            how,
        })
    }

    fn cross(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        self.join(py, other, "", "cross")
    }

    fn union(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        let right = other.plan.clone();
        self.derive_plan(py, |left| LogicalPlan::SetOp {
            op: SetOpKind::UnionAll,
            left,
            right,
        })
    }

    fn union_all(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        self.union(py, other)
    }

    fn distinct_union(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        let right = other.plan.clone();
        self.derive_plan(py, |left| LogicalPlan::SetOp {
            op: SetOpKind::Union,
            left,
            right,
        })
    }

    fn intersect(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        let right = other.plan.clone();
        self.derive_plan(py, |left| LogicalPlan::SetOp {
            op: SetOpKind::Intersect,
            left,
            right,
        })
    }

    fn except_(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        let right = other.plan.clone();
        self.derive_plan(py, |left| LogicalPlan::SetOp {
            op: SetOpKind::Except,
            left,
            right,
        })
    }

    pub fn distinct(&self, py: Python<'_>) -> PyResult<Self> {
        self.derive_plan(py, |input| LogicalPlan::Distinct { input })
    }

    #[pyo3(signature = (n, offset=0))]
    fn limit(&self, py: Python<'_>, n: usize, offset: usize) -> PyResult<Self> {
        self.derive_plan(py, |input| LogicalPlan::Limit { input, n, offset })
    }

    pub fn order(&self, py: Python<'_>, order_expr: &Bound<'_, PyAny>) -> PyResult<Self> {
        let keys = exprs_to_sql_list(order_expr)?;
        self.derive_plan(py, |input| LogicalPlan::Order { input, keys })
    }

    fn sort(&self, py: Python<'_>, columns: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.order(py, columns)
    }

    pub fn set_alias(&self, py: Python<'_>, name: &str) -> PyResult<Self> {
        let name = name.to_string();
        self.derive_plan(py, |input| LogicalPlan::Alias { input, name })
    }

    /// Run an arbitrary SQL query using this relation as a named CTE/view.
    fn query(&self, py: Python<'_>, view_name: &str, sql: &str) -> PyResult<Self> {
        let cte = view_name.to_string();
        let sql = sql.to_string();
        self.derive_plan(py, |input| LogicalPlan::Query { input, cte, sql })
    }

    // ---- Aggregate shortcuts (Vane signature: expr, groups, window_spec, projected_columns) ----

    #[pyo3(signature = (column=None, groups="", window_spec="", projected_columns=""))]
    fn count(
        &self,
        py: Python<'_>,
        column: Option<&str>,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        let arg = column.unwrap_or("*");
        self.agg_or_window(
            py,
            &format!("COUNT({arg})"),
            "count",
            groups,
            window_spec,
            projected_columns,
        )
    }

    #[pyo3(signature = (column, groups="", window_spec="", projected_columns=""))]
    fn sum(
        &self,
        py: Python<'_>,
        column: &str,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        let c = self.sanitize_agg_arg(py, column);
        self.agg_or_window(
            py,
            &format!("SUM({c})"),
            "sum",
            groups,
            window_spec,
            projected_columns,
        )
    }

    #[pyo3(signature = (column, groups="", window_spec="", projected_columns=""))]
    fn avg(
        &self,
        py: Python<'_>,
        column: &str,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        let c = self.sanitize_agg_arg(py, column);
        self.agg_or_window(
            py,
            &format!("AVG({c})"),
            "avg",
            groups,
            window_spec,
            projected_columns,
        )
    }

    #[pyo3(signature = (column, groups="", window_spec="", projected_columns=""))]
    fn mean(
        &self,
        py: Python<'_>,
        column: &str,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.avg(py, column, groups, window_spec, projected_columns)
    }

    #[pyo3(signature = (column, groups="", window_spec="", projected_columns=""))]
    fn min(
        &self,
        py: Python<'_>,
        column: &str,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.agg_or_window(
            py,
            &format!("MIN({column})"),
            "min",
            groups,
            window_spec,
            projected_columns,
        )
    }

    #[pyo3(signature = (column, groups="", window_spec="", projected_columns=""))]
    fn max(
        &self,
        py: Python<'_>,
        column: &str,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.agg_or_window(
            py,
            &format!("MAX({column})"),
            "max",
            groups,
            window_spec,
            projected_columns,
        )
    }

    #[pyo3(signature = (column, groups="", window_spec="", projected_columns=""))]
    fn median(
        &self,
        py: Python<'_>,
        column: &str,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.agg_or_window(
            py,
            &format!("MEDIAN({column})"),
            "median",
            groups,
            window_spec,
            projected_columns,
        )
    }

    #[pyo3(name = "stddev_samp", signature = (column, groups="", window_spec="", projected_columns=""))]
    fn stddev_samp(
        &self,
        py: Python<'_>,
        column: &str,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.agg_or_window(
            py,
            &format!("STDDEV_SAMP({column})"),
            "std",
            groups,
            window_spec,
            projected_columns,
        )
    }

    #[pyo3(name = "stddev", signature = (column, groups="", window_spec="", projected_columns=""))]
    fn stddev(
        &self,
        py: Python<'_>,
        column: &str,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.stddev_samp(py, column, groups, window_spec, projected_columns)
    }

    #[pyo3(name = "std", signature = (column, groups="", window_spec="", projected_columns=""))]
    fn std(
        &self,
        py: Python<'_>,
        column: &str,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.stddev_samp(py, column, groups, window_spec, projected_columns)
    }

    #[pyo3(name = "var_samp", signature = (column, groups="", window_spec="", projected_columns=""))]
    fn var_samp(
        &self,
        py: Python<'_>,
        column: &str,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.agg_or_window(
            py,
            &format!("VAR_SAMP({column})"),
            "variance",
            groups,
            window_spec,
            projected_columns,
        )
    }

    #[pyo3(name = "variance", signature = (column, groups="", window_spec="", projected_columns=""))]
    fn variance(
        &self,
        py: Python<'_>,
        column: &str,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.var_samp(py, column, groups, window_spec, projected_columns)
    }

    #[pyo3(name = "var", signature = (column, groups="", window_spec="", projected_columns=""))]
    fn var(
        &self,
        py: Python<'_>,
        column: &str,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.var_samp(py, column, groups, window_spec, projected_columns)
    }

    #[pyo3(signature = (column, groups="", window_spec="", projected_columns=""))]
    fn favg(
        &self,
        py: Python<'_>,
        column: &str,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.agg_or_window(
            py,
            &format!("FAVG({column})"),
            "favg",
            groups,
            window_spec,
            projected_columns,
        )
    }

    #[pyo3(signature = (column, groups="", window_spec="", projected_columns=""))]
    fn fsum(
        &self,
        py: Python<'_>,
        column: &str,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.agg_or_window(
            py,
            &format!("FSUM({column})"),
            "fsum",
            groups,
            window_spec,
            projected_columns,
        )
    }

    #[pyo3(signature = (column, groups="", projected_columns=""))]
    fn geomean(
        &self,
        py: Python<'_>,
        column: &str,
        groups: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.agg_or_window(
            py,
            &format!("GEOMEAN({column})"),
            "geomean",
            groups,
            "",
            projected_columns,
        )
    }

    #[pyo3(signature = (column, min=None, max=None, groups="", window_spec="", projected_columns=""))]
    fn bitstring_agg(
        &self,
        py: Python<'_>,
        column: &str,
        min: Option<&Bound<'_, PyAny>>,
        max: Option<&Bound<'_, PyAny>>,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        // DuckDB requires min and max together, and both must be integers.
        let call = match (min, max) {
            (None, None) => format!("BITSTRING_AGG({column})"),
            (Some(_), None) | (None, Some(_)) => {
                return Err(jude_exception(
                    py,
                    "InvalidInputException",
                    "Both min and max values must be set for bitstring_agg",
                ));
            }
            (Some(lo), Some(hi)) => {
                let lo_i = lo.extract::<i64>().map_err(|_| {
                    jude_exception(
                        py,
                        "InvalidTypeException",
                        "bitstring_agg min value must be an integer",
                    )
                })?;
                let hi_i = hi.extract::<i64>().map_err(|_| {
                    jude_exception(
                        py,
                        "InvalidTypeException",
                        "bitstring_agg max value must be an integer",
                    )
                })?;
                format!("BITSTRING_AGG({column}, {lo_i}, {hi_i})")
            }
        };
        self.agg_or_window(
            py,
            &call,
            "bitstring_agg",
            groups,
            window_spec,
            projected_columns,
        )
    }

    // ---- Extended aggregates (Vane surface): groups + optional window_spec ----

    #[pyo3(signature = (column, groups="", window_spec="", projected_columns=""))]
    fn any_value(
        &self,
        py: Python<'_>,
        column: &str,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.agg_or_window(
            py,
            &format!("ANY_VALUE({column})"),
            "any_value",
            groups,
            window_spec,
            projected_columns,
        )
    }

    #[pyo3(signature = (arg_column, value_column, groups="", window_spec="", projected_columns=""))]
    fn arg_max(
        &self,
        py: Python<'_>,
        arg_column: &str,
        value_column: &str,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.agg_or_window(
            py,
            &format!("ARG_MAX({arg_column}, {value_column})"),
            "arg_max",
            groups,
            window_spec,
            projected_columns,
        )
    }

    #[pyo3(signature = (arg_column, value_column, groups="", window_spec="", projected_columns=""))]
    fn arg_min(
        &self,
        py: Python<'_>,
        arg_column: &str,
        value_column: &str,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.agg_or_window(
            py,
            &format!("ARG_MIN({arg_column}, {value_column})"),
            "arg_min",
            groups,
            window_spec,
            projected_columns,
        )
    }

    #[pyo3(signature = (column, groups="", window_spec="", projected_columns=""))]
    fn product(
        &self,
        py: Python<'_>,
        column: &str,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.agg_or_window(
            py,
            &format!("PRODUCT({column})"),
            "product",
            groups,
            window_spec,
            projected_columns,
        )
    }

    #[pyo3(signature = (column, groups="", window_spec="", projected_columns=""))]
    fn list(
        &self,
        py: Python<'_>,
        column: &str,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.agg_or_window(
            py,
            &format!("LIST({column})"),
            "list",
            groups,
            window_spec,
            projected_columns,
        )
    }

    #[pyo3(signature = (column, sep=",", groups="", window_spec="", projected_columns=""))]
    fn string_agg(
        &self,
        py: Python<'_>,
        column: &str,
        sep: &str,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.agg_or_window(
            py,
            &format!("STRING_AGG({column}, '{}')", sep.replace('\'', "''")),
            "string_agg",
            groups,
            window_spec,
            projected_columns,
        )
    }

    #[pyo3(signature = (column, groups="", window_spec="", projected_columns=""))]
    fn mode(
        &self,
        py: Python<'_>,
        column: &str,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.agg_or_window(
            py,
            &format!("MODE({column})"),
            "mode",
            groups,
            window_spec,
            projected_columns,
        )
    }

    #[pyo3(signature = (column, q=None, groups="", window_spec="", projected_columns=""))]
    fn quantile_cont(
        &self,
        py: Python<'_>,
        column: &str,
        q: Option<&Bound<'_, PyAny>>,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        let qarg = render_quantile_arg(q, "0.5")?;
        self.agg_or_window(
            py,
            &format!("QUANTILE_CONT({column}, {qarg})"),
            "quantile_cont",
            groups,
            window_spec,
            projected_columns,
        )
    }

    #[pyo3(signature = (column, q=None, groups="", window_spec="", projected_columns=""))]
    fn quantile_disc(
        &self,
        py: Python<'_>,
        column: &str,
        q: Option<&Bound<'_, PyAny>>,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        let qarg = render_quantile_arg(q, "0.5")?;
        self.agg_or_window(
            py,
            &format!("QUANTILE_DISC({column}, {qarg})"),
            "quantile_disc",
            groups,
            window_spec,
            projected_columns,
        )
    }

    #[pyo3(name = "quantile", signature = (column, q=None, groups="", window_spec="", projected_columns=""))]
    fn quantile(
        &self,
        py: Python<'_>,
        column: &str,
        q: Option<&Bound<'_, PyAny>>,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.quantile_disc(py, column, q, groups, window_spec, projected_columns)
    }

    #[pyo3(signature = (column, groups="", window_spec="", projected_columns=""))]
    fn std_pop(
        &self,
        py: Python<'_>,
        column: &str,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.agg_or_window(
            py,
            &format!("STDDEV_POP({column})"),
            "std_pop",
            groups,
            window_spec,
            projected_columns,
        )
    }

    #[pyo3(name = "stddev_pop", signature = (column, groups="", window_spec="", projected_columns=""))]
    fn stddev_pop(
        &self,
        py: Python<'_>,
        column: &str,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.std_pop(py, column, groups, window_spec, projected_columns)
    }

    #[pyo3(signature = (column, groups="", window_spec="", projected_columns=""))]
    fn var_pop(
        &self,
        py: Python<'_>,
        column: &str,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.agg_or_window(
            py,
            &format!("VAR_POP({column})"),
            "var_pop",
            groups,
            window_spec,
            projected_columns,
        )
    }

    // first/last/geomean use the 3-arg (expression, groups, projected_columns)
    // signature — no window_spec — matching Vane.
    #[pyo3(signature = (column, groups="", projected_columns=""))]
    fn first(
        &self,
        py: Python<'_>,
        column: &str,
        groups: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.agg_or_window(
            py,
            &format!("FIRST({column})"),
            "first",
            groups,
            "",
            projected_columns,
        )
    }

    #[pyo3(signature = (column, groups="", projected_columns=""))]
    fn last(
        &self,
        py: Python<'_>,
        column: &str,
        groups: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.agg_or_window(
            py,
            &format!("LAST({column})"),
            "last",
            groups,
            "",
            projected_columns,
        )
    }

    #[pyo3(signature = (column, groups="", window_spec="", projected_columns=""))]
    fn bit_and(
        &self,
        py: Python<'_>,
        column: &str,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.agg_or_window(
            py,
            &format!("BIT_AND({column})"),
            "bit_and",
            groups,
            window_spec,
            projected_columns,
        )
    }

    #[pyo3(signature = (column, groups="", window_spec="", projected_columns=""))]
    fn bit_or(
        &self,
        py: Python<'_>,
        column: &str,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.agg_or_window(
            py,
            &format!("BIT_OR({column})"),
            "bit_or",
            groups,
            window_spec,
            projected_columns,
        )
    }

    #[pyo3(signature = (column, groups="", window_spec="", projected_columns=""))]
    fn bit_xor(
        &self,
        py: Python<'_>,
        column: &str,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.agg_or_window(
            py,
            &format!("BIT_XOR({column})"),
            "bit_xor",
            groups,
            window_spec,
            projected_columns,
        )
    }

    #[pyo3(signature = (column, groups="", window_spec="", projected_columns=""))]
    fn bool_and(
        &self,
        py: Python<'_>,
        column: &str,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.agg_or_window(
            py,
            &format!("BOOL_AND({column})"),
            "bool_and",
            groups,
            window_spec,
            projected_columns,
        )
    }

    #[pyo3(signature = (column, groups="", window_spec="", projected_columns=""))]
    fn bool_or(
        &self,
        py: Python<'_>,
        column: &str,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.agg_or_window(
            py,
            &format!("BOOL_OR({column})"),
            "bool_or",
            groups,
            window_spec,
            projected_columns,
        )
    }

    #[pyo3(signature = (column, groups="", window_spec="", projected_columns=""))]
    fn histogram(
        &self,
        py: Python<'_>,
        column: &str,
        groups: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.agg_or_window(
            py,
            &format!("HISTOGRAM({column})"),
            "histogram",
            groups,
            window_spec,
            projected_columns,
        )
    }

    /// User-defined aggregate UDF via group-apply. DuckDB collects each group's
    /// rows with `list(...)`; `fn_` then receives a `pyarrow.Table` of the
    /// group's rows (the named `columns`) and returns a scalar (-> `result_name`)
    /// or a dict of output-column -> scalar. Empty `group_by` = one global row.
    #[pyo3(signature = (fn_, columns, group_by=None, result_name="result"))]
    fn aggregate_udf(
        &self,
        py: Python<'_>,
        fn_: &Bound<'_, PyAny>,
        columns: Vec<String>,
        group_by: Option<Vec<String>>,
        result_name: &str,
    ) -> PyResult<Self> {
        if columns.is_empty() {
            return Err(jude_exception(
                py,
                "InvalidInputException",
                "aggregate_udf requires at least one input column",
            ));
        }
        let groups = group_by.unwrap_or_default();
        let inner = self.to_subquery_sql(py)?;
        // Collect each group's values per column with list(); the group keys pass
        // through. DuckDB does the (fast, in-engine) grouping.
        let list_names: Vec<String> = (0..columns.len()).map(|i| format!("_agg_{i}")).collect();
        let list_sel: Vec<String> = columns
            .iter()
            .zip(&list_names)
            .map(|(c, ln)| format!("list({}) AS {ln}", quote_ident(c)))
            .collect();
        let group_sel: Vec<String> = groups.iter().map(|g| quote_ident(g)).collect();
        let select = if group_sel.is_empty() {
            list_sel.join(", ")
        } else {
            format!("{}, {}", group_sel.join(", "), list_sel.join(", "))
        };
        let sql = if group_sel.is_empty() {
            format!("SELECT {select} FROM ({inner}) AS _t")
        } else {
            format!(
                "SELECT {select} FROM ({inner}) AS _t GROUP BY {}",
                group_sel.join(", ")
            )
        };
        // Materialize the grouped result, then reduce each group in Python.
        let grouped = {
            let conn = self.conn.borrow(py);
            let (batches, schema) = conn.run_sql_with_schema(&sql).map_err(PyErr::from)?;
            crate::arrow_ffi::batches_to_pyarrow_table(py, &batches, &schema)?
        };
        let helper = py.import("jude._aggregate_udf")?;
        let out = helper.call_method1(
            "apply_group_aggregate",
            (
                grouped,
                list_names,
                groups.clone(),
                columns,
                fn_,
                result_name,
            ),
        )?;
        let batches = crate::arrow_ffi::py_arrow_to_batches(py, &out)?;
        let conn_ref = self.conn.borrow(py);
        Self::new_materialized(py, &conn_ref, batches)
    }

    #[pyo3(signature = (column, groups=""))]
    fn value_counts(&self, py: Python<'_>, column: &str, groups: &str) -> PyResult<Self> {
        // COUNT per distinct value of `column` within each group. The grouping
        // key is the group columns plus the counted column, de-duplicated — so
        // value_counts("v", groups="v") groups by v once (yielding (v, count)),
        // not twice (which would project v redundantly).
        let mut cols: Vec<String> = Vec::new();
        for c in groups.split(',').map(str::trim).filter(|c| !c.is_empty()) {
            if !cols.iter().any(|x| x == c) {
                cols.push(c.to_string());
            }
        }
        if !cols.iter().any(|x| x == column) {
            cols.push(column.to_string());
        }
        let g = cols.join(", ");
        self.aggregate(py, &format!("COUNT({column}) AS count"), &g)
    }

    // ---- Window functions ----

    #[pyo3(signature = (window_spec, projected_columns=""))]
    fn row_number(
        &self,
        py: Python<'_>,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.window_fn(
            py,
            "ROW_NUMBER()",
            "row_number",
            window_spec,
            projected_columns,
        )
    }

    #[pyo3(signature = (window_spec, projected_columns=""))]
    fn rank(&self, py: Python<'_>, window_spec: &str, projected_columns: &str) -> PyResult<Self> {
        self.window_fn(py, "RANK()", "rank", window_spec, projected_columns)
    }

    #[pyo3(signature = (window_spec, projected_columns=""))]
    fn dense_rank(
        &self,
        py: Python<'_>,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.window_fn(
            py,
            "DENSE_RANK()",
            "dense_rank",
            window_spec,
            projected_columns,
        )
    }

    #[pyo3(name = "rank_dense", signature = (window_spec, projected_columns=""))]
    fn rank_dense(
        &self,
        py: Python<'_>,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.dense_rank(py, window_spec, projected_columns)
    }

    #[pyo3(signature = (window_spec, projected_columns=""))]
    fn percent_rank(
        &self,
        py: Python<'_>,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.window_fn(
            py,
            "PERCENT_RANK()",
            "percent_rank",
            window_spec,
            projected_columns,
        )
    }

    #[pyo3(signature = (window_spec, projected_columns=""))]
    fn cume_dist(
        &self,
        py: Python<'_>,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.window_fn(
            py,
            "CUME_DIST()",
            "cume_dist",
            window_spec,
            projected_columns,
        )
    }

    #[pyo3(signature = (window_spec, num_buckets, projected_columns=""))]
    fn n_tile(
        &self,
        py: Python<'_>,
        window_spec: &str,
        num_buckets: i64,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.window_fn(
            py,
            &format!("NTILE({num_buckets})"),
            "ntile",
            window_spec,
            projected_columns,
        )
    }

    #[pyo3(signature = (column, window_spec="", projected_columns=""))]
    fn first_value(
        &self,
        py: Python<'_>,
        column: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.window_fn(
            py,
            &format!("FIRST_VALUE({column})"),
            "first_value",
            window_spec,
            projected_columns,
        )
    }

    #[pyo3(signature = (column, window_spec="", projected_columns=""))]
    fn last_value(
        &self,
        py: Python<'_>,
        column: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.window_fn(
            py,
            &format!("LAST_VALUE({column})"),
            "last_value",
            window_spec,
            projected_columns,
        )
    }

    #[pyo3(signature = (column, window_spec, offset, projected_columns=""))]
    fn nth_value(
        &self,
        py: Python<'_>,
        column: &str,
        window_spec: &str,
        offset: i64,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.window_fn(
            py,
            &format!("NTH_VALUE({column}, {offset})"),
            "nth_value",
            window_spec,
            projected_columns,
        )
    }

    #[pyo3(signature = (column, window_spec, offset=1, default_value=None, projected_columns=""))]
    fn lag(
        &self,
        py: Python<'_>,
        column: &str,
        window_spec: &str,
        offset: i64,
        default_value: Option<&str>,
        projected_columns: &str,
    ) -> PyResult<Self> {
        let call = match default_value {
            Some(d) => format!("LAG({column}, {offset}, {d})"),
            None => format!("LAG({column}, {offset})"),
        };
        self.window_fn(py, &call, "lag", window_spec, projected_columns)
    }

    #[pyo3(signature = (column, window_spec, offset=1, default_value=None, projected_columns=""))]
    fn lead(
        &self,
        py: Python<'_>,
        column: &str,
        window_spec: &str,
        offset: i64,
        default_value: Option<&str>,
        projected_columns: &str,
    ) -> PyResult<Self> {
        let call = match default_value {
            Some(d) => format!("LEAD({column}, {offset}, {d})"),
            None => format!("LEAD({column}, {offset})"),
        };
        self.window_fn(py, &call, "lead", window_spec, projected_columns)
    }

    /// Generic window function: `function_name(function_parameters) OVER (window_spec)`.
    #[pyo3(signature = (function_name, function_parameters, window_spec, projected_columns=""))]
    fn generic_window_function(
        &self,
        py: Python<'_>,
        function_name: &str,
        function_parameters: &str,
        window_spec: &str,
        projected_columns: &str,
    ) -> PyResult<Self> {
        self.window_fn(
            py,
            &format!("{function_name}({function_parameters})"),
            &function_name.to_lowercase(),
            window_spec,
            projected_columns,
        )
    }

    // ---- Explode / sample ----

    /// UNNEST a list/array column into one row per element (explode). The other
    /// columns are carried along. Central to multimodal fan-out (1 video -> N
    /// frames, 1 document -> N chunks).
    #[pyo3(signature = (column, recursive=false))]
    fn unnest(&self, py: Python<'_>, column: &str, recursive: bool) -> PyResult<Self> {
        let column = column.to_string();
        self.derive_plan(py, |input| LogicalPlan::Unnest {
            input,
            column,
            recursive,
        })
    }

    /// Alias for `unnest` (Spark/DataFrame naming).
    #[pyo3(signature = (column, recursive=false))]
    fn explode(&self, py: Python<'_>, column: &str, recursive: bool) -> PyResult<Self> {
        self.unnest(py, column, recursive)
    }

    /// Random sample. `spec` is a DuckDB sample clause fragment, e.g. "10%" or
    /// "100 ROWS" or "reservoir(50)".
    fn sample(&self, py: Python<'_>, spec: &str) -> PyResult<Self> {
        let spec = spec.to_string();
        self.derive_plan(py, |input| LogicalPlan::Sample { input, spec })
    }

    // ---- I/O ----

    fn to_csv(&self, py: Python<'_>, filename: &str) -> PyResult<()> {
        let inner = self.to_subquery_sql(py)?;
        let conn_ref = self.conn.borrow(py);
        conn_ref
            .inner
            .execute_batch(&format!(
                "COPY ({inner}) TO '{}' (FORMAT CSV, HEADER)",
                crate::connection::escape_sql_string(filename)
            ))
            .map_err(|e| Error::DuckDb(e).into())
    }

    pub fn write_csv(&self, py: Python<'_>, filename: &str) -> PyResult<()> {
        self.to_csv(py, filename)
    }

    fn to_parquet(&self, py: Python<'_>, filename: &str) -> PyResult<()> {
        let inner = self.to_subquery_sql(py)?;
        let conn_ref = self.conn.borrow(py);
        conn_ref
            .inner
            .execute_batch(&format!(
                "COPY ({inner}) TO '{}' (FORMAT PARQUET)",
                crate::connection::escape_sql_string(filename)
            ))
            .map_err(|e| Error::DuckDb(e).into())
    }

    fn write_parquet(&self, py: Python<'_>, filename: &str) -> PyResult<()> {
        self.to_parquet(py, filename)
    }

    fn to_table(&self, py: Python<'_>, table_name: &str) -> PyResult<()> {
        let inner = self.to_subquery_sql(py)?;
        let conn_ref = self.conn.borrow(py);
        conn_ref
            .inner
            .execute_batch(&format!(
                "CREATE OR REPLACE TABLE {} AS {inner}",
                quote_ident(table_name)
            ))
            .map_err(|e| Error::DuckDb(e).into())
    }

    fn create(&self, py: Python<'_>, table_name: &str) -> PyResult<()> {
        self.to_table(py, table_name)
    }

    #[pyo3(signature = (view_name, replace=true))]
    fn to_view(&self, py: Python<'_>, view_name: &str, replace: bool) -> PyResult<()> {
        let inner = self.to_subquery_sql(py)?;
        let kw = if replace {
            "CREATE OR REPLACE VIEW"
        } else {
            "CREATE VIEW"
        };
        let conn_ref = self.conn.borrow(py);
        conn_ref
            .inner
            .execute_batch(&format!("{kw} {} AS {inner}", quote_ident(view_name)))
            .map_err(|e| Error::DuckDb(e).into())
    }

    #[pyo3(signature = (view_name, replace=true))]
    fn create_view(&self, py: Python<'_>, view_name: &str, replace: bool) -> PyResult<()> {
        self.to_view(py, view_name, replace)
    }

    fn insert_into(&self, py: Python<'_>, table_name: &str) -> PyResult<()> {
        let inner = self.to_subquery_sql(py)?;
        let conn_ref = self.conn.borrow(py);
        conn_ref
            .inner
            .execute_batch(&format!("INSERT INTO {} {inner}", quote_ident(table_name)))
            .map_err(|e| Error::DuckDb(e).into())
    }

    /// Insert one row of literal values into the base table. Only valid on a
    /// table relation (matches DuckDB, which raises otherwise).
    fn insert(&self, py: Python<'_>, values: &Bound<'_, PyAny>) -> PyResult<()> {
        let name = base_table_name(&self.plan).ok_or_else(|| {
            jude_exception(
                py,
                "InvalidInputException",
                "'DuckDBPyRelation.insert' can only be used on a table relation",
            )
        })?;
        let mut cells = Vec::new();
        for item in values.try_iter()? {
            cells.push(literal_to_sql(&item?)?);
        }
        let sql = format!(
            "INSERT INTO {} VALUES ({})",
            quote_qualified_name(&name),
            cells.join(", ")
        );
        let conn_ref = self.conn.borrow(py);
        conn_ref
            .inner
            .execute_batch(&sql)
            .map_err(|e| Error::DuckDb(e).into())
    }

    /// UPDATE the base table: `set` maps column -> value/Expression, `condition`
    /// is an optional WHERE Expression. Only valid on a table relation.
    #[pyo3(signature = (set, condition=None))]
    fn update(
        &self,
        py: Python<'_>,
        set: &Bound<'_, PyAny>,
        condition: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let name = base_table_name(&self.plan).ok_or_else(|| {
            jude_exception(
                py,
                "InvalidInputException",
                "'DuckDBPyRelation.update' can only be used on a table relation",
            )
        })?;
        let dict = set.cast::<pyo3::types::PyDict>().map_err(|_| {
            jude_exception(
                py,
                "InvalidInputException",
                "Please provide 'set' as a dictionary of column name to Expression",
            )
        })?;
        if dict.is_empty() {
            return Err(jude_exception(
                py,
                "InvalidInputException",
                "Please provide at least one set expression",
            ));
        }
        let mut assigns = Vec::new();
        for (k, v) in dict.iter() {
            let col = k.extract::<String>().map_err(|_| {
                jude_exception(
                    py,
                    "InvalidInputException",
                    "Please provide the column name as the key of the dictionary",
                )
            })?;
            // The value must be an Expression or a plain scalar literal; anything
            // else (a set, list, …) is rejected with DuckDB's exact message.
            let is_scalar = v.is_none()
                || v.is_instance_of::<pyo3::types::PyBool>()
                || v.is_instance_of::<pyo3::types::PyInt>()
                || v.is_instance_of::<pyo3::types::PyFloat>()
                || v.is_instance_of::<pyo3::types::PyString>();
            if !is_scalar && v.extract::<crate::expressions::Expression>().is_err() {
                let tn = v.get_type().name()?;
                return Err(jude_exception(
                    py,
                    "InvalidInputException",
                    &format!("Please provide an object of type Expression as the value, not <class '{tn}'>"),
                ));
            }
            assigns.push(format!("{} = {}", quote_ident(&col), literal_to_sql(&v)?));
        }
        let mut sql = format!(
            "UPDATE {} SET {}",
            quote_qualified_name(&name),
            assigns.join(", ")
        );
        if let Some(cond) = condition {
            if !cond.is_none() {
                sql.push_str(&format!(" WHERE {}", expr_to_sql(cond)?));
            }
        }
        let conn_ref = self.conn.borrow(py);
        conn_ref
            .inner
            .execute_batch(&sql)
            .map_err(|e| Error::DuckDb(e).into())
    }

    /// Write this relation to an Apache Iceberg table (distributed-write P2).
    ///
    /// jude is a *write* engine here: the DATA files are written in Rust via
    /// DuckDB `COPY TO ... (FORMAT PARQUET)` — one file per partition (from the
    /// Rust `WorkerManager`'s partition plan), the heavy path staying off Python
    /// — and only the table-format **commit** (register the files as a new
    /// snapshot) goes through the thin `jude._iceberg_commit` shim (pyiceberg's
    /// add_files). `mode` is "append" or "overwrite". Returns the table's new
    /// metadata location.
    #[pyo3(signature = (warehouse, table, mode="append", num_files=None))]
    fn write_iceberg(
        &self,
        py: Python<'_>,
        warehouse: &str,
        table: &str,
        mode: &str,
        num_files: Option<usize>,
    ) -> PyResult<String> {
        use std::path::Path;
        // Decide how many data files to write: an explicit override, else the
        // Rust scheduler's partition count for this relation's size.
        self.materialize(py)?;
        let (num_rows, nbytes) = self.with_batches(py, |batches, _| {
            let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            let bytes: usize = batches.iter().map(|b| b.get_array_memory_size()).sum();
            (rows, bytes as u64)
        })?;
        let n = num_files.unwrap_or_else(|| crate::dist::default_partition_count(num_rows, nbytes));

        // Stage the per-partition Parquet files in a fresh temp dir.
        let stage =
            std::env::temp_dir().join(format!("jude_iceberg_{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&stage).map_err(Error::Io)?;
        let inner = self.to_subquery_sql(py)?;
        let mut files: Vec<String> = Vec::new();
        {
            let conn_ref = self.conn.borrow(py);
            if n <= 1 || num_rows == 0 {
                let f = stage.join("part-0.parquet");
                conn_ref
                    .inner
                    .execute_batch(&format!(
                        "COPY ({inner}) TO '{}' (FORMAT PARQUET)",
                        f.display()
                    ))
                    .map_err(Error::DuckDb)?;
                files.push(f.to_string_lossy().into_owned());
            } else {
                // One file per row-slice partition (the Rust partition plan) via
                // a windowed row_number split — parallelizable data write in Rust.
                let step = num_rows.div_ceil(n);
                for i in 0..n {
                    let lo = i * step;
                    if lo >= num_rows {
                        break;
                    }
                    let f: &Path = &stage.join(format!("part-{i}.parquet"));
                    let sql = format!(
                        "COPY (SELECT * EXCLUDE (_jude_rn) FROM (SELECT *, row_number() OVER () AS _jude_rn FROM ({inner}) AS _s) AS _t WHERE _jude_rn > {lo} AND _jude_rn <= {}) TO '{}' (FORMAT PARQUET)",
                        lo + step,
                        f.display()
                    );
                    conn_ref.inner.execute_batch(&sql).map_err(Error::DuckDb)?;
                    files.push(f.to_string_lossy().into_owned());
                }
            }
        }

        // Commit: register the jude-written Parquet files into the Iceberg table.
        let module = py.import("jude._iceberg_commit")?;
        let meta: String = module
            .call_method1("commit", (warehouse, table, files, mode))?
            .extract()?;
        Ok(meta)
    }

    /// Write this relation to a Lance dataset (single-machine). The data path is
    /// Lance's Rust writer; jude materializes to Arrow and hands it over.
    /// `mode` is create | append | overwrite. See `docs/storage_design.zh.md`.
    #[pyo3(signature = (path, mode="create"))]
    fn write_lance(&self, py: Python<'_>, path: &str, mode: &str) -> PyResult<Py<PyAny>> {
        let table = self.to_arrow(py)?;
        let helper = py.import("jude._lance")?;
        Ok(helper.call_method1("write", (table, path, mode))?.into())
    }

    // ---- Fetch (DBAPI cursor: fetchone/fetchmany/fetchall share a position) ----

    /// Collect up to `limit` rows (or all remaining when `None`) starting at the
    /// current cursor position, advancing it past what was read.
    fn take_rows(&self, py: Python<'_>, limit: Option<usize>) -> PyResult<Vec<Py<PyAny>>> {
        self.materialize(py)?;
        let hints = relation_logical_hints(self, py);
        let cache = self.cache.borrow();
        let (batches, _) = cache.as_ref().unwrap();
        let start = *self.read_pos.borrow();
        let mut out: Vec<Py<PyAny>> = Vec::new();
        let mut seen = 0usize; // absolute row index across batches
        let mut taken = 0usize;
        for batch in batches {
            let n = batch.num_rows();
            if seen + n <= start {
                seen += n;
                continue;
            }
            let local_start = start.saturating_sub(seen);
            for r in local_start..n {
                if let Some(lim) = limit {
                    if taken >= lim {
                        break;
                    }
                }
                let mut row = Vec::with_capacity(batch.num_columns());
                for col_idx in 0..batch.num_columns() {
                    let col = batch.column(col_idx);
                    let v = if col.is_null(r) {
                        py.None()
                    } else {
                        array_value_to_py(col, r, py)?
                    };
                    row.push(apply_hint(py, v, hints.get(col_idx).copied())?);
                }
                out.push(pyo3::types::PyTuple::new(py, row)?.into());
                taken += 1;
            }
            seen += n;
            if limit.is_some_and(|lim| taken >= lim) {
                break;
            }
        }
        *self.read_pos.borrow_mut() = start + taken;
        Ok(out)
    }

    fn fetchall(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // DuckDB relational quirk: fetchall() on an already-exhausted cursor
        // re-scans from the start (returns the full result again), unlike
        // fetchone/fetchmany which stay exhausted.
        self.materialize(py)?;
        let total: usize = self
            .cache
            .borrow()
            .as_ref()
            .map(|(b, _)| b.iter().map(|x| x.num_rows()).sum())
            .unwrap_or(0);
        if total > 0 && *self.read_pos.borrow() >= total {
            *self.read_pos.borrow_mut() = 0;
        }
        let rows = self.take_rows(py, None)?;
        Ok(PyList::new(py, rows.iter().map(|r| r.bind(py)))?.into())
    }

    fn fetchone(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let rows = self.take_rows(py, Some(1))?;
        Ok(rows.into_iter().next().unwrap_or_else(|| py.None()))
    }

    #[pyo3(signature = (size=1))]
    fn fetchmany(&self, py: Python<'_>, size: usize) -> PyResult<Py<PyAny>> {
        let rows = self.take_rows(py, Some(size))?;
        Ok(PyList::new(py, rows.iter().map(|r| r.bind(py)))?.into())
    }

    fn fetch_df(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.to_df(py)
    }
    fn fetchdf(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.to_df(py)
    }
    fn df(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.to_df(py)
    }

    // ---- Conversion ----

    fn to_arrow(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.with_batches(py, |batches, schema| {
            crate::arrow_ffi::batches_to_pyarrow_table(py, batches, schema)
        })?
    }

    fn arrow(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.to_arrow(py)
    }

    fn to_arrow_table(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.to_arrow(py)
    }

    fn to_df(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let table = self.to_arrow(py)?;
        let df = table.bind(py).call_method0("to_pandas")?;
        Ok(df.into())
    }

    fn to_pandas(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.to_df(py)
    }

    /// DuckDB `fetch_df_chunk(vectors_per_chunk=1)`: return the next
    /// `vectors_per_chunk * STANDARD_VECTOR_SIZE` (2048) rows from the DBAPI
    /// cursor as a pandas DataFrame, advancing the cursor. Empty DataFrame when
    /// the cursor is exhausted.
    #[pyo3(signature = (vectors_per_chunk=1, date_as_object=false))]
    fn fetch_df_chunk(
        &self,
        py: Python<'_>,
        vectors_per_chunk: usize,
        date_as_object: bool,
    ) -> PyResult<Py<PyAny>> {
        const STANDARD_VECTOR_SIZE: usize = 2048;
        let n = vectors_per_chunk.max(1) * STANDARD_VECTOR_SIZE;
        self.materialize(py)?;
        let start = *self.read_pos.borrow();
        // Slice the next `n` rows out of the cached batches into an Arrow table,
        // then hand to pandas — reuses the same cursor position as fetchmany.
        let table = self.with_batches(py, |batches, schema| {
            let pyarrow = py.import("pyarrow")?;
            let mut sliced: Vec<Py<PyAny>> = Vec::new();
            let mut seen = 0usize;
            let mut taken = 0usize;
            for b in batches {
                let rows = b.num_rows();
                if seen + rows <= start {
                    seen += rows;
                    continue;
                }
                let local_start = start.saturating_sub(seen);
                let avail = rows - local_start;
                let want = (n - taken).min(avail);
                let pb = crate::arrow_ffi::batches_to_pyarrow_table(
                    py,
                    std::slice::from_ref(b),
                    schema,
                )?;
                let slice = pb.bind(py).call_method1("slice", (local_start, want))?;
                sliced.push(slice.unbind());
                taken += want;
                seen += rows;
                if taken >= n {
                    break;
                }
            }
            let list = PyList::new(py, sliced.iter().map(|t| t.bind(py)))?;
            let combined = pyarrow.call_method1("concat_tables", (list,))?;
            *self.read_pos.borrow_mut() = start + taken;
            Ok::<Py<PyAny>, PyErr>(combined.unbind())
        })??;
        // date_as_object controls whether DATE columns come back as pandas
        // datetime64 (pd.Timestamp) or Python datetime.date, matching DuckDB.
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("date_as_object", date_as_object)?;
        Ok(table
            .bind(py)
            .call_method("to_pandas", (), Some(&kwargs))?
            .into())
    }

    /// Fetch the result as a dict of column-name -> numpy array (DuckDB's
    /// `fetchnumpy`).
    fn fetchnumpy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let table = self.to_arrow(py)?;
        let table = table.bind(py);
        let names: Vec<String> = table.getattr("column_names")?.extract()?;
        let out = pyo3::types::PyDict::new(py);
        for name in names {
            let col = table.call_method1("column", (&name,))?;
            let np = col.call_method0("to_numpy")?;
            out.set_item(name, np)?;
        }
        Ok(out.into())
    }

    fn to_arrow_reader(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let table = self.to_arrow(py)?;
        let reader = table.bind(py).call_method0("to_reader")?;
        Ok(reader.into())
    }

    /// A streaming iterator over this relation's result batches (one
    /// `pyarrow.RecordBatch` per `next()`), pulled lazily from DuckDB rather than
    /// materialized all at once — bounded jude-side memory, and consumers can
    /// pipeline. `for batch in rel.record_batch_stream(): ...`.
    fn record_batch_stream(&self, py: Python<'_>) -> PyResult<crate::stream::RecordBatchStream> {
        let sql = self.to_subquery_sql(py)?;
        let conn = self.conn.borrow(py);
        crate::stream::RecordBatchStream::new(conn.inner.clone(), &sql).map_err(PyErr::from)
    }

    /// DuckDB-compatible `fetch_record_batch(rows_per_batch)` — a lazy
    /// `pyarrow.RecordBatchReader` backed by the streaming iterator. `rows_per_batch`
    /// is advisory (DuckDB controls the vector size); accepted for API parity.
    #[pyo3(signature = (rows_per_batch=1000000))]
    fn fetch_record_batch(&self, py: Python<'_>, rows_per_batch: usize) -> PyResult<Py<PyAny>> {
        let _ = rows_per_batch;
        let stream = self.record_batch_stream(py)?;
        let stream_obj = Py::new(py, stream)?;
        let schema = stream_obj.bind(py).getattr("schema")?;
        let pyarrow = py.import("pyarrow")?;
        let reader_cls = pyarrow.getattr("RecordBatchReader")?;
        Ok(reader_cls
            .call_method1("from_batches", (schema, stream_obj))?
            .into())
    }

    fn pl(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let table = self.to_arrow(py)?;
        let polars = py.import("polars")?;
        Ok(polars.call_method1("from_arrow", (table,))?.into())
    }

    fn to_polars(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.pl(py)
    }

    /// This relation as a Daft DataFrame (zero-copy via Arrow), to run Daft's
    /// multimodal / embedding / model ops.
    fn to_daft(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let table = self.to_arrow(py)?;
        let helper = py.import("jude._daft")?;
        Ok(helper.call_method1("to_daft", (table,))?.into())
    }

    /// Apply a Daft transform `fn(daft.DataFrame) -> daft.DataFrame` to this
    /// relation and return the result as a new jude relation. Full access to
    /// Daft's expression API (image decode/resize, url.download, embed_text /
    /// embed_image / classify_image) with the result flowing back into jude.
    fn daft_transform(&self, py: Python<'_>, fn_: &Bound<'_, PyAny>) -> PyResult<Self> {
        let table = self.to_arrow(py)?;
        let helper = py.import("jude._daft")?;
        let out = helper.call_method1("transform", (table, fn_))?;
        let batches = crate::arrow_ffi::py_arrow_to_batches(py, &out)?;
        let conn_ref = self.conn.borrow(py);
        Self::new_materialized(py, &conn_ref, batches)
    }

    // ---- Introspection ----

    fn sql_query(&self, py: Python<'_>) -> PyResult<String> {
        self.to_subquery_sql(py)
    }

    /// Pretty-print the logical plan tree (jude's IR), independent of SQL.
    /// The distributed stage DAG for this relation's plan: the operator tree cut
    /// into stages at shuffle boundaries (aggregate/join/distinct/order/set-op/
    /// repartition). Returns a list of dicts `{id, kind, op, partition_keys,
    /// inputs}` in dependency order. See src/dist/stage.rs.
    /// If the root operator is an Aggregate, return (input relation, group cols,
    /// agg exprs) so the distributed runner can partition the INPUT and build a
    /// two-phase plan; None otherwise.
    fn aggregate_spec(&self, py: Python<'_>) -> PyResult<Option<(Self, Vec<String>, Vec<String>)>> {
        if let LogicalPlan::Aggregate { input, group, aggs } = root_op_plan(&self.plan) {
            let conn = self.conn.borrow(py);
            let inp = Relation::from_plan(py, &conn, input.as_ref().clone())?;
            return Ok(Some((inp, group.clone(), aggs.clone())));
        }
        Ok(None)
    }

    /// If the root is an equi-join, return (left, right, keys, how) for a
    /// hash-shuffle join; None if not a join or the condition isn't equi-key.
    fn join_spec(&self, py: Python<'_>) -> PyResult<Option<(Self, Self, Vec<String>, String)>> {
        if let LogicalPlan::Join {
            left,
            right,
            condition,
            how,
        } = root_op_plan(&self.plan)
        {
            if let Some(keys) = extract_join_keys(condition) {
                let conn = self.conn.borrow(py);
                let l = Relation::from_plan(py, &conn, left.as_ref().clone())?;
                let r = Relation::from_plan(py, &conn, right.as_ref().clone())?;
                return Ok(Some((l, r, keys, how.name().to_string())));
            }
        }
        Ok(None)
    }

    /// One step of the general streaming stage-DAG executor: decompose this
    /// relation into (a) the partition-wise region at the top (SQL over a `part`
    /// placeholder, pushable per-partition or not), (b) the shuffle boundary it
    /// sits on, and (c) the child sub-relations feeding that boundary — which the
    /// Python executor recurses on, running each child distributed, exchanging the
    /// result through the object store, then applying the boundary + local SQL.
    ///
    /// Returns a dict:
    ///   local_sql: Option<str>   — pw region over `part` (None if root is boundary)
    ///   pushable: bool           — safe to run local_sql per-output-partition
    ///   has_udf: bool            — pw region contains a UDF (→ caller falls back)
    ///   boundary: str            — "Aggregate"|"Join"|"Order"|"Distinct"|"SetOp"|
    ///                              "Repartition"|"Scan"
    ///   keys: [str]              — shuffle keys (group-by / order-by / repartition)
    ///   join_keys: Option<[str]> — equi-join keys (Join only, None if non-equi)
    ///   how: Option<str>         — join type (Join only)
    ///   setop: Option<str>       — set-op kind (SetOp only)
    ///   agg_group / agg_exprs    — for Aggregate boundary (two-phase inputs)
    ///   children: [Relation]     — sub-relations to recurse on
    ///   has_shuffle: bool        — whether this subtree has ANY shuffle at all
    fn dist_step(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use crate::dist::physical;

        let d = pyo3::types::PyDict::new(py);
        let peeled = physical::peel(&self.plan);
        let local_sql = match &peeled.local {
            Some(l) => match physical::render_local_sql(l) {
                Ok(s) => Some(s),
                Err(_) => None, // unrenderable (e.g. survived Materialized) -> caller falls back
            },
            None => None,
        };
        d.set_item("local_sql", local_sql)?;
        d.set_item("pushable", peeled.pushable)?;
        d.set_item("has_udf", peeled.has_udf)?;
        d.set_item("has_shuffle", physical::subtree_has_shuffle(&self.plan))?;

        let conn = self.conn.borrow(py);
        let mk =
            |p: &LogicalPlan| -> PyResult<Relation> { Relation::from_plan(py, &conn, p.clone()) };

        match &peeled.boundary {
            LogicalPlan::Aggregate { input, group, aggs } => {
                d.set_item("boundary", "Aggregate")?;
                d.set_item("keys", group.clone())?;
                d.set_item("agg_group", group.clone())?;
                d.set_item("agg_exprs", aggs.clone())?;
                d.set_item("children", vec![mk(input)?])?;
            }
            LogicalPlan::Join {
                left,
                right,
                condition,
                how,
            } => {
                d.set_item("boundary", "Join")?;
                d.set_item("keys", vec![condition.clone()])?;
                d.set_item("join_keys", extract_join_keys(condition))?;
                d.set_item("how", how.name())?;
                d.set_item("children", vec![mk(left)?, mk(right)?])?;
            }
            LogicalPlan::SetOp { op, left, right } => {
                d.set_item("boundary", "SetOp")?;
                d.set_item("keys", Vec::<String>::new())?;
                d.set_item("setop", op.sql_kw())?;
                d.set_item("children", vec![mk(left)?, mk(right)?])?;
            }
            LogicalPlan::Order { input, keys } => {
                d.set_item("boundary", "Order")?;
                d.set_item("keys", keys.clone())?;
                d.set_item("children", vec![mk(input)?])?;
            }
            LogicalPlan::Distinct { input } => {
                d.set_item("boundary", "Distinct")?;
                d.set_item("keys", Vec::<String>::new())?;
                d.set_item("children", vec![mk(input)?])?;
            }
            LogicalPlan::Repartition { input, by, .. } => {
                d.set_item("boundary", "Repartition")?;
                d.set_item("keys", by.clone())?;
                d.set_item("children", vec![mk(input)?])?;
            }
            // A leaf (Table/ScanFunction/RawSql/Materialized/MapBatches): no
            // shuffle below the pw region — the whole subtree is one stage.
            _ => {
                d.set_item("boundary", "Scan")?;
                d.set_item("keys", Vec::<String>::new())?;
                d.set_item("children", Vec::<Relation>::new())?;
            }
        }
        Ok(d.into())
    }

    fn plan_stages(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let stages = crate::dist::stage::plan_stages(&self.plan);
        let out = PyList::empty(py);
        for s in &stages {
            let d = pyo3::types::PyDict::new(py);
            d.set_item("id", s.id)?;
            d.set_item(
                "kind",
                match s.kind {
                    crate::dist::stage::StageKind::Scan => "scan",
                    crate::dist::stage::StageKind::Partitionwise => "partitionwise",
                    crate::dist::stage::StageKind::Shuffle => "shuffle",
                },
            )?;
            d.set_item("op", s.op)?;
            d.set_item("partition_keys", s.partition_keys.clone())?;
            d.set_item("inputs", s.inputs.clone())?;
            out.append(d)?;
        }
        Ok(out.into())
    }

    fn plan_tree(&self) -> String {
        fn go(p: &LogicalPlan, depth: usize, out: &mut String) {
            for _ in 0..depth {
                out.push_str("  ");
            }
            out.push_str(p.op_name());
            out.push('\n');
            match p {
                LogicalPlan::Filter { input, .. }
                | LogicalPlan::Project { input, .. }
                | LogicalPlan::Aggregate { input, .. }
                | LogicalPlan::Order { input, .. }
                | LogicalPlan::Limit { input, .. }
                | LogicalPlan::Distinct { input }
                | LogicalPlan::Alias { input, .. }
                | LogicalPlan::Summarize { input }
                | LogicalPlan::Query { input, .. }
                | LogicalPlan::Repartition { input, .. }
                | LogicalPlan::MapBatches { input, .. }
                | LogicalPlan::Unnest { input, .. }
                | LogicalPlan::Sample { input, .. } => go(input, depth + 1, out),
                LogicalPlan::Join { left, right, .. } | LogicalPlan::SetOp { left, right, .. } => {
                    go(left, depth + 1, out);
                    go(right, depth + 1, out);
                }
                _ => {}
            }
        }
        let mut out = String::new();
        go(&self.plan, 0, &mut out);
        out
    }

    /// Serialize the top of the logical plan as a dict the Python stage-planner
    /// can walk to find shuffle boundaries. Each node: {op, is_shuffle,
    /// partition_keys, child_sql:[...], child_ops:[...]}. child_sql is the
    /// SQL lowering of each input subtree (so a stage can execute it directly).
    fn plan_json(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = pyo3::types::PyDict::new(py);
        let plan = &*self.plan;
        d.set_item("op", plan.op_name())?;
        // classify shuffle boundary + partition keys
        let (is_shuffle, keys): (bool, Vec<String>) = match plan {
            LogicalPlan::Aggregate { group, .. } => (true, group.clone()),
            LogicalPlan::Join { condition, .. } => (true, vec![condition.clone()]),
            LogicalPlan::Distinct { .. } => (true, vec![]),
            LogicalPlan::Order { keys, .. } => (true, keys.clone()),
            LogicalPlan::SetOp { .. } => (true, vec![]),
            LogicalPlan::Repartition { by, .. } => (true, by.clone()),
            _ => (false, vec![]),
        };
        d.set_item("is_shuffle", is_shuffle)?;
        d.set_item("partition_keys", keys)?;
        // children: their op names + SQL lowerings
        let conn_ref = self.conn.borrow(py);
        let inner = &conn_ref.inner;
        let mut lower = |node: &LogicalPlan| -> Result<String, Error> {
            match node {
                LogicalPlan::Materialized { batches } => {
                    crate::arrow_ffi::batches_to_temp_table(inner, batches)
                }
                other => Err(Error::Other(format!("cannot lower {}", other.op_name()))),
            }
        };
        let (child_ops, child_sql): (Vec<&str>, Vec<String>) = match plan {
            LogicalPlan::Join { left, right, .. } | LogicalPlan::SetOp { left, right, .. } => (
                vec![left.op_name(), right.op_name()],
                vec![
                    left.to_sql(&mut lower).map_err(PyErr::from)?,
                    right.to_sql(&mut lower).map_err(PyErr::from)?,
                ],
            ),
            LogicalPlan::Filter { input, .. }
            | LogicalPlan::Project { input, .. }
            | LogicalPlan::Aggregate { input, .. }
            | LogicalPlan::Order { input, .. }
            | LogicalPlan::Limit { input, .. }
            | LogicalPlan::Distinct { input }
            | LogicalPlan::Alias { input, .. }
            | LogicalPlan::Summarize { input }
            | LogicalPlan::Query { input, .. }
            | LogicalPlan::Repartition { input, .. }
            | LogicalPlan::MapBatches { input, .. }
            | LogicalPlan::Unnest { input, .. }
            | LogicalPlan::Sample { input, .. } => (
                vec![input.op_name()],
                vec![input.to_sql(&mut lower).map_err(PyErr::from)?],
            ),
            _ => (vec![], vec![]),
        };
        d.set_item("child_ops", child_ops)?;
        d.set_item("child_sql", child_sql)?;
        d.set_item(
            "full_sql",
            self.plan.to_sql(&mut lower).map_err(PyErr::from)?,
        )?;
        Ok(d.into())
    }

    fn execute(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<Self>> {
        // Re-executing rewinds the DBAPI cursor to the first row.
        *slf.read_pos.borrow_mut() = 0;
        slf.materialize(py)?;
        Ok(slf.into())
    }

    fn explain(&self, py: Python<'_>) -> PyResult<String> {
        let inner = self.to_subquery_sql(py)?;
        let conn_ref = self.conn.borrow(py);
        let batches = conn_ref
            .run_sql(&format!("EXPLAIN {inner}"))
            .map_err(PyErr::from)?;
        let mut out = String::new();
        for batch in &batches {
            for col_idx in 0..batch.num_columns() {
                if let Some(arr) = batch.column(col_idx).as_any().downcast_ref::<StringArray>() {
                    for i in 0..arr.len() {
                        if !arr.is_null(i) {
                            out.push_str(arr.value(i));
                            out.push('\n');
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    fn describe(&self, py: Python<'_>) -> PyResult<Self> {
        self.derive_plan(py, |input| LogicalPlan::Summarize { input })
    }

    // ---- Map / UDF (out-of-band batch transform) ----

    /// Apply a Python function to each batch (Arrow Table in, Arrow Table out).
    ///
    /// Matches Vane's `map_batches(fn, *, schema=None, batch_size=None,
    /// execution_backend=None, num_workers=None, ...)`. With
    /// `execution_backend="subprocess"` the UDF runs in a pool of worker
    /// processes (GIL-free parallelism); otherwise it runs in-process.
    #[pyo3(name = "map_batches", signature = (fn_, *, schema=None, batch_size=None, max_batch_bytes=None, execution_backend=None, num_workers=None, num_gpus=None, **_kwargs))]
    fn map_batches_py(
        &self,
        py: Python<'_>,
        fn_: &Bound<'_, PyAny>,
        schema: Option<&Bound<'_, PyAny>>,
        batch_size: Option<usize>,
        max_batch_bytes: Option<usize>,
        execution_backend: Option<&str>,
        num_workers: Option<usize>,
        num_gpus: Option<f64>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Self> {
        let _ = schema;
        // Byte-based batching (env fallback: VANE_UDF_TARGET_MAX_BATCH_BYTES).
        let max_bytes = max_batch_bytes.or_else(udf_target_max_batch_bytes_env);
        if matches!(
            execution_backend,
            Some("subprocess") | Some("subprocess_task") | Some("subprocess_actor")
        ) {
            return self.map_batches_subprocess(
                py,
                fn_,
                batch_size,
                max_bytes,
                num_workers,
                "map_batches",
            );
        }
        // ray_task / ray_actor: per-batch Ray execution (Vane's execution model)
        // via jude.execution. Plain "ray" uses partition-level map_relation.
        if matches!(execution_backend, Some("ray_task") | Some("ray_actor")) {
            return self.map_batches_exec(
                py,
                fn_,
                batch_size,
                execution_backend.unwrap(),
                num_workers,
                num_gpus,
            );
        }
        if matches!(execution_backend, Some("ray")) {
            return self.map_batches_ray(py, fn_, batch_size);
        }
        self.map_batches_inprocess(py, fn_, batch_size)
    }

    /// Apply a Python function per input batch that returns zero-or-more output
    /// rows (one-to-many). The callable receives a pyarrow Table and returns a
    /// Table (or something convertible to one).
    #[pyo3(name = "flat_map", signature = (fn_, *, schema=None, batch_size=None, max_batch_bytes=None, execution_backend=None, num_workers=None, num_gpus=None, **_kwargs))]
    fn flat_map_py(
        &self,
        py: Python<'_>,
        fn_: &Bound<'_, PyAny>,
        schema: Option<&Bound<'_, PyAny>>,
        batch_size: Option<usize>,
        max_batch_bytes: Option<usize>,
        execution_backend: Option<&str>,
        num_workers: Option<usize>,
        num_gpus: Option<f64>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Self> {
        // Same mechanics as map_batches for the local path; the difference is
        // purely semantic (output cardinality may differ from input).
        self.map_batches_py(
            py,
            fn_,
            schema,
            batch_size,
            max_batch_bytes,
            execution_backend,
            num_workers,
            num_gpus,
            _kwargs,
        )
    }

    /// Scalar map: apply `fn_` per row over `column` (first column if omitted),
    /// producing a row-preserving `output_column`. Vane's `map`/scalar call mode.
    #[pyo3(name = "map", signature = (fn_, column=None, *, output_column="result", batch_size=None))]
    fn map_scalar(
        &self,
        py: Python<'_>,
        fn_: &Bound<'_, PyAny>,
        column: Option<&str>,
        output_column: &str,
        batch_size: Option<usize>,
    ) -> PyResult<Self> {
        self.materialize(py)?;
        let cache = self.cache.borrow();
        let (batches, sch) = cache.as_ref().unwrap();
        let table = crate::arrow_ffi::batches_to_pyarrow_table(py, batches, sch)?;
        drop(cache);
        let table = table.bind(py);
        let col_name = column.map(|s| s.to_string());
        let pa_batches = match batch_size {
            Some(n) if n > 0 => table.call_method1("to_batches", (n,))?,
            _ => table.call_method0("to_batches")?,
        };
        let pyarrow = py.import("pyarrow")?;
        let mut out_tables = Vec::new();
        for b in pa_batches.try_iter()? {
            let b = b?;
            // pick the source column (named or first)
            let col = match &col_name {
                Some(name) => b.call_method1("column", (name.as_str(),))?,
                None => b.call_method1("column", (0i64,))?,
            };
            let values = col.call_method0("to_pylist")?;
            let mut results: Vec<Py<PyAny>> = Vec::new();
            for v in values.try_iter()? {
                results.push(fn_.call1((v?,))?.unbind());
            }
            let result_arr = pyarrow.call_method1("array", (PyList::new(py, results)?,))?;
            // append as output_column onto the input batch (row-preserving)
            let appended = b.call_method1("append_column", (output_column, result_arr))?;
            let one = PyList::new(py, [appended])?;
            out_tables.push(
                pyarrow
                    .getattr("Table")?
                    .call_method1("from_batches", (one,))?
                    .unbind(),
            );
        }
        let out_table = if out_tables.is_empty() {
            table.call_method0("schema")?; // no-op to keep types; fall through
            pyarrow.call_method0("table")?
        } else {
            let list = PyList::new(py, out_tables.iter().map(|t| t.bind(py)))?;
            pyarrow.call_method1("concat_tables", (list,))?
        };
        let batches = crate::arrow_ffi::py_arrow_to_batches(py, &out_table)?;
        let conn_ref = self.conn.borrow(py);
        Self::new_materialized(py, &conn_ref, batches)
    }

    /// Redistribute rows into `num_partitions` partitions. Without a DuckDB
    /// engine fork this is a logical hint consumed by the runner (a Repartition
    /// plan node); the row set is unchanged so downstream ops still work.
    #[pyo3(signature = (num_partitions, *partition_by))]
    fn repartition(
        &self,
        py: Python<'_>,
        num_partitions: usize,
        partition_by: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let by = exprs_to_sql_list(partition_by).unwrap_or_default();
        self.derive_plan(py, |input| LogicalPlan::Repartition {
            input,
            num_partitions,
            by,
        })
    }

    /// Local (in-process) exchange into `num_partitions`. Logical hint; identity
    /// on the row set.
    fn local_exchange(&self, py: Python<'_>, num_partitions: usize) -> PyResult<Self> {
        self.repartition(py, num_partitions, &PyList::empty(py))
    }

    #[getter]
    fn num_partitions(&self) -> usize {
        self.plan.partition_hint().unwrap_or(1)
    }

    // ---- AI convenience ----

    #[pyo3(signature = (column, provider=None, model=None, dimensions=None, output_column="embedding"))]
    fn embed_text(
        &self,
        py: Python<'_>,
        column: &str,
        provider: Option<&str>,
        model: Option<&str>,
        dimensions: Option<usize>,
        output_column: &str,
    ) -> PyResult<Self> {
        crate::ai::functions::embed_text(
            py,
            self,
            column,
            provider,
            model,
            dimensions,
            output_column,
        )
    }

    #[pyo3(signature = (column, labels, provider=None, model=None, output_column="label"))]
    fn classify_text(
        &self,
        py: Python<'_>,
        column: &str,
        labels: Vec<String>,
        provider: Option<&str>,
        model: Option<&str>,
        output_column: &str,
    ) -> PyResult<Self> {
        crate::ai::functions::classify_text(
            py,
            self,
            column,
            &labels,
            provider,
            model,
            output_column,
        )
    }

    #[pyo3(signature = (column, provider=None, model=None, system_message=None, output_column="response"))]
    fn prompt(
        &self,
        py: Python<'_>,
        column: &str,
        provider: Option<&str>,
        model: Option<&str>,
        system_message: Option<&str>,
        output_column: &str,
    ) -> PyResult<Self> {
        crate::ai::functions::prompt_relation(
            py,
            self,
            column,
            provider,
            model,
            system_message,
            output_column,
        )
    }
}

// ---- Helpers ----

/// Reassemble a list of per-batch UDF outputs (each a pyarrow Table or
/// RecordBatch) into a single pyarrow Table.
fn reassemble_table(py: Python<'_>, out_py: Vec<Py<PyAny>>) -> PyResult<Py<PyAny>> {
    let pyarrow = py.import("pyarrow")?;
    if out_py.is_empty() {
        return Ok(pyarrow.call_method0("table")?.into());
    }
    let table_cls = pyarrow.getattr("Table")?;
    let batch_cls = pyarrow.getattr("RecordBatch")?;
    // Normalize each output to a pyarrow.Table, then concat.
    let mut tables = Vec::with_capacity(out_py.len());
    for obj in &out_py {
        let b = obj.bind(py);
        let tbl = if b.is_instance(&table_cls)? {
            b.clone()
        } else if b.is_instance(&batch_cls)? {
            let list = PyList::new(py, [&b])?;
            table_cls.call_method1("from_batches", (list,))?
        } else {
            // Anything else convertible to a Table (dict, pandas, etc.).
            pyarrow.call_method1("table", (b,))?
        };
        tables.push(tbl.unbind());
    }
    if tables.len() == 1 {
        return Ok(tables.into_iter().next().unwrap());
    }
    let list = PyList::new(py, tables.iter().map(|t| t.bind(py)))?;
    Ok(pyarrow.call_method1("concat_tables", (list,))?.into())
}

/// Build a jude.exceptions.ParserException PyErr.
fn parser_exception(py: Python<'_>, msg: &str) -> PyErr {
    match py
        .import("jude.exceptions")
        .and_then(|m| m.getattr("ParserException"))
    {
        Ok(exc) => match exc.call1((msg.to_string(),)) {
            Ok(inst) => PyErr::from_value(inst),
            Err(e) => e,
        },
        Err(_) => pyo3::exceptions::PyValueError::new_err(msg.to_string()),
    }
}

/// Build a jude.exceptions PyErr of the named class (falls back to ValueError).
fn jude_exception(py: Python<'_>, class: &str, msg: &str) -> PyErr {
    match py.import("jude.exceptions").and_then(|m| m.getattr(class)) {
        Ok(exc) => match exc.call1((msg.to_string(),)) {
            Ok(inst) => PyErr::from_value(inst),
            Err(e) => e,
        },
        Err(_) => pyo3::exceptions::PyValueError::new_err(msg.to_string()),
    }
}

/// Strip a trailing `ORDER BY …` from a GROUP BY spec. DuckDB's relational
/// aggregate API accepts `groups="id order by t asc"`; the ordering does not
/// change grouped-aggregate content, so we group by the leading key list only.
fn strip_trailing_order_by(spec: &str) -> &str {
    let lower = spec.to_ascii_lowercase();
    match lower.find(" order by ") {
        Some(pos) => spec[..pos].trim(),
        None => spec.trim(),
    }
}

/// Map an Arrow DataType to the DuckDB SQL type name DuckDB's relational API
/// reports via `rel.types` (INTEGER, BIGINT, VARCHAR, …).
fn arrow_type_to_duckdb_name(dt: &arrow::datatypes::DataType) -> String {
    use arrow::datatypes::{DataType, TimeUnit};
    match dt {
        DataType::Boolean => "BOOLEAN".to_string(),
        DataType::Int8 => "TINYINT".to_string(),
        DataType::Int16 => "SMALLINT".to_string(),
        DataType::Int32 => "INTEGER".to_string(),
        DataType::Int64 => "BIGINT".to_string(),
        DataType::UInt8 => "UTINYINT".to_string(),
        DataType::UInt16 => "USMALLINT".to_string(),
        DataType::UInt32 => "UINTEGER".to_string(),
        DataType::UInt64 => "UBIGINT".to_string(),
        DataType::Float16 | DataType::Float32 => "FLOAT".to_string(),
        DataType::Float64 => "DOUBLE".to_string(),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => "VARCHAR".to_string(),
        DataType::Binary | DataType::LargeBinary | DataType::BinaryView => "BLOB".to_string(),
        DataType::Date32 | DataType::Date64 => "DATE".to_string(),
        DataType::Time32(_) | DataType::Time64(_) => "TIME".to_string(),
        DataType::Timestamp(_, Some(_)) => "TIMESTAMP WITH TIME ZONE".to_string(),
        DataType::Timestamp(unit, None) => match unit {
            TimeUnit::Second => "TIMESTAMP_S".to_string(),
            TimeUnit::Millisecond => "TIMESTAMP_MS".to_string(),
            TimeUnit::Microsecond => "TIMESTAMP".to_string(),
            TimeUnit::Nanosecond => "TIMESTAMP_NS".to_string(),
        },
        DataType::Interval(_) | DataType::Duration(_) => "INTERVAL".to_string(),
        DataType::Decimal128(p, s) | DataType::Decimal256(p, s) => format!("DECIMAL({p},{s})"),
        DataType::List(f) | DataType::LargeList(f) | DataType::ListView(f) => {
            format!("{}[]", arrow_type_to_duckdb_name(f.data_type()))
        }
        DataType::FixedSizeList(f, n) => {
            format!("{}[{n}]", arrow_type_to_duckdb_name(f.data_type()))
        }
        DataType::Struct(fields) => {
            let parts: Vec<String> = fields
                .iter()
                .map(|f| format!("{} {}", f.name(), arrow_type_to_duckdb_name(f.data_type())))
                .collect();
            format!("STRUCT({})", parts.join(", "))
        }
        DataType::Map(f, _) => {
            // Map entries are a struct<key, value>.
            if let DataType::Struct(kv) = f.data_type() {
                if kv.len() == 2 {
                    return format!(
                        "MAP({}, {})",
                        arrow_type_to_duckdb_name(kv[0].data_type()),
                        arrow_type_to_duckdb_name(kv[1].data_type())
                    );
                }
            }
            "MAP".to_string()
        }
        DataType::Dictionary(_, value) => arrow_type_to_duckdb_name(value),
        other => format!("{other:?}").to_uppercase(),
    }
}

/// Normalize a window spec into a trailing SQL `OVER (...)` clause.
///
/// DuckDB's relational window API takes the *full* over clause as a string,
/// e.g. `"over ()"` or `"over (partition by id order by t)"`. We accept that
/// form as-is (just concatenated after the function call), and also tolerate a
/// bare spec without the `over` keyword by wrapping it in `OVER (...)`.
fn render_over_clause(window_spec: &str) -> String {
    let trimmed = window_spec.trim();
    if trimmed.is_empty() {
        return "OVER ()".to_string();
    }
    if trimmed.len() >= 4 && trimmed[..4].eq_ignore_ascii_case("over") {
        // Already a complete OVER clause — use verbatim.
        trimmed.to_string()
    } else {
        format!("OVER ({trimmed})")
    }
}

/// Render a quantile argument: a scalar (int/float) → `0.5`, or a list/tuple of
/// numbers → `[0.2, 0.5]`. `None` uses the given default (DuckDB's median).
fn render_quantile_arg(obj: Option<&Bound<'_, PyAny>>, default: &str) -> PyResult<String> {
    let Some(o) = obj else {
        return Ok(default.to_string());
    };
    if o.is_none() {
        return Ok(default.to_string());
    }
    // A list/tuple of numbers → DuckDB list literal.
    if o.is_instance_of::<pyo3::types::PyList>() || o.is_instance_of::<pyo3::types::PyTuple>() {
        let mut parts = Vec::new();
        for item in o.try_iter()? {
            parts.push(item?.extract::<f64>()?.to_string());
        }
        return Ok(format!("[{}]", parts.join(", ")));
    }
    if let Ok(f) = o.extract::<f64>() {
        return Ok(f.to_string());
    }
    Ok(o.str()?.extract::<String>()?)
}

/// Coerce a Python value (str, jude Expression, or anything with `to_sql()`/
/// `__str__`) into a SQL fragment string.
fn expr_to_sql(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(s) = obj.extract::<String>() {
        return Ok(s);
    }
    if let Ok(expr) = obj.extract::<crate::expressions::Expression>() {
        return Ok(expr.render_sql());
    }
    if let Ok(m) = obj.getattr("to_sql") {
        if m.is_callable() {
            if let Ok(s) = m.call0()?.extract::<String>() {
                return Ok(s);
            }
        }
    }
    // Fall back to str().
    Ok(obj.str()?.extract::<String>()?)
}

/// Peel `Alias`/`Repartition` wrappers to reach the operator that defines a
/// relation's shape (both are row-set identity for routing purposes).
fn root_op_plan(plan: &LogicalPlan) -> &LogicalPlan {
    match plan {
        LogicalPlan::Alias { input, .. } | LogicalPlan::Repartition { input, .. } => {
            root_op_plan(input)
        }
        other => other,
    }
}

/// Extract equi-join key columns from a join condition, for hash-shuffle joins:
/// a bare key list ("i" / "i, j"), or "lhs.k = rhs.k [AND lhs.m = rhs.m …]".
/// None if the condition isn't a same-named equi-key (caller falls back).
fn extract_join_keys(condition: &str) -> Option<Vec<String>> {
    let c = condition.trim();
    if c.is_empty() {
        return None;
    }
    let is_bare = c.split(',').all(|p| {
        let p = p.trim();
        !p.is_empty()
            && p.chars()
                .next()
                .is_some_and(|ch| ch.is_ascii_alphabetic() || ch == '_')
            && p.chars().all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
    });
    if is_bare {
        return Some(c.split(',').map(|p| p.trim().to_string()).collect());
    }
    let norm = c.replace(" and ", " AND ");
    let mut keys = Vec::new();
    for part in norm.split(" AND ") {
        let (l, r) = part.split_once('=')?;
        let lname = l.trim().strip_prefix("lhs.")?.trim();
        let rname = r.trim().strip_prefix("rhs.")?.trim();
        if lname != rname
            || lname.is_empty()
            || !lname
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
        {
            return None;
        }
        keys.push(lname.to_string());
    }
    (!keys.is_empty()).then_some(keys)
}

/// The base-table name of a relation, if it is (an alias of) a plain table scan
/// — used by `insert`/`update` to know they may write. None for any derived
/// relation (projection, query, materialized, …).
fn base_table_name(plan: &LogicalPlan) -> Option<String> {
    match plan {
        LogicalPlan::Table { name } => Some(name.clone()),
        LogicalPlan::Alias { input, .. } => base_table_name(input),
        _ => None,
    }
}

/// Render a Python value as a SQL literal for INSERT/UPDATE: a jude Expression
/// renders via its SQL (so DefaultExpression -> DEFAULT), None -> NULL, bool ->
/// TRUE/FALSE, numbers as-is, and strings single-quoted.
fn literal_to_sql(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    if obj.is_none() {
        return Ok("NULL".to_string());
    }
    if let Ok(expr) = obj.extract::<crate::expressions::Expression>() {
        return Ok(expr.render_sql());
    }
    if obj.is_instance_of::<pyo3::types::PyBool>() {
        return Ok(if obj.extract::<bool>()? {
            "TRUE".to_string()
        } else {
            "FALSE".to_string()
        });
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(i.to_string());
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(f.to_string());
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(format!("'{}'", s.replace('\'', "''")));
    }
    // Duck-typed Expression via to_sql(), else str().
    if let Ok(m) = obj.getattr("to_sql") {
        if m.is_callable() {
            if let Ok(s) = m.call0()?.extract::<String>() {
                return Ok(s);
            }
        }
    }
    Ok(format!(
        "'{}'",
        obj.str()?.extract::<String>()?.replace('\'', "''")
    ))
}

/// Coerce a Python value that may be a single expression or a list/tuple of
/// expressions into a list of SQL fragments.
fn exprs_to_sql_list(obj: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    // A bare string is a single item (not iterated char-by-char).
    if obj.extract::<String>().is_ok() {
        return Ok(vec![expr_to_sql(obj)?]);
    }
    // A jude Expression is a single item.
    if obj.extract::<crate::expressions::Expression>().is_ok() {
        return Ok(vec![expr_to_sql(obj)?]);
    }
    // Otherwise try to iterate.
    if let Ok(iter) = obj.try_iter() {
        let mut out = Vec::new();
        for item in iter {
            out.push(expr_to_sql(&item?)?);
        }
        return Ok(out);
    }
    Ok(vec![expr_to_sql(obj)?])
}

/// DuckDB logical-type hints (UUID/BIT that Arrow erases) for a materialized
/// relation's columns, via DESCRIBE. Empty when nothing needs refining (no
/// string/binary column) or on any error — the common, zero-cost path.
fn relation_logical_hints(rel: &Relation, py: Python<'_>) -> Vec<LogicalHint> {
    let Ok(sql) = rel.to_subquery_sql(py) else {
        return Vec::new();
    };
    let cache = rel.cache.borrow();
    let Some((_, schema)) = cache.as_ref() else {
        return Vec::new();
    };
    let stringy = schema.fields().iter().any(|f| {
        matches!(
            f.data_type(),
            DataType::Utf8
                | DataType::LargeUtf8
                | DataType::Utf8View
                | DataType::Binary
                | DataType::LargeBinary
                | DataType::BinaryView
        )
    });
    if !stringy {
        return Vec::new();
    }
    let conn = rel.conn.borrow(py);
    match conn.logical_type_names(&sql) {
        Ok(names) => hints_from_type_names(&names, schema),
        Err(_) => Vec::new(),
    }
}

fn rows_to_py(
    batches: &[RecordBatch],
    py: Python<'_>,
    limit: Option<usize>,
    hints: &[LogicalHint],
) -> PyResult<Py<PyAny>> {
    let mut rows: Vec<Py<PyAny>> = Vec::new();
    let mut count = 0usize;
    for batch in batches {
        for row_idx in 0..batch.num_rows() {
            if let Some(l) = limit {
                if count >= l {
                    return Ok(PyList::new(py, rows)?.into());
                }
            }
            let mut row = Vec::with_capacity(batch.num_columns());
            for col_idx in 0..batch.num_columns() {
                let col = batch.column(col_idx);
                let v = if col.is_null(row_idx) {
                    py.None()
                } else {
                    array_value_to_py(col, row_idx, py)?
                };
                row.push(apply_hint(py, v, hints.get(col_idx).copied())?);
            }
            // DuckDB / DBAPI semantics: each row is a tuple.
            rows.push(pyo3::types::PyTuple::new(py, row)?.into());
            count += 1;
        }
    }
    Ok(PyList::new(py, rows)?.into())
}

/// Public: all rows as a list of tuples (used by Connection fetch methods).
pub fn all_rows_to_py(
    batches: &[RecordBatch],
    py: Python<'_>,
    limit: Option<usize>,
    hints: &[LogicalHint],
) -> PyResult<Py<PyAny>> {
    rows_to_py(batches, py, limit, hints)
}

/// Public: the first row as a tuple, or None if empty.
pub fn first_row_to_py(
    batches: &[RecordBatch],
    py: Python<'_>,
    hints: &[LogicalHint],
) -> PyResult<Py<PyAny>> {
    for batch in batches {
        if batch.num_rows() > 0 {
            let mut row = Vec::with_capacity(batch.num_columns());
            for col_idx in 0..batch.num_columns() {
                let col = batch.column(col_idx);
                let v = if col.is_null(0) {
                    py.None()
                } else {
                    array_value_to_py(col, 0, py)?
                };
                row.push(apply_hint(py, v, hints.get(col_idx).copied())?);
            }
            return Ok(pyo3::types::PyTuple::new(py, row)?.into());
        }
    }
    Ok(py.None())
}

/// A per-column refinement for DuckDB logical types that Arrow erases: UUID
/// arrives as Utf8 and BIT as Binary, indistinguishable from VARCHAR/BLOB
/// unless we consult DuckDB's own column types (via DESCRIBE). `Plain` means
/// the Arrow value already matches DuckDB — the overwhelmingly common case.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LogicalHint {
    Plain,
    Uuid,
    Bit,
}

fn apply_hint(py: Python<'_>, value: Py<PyAny>, hint: Option<LogicalHint>) -> PyResult<Py<PyAny>> {
    use pyo3::conversion::IntoPyObjectExt;
    match hint {
        None | Some(LogicalHint::Plain) => Ok(value),
        Some(_) if value.is_none(py) => Ok(value),
        Some(LogicalHint::Uuid) => {
            // Arrow gave the canonical hyphenated string; wrap in uuid.UUID.
            Ok(py
                .import("uuid")?
                .getattr("UUID")?
                .call1((value.bind(py),))?
                .into())
        }
        Some(LogicalHint::Bit) => {
            let bytes: Vec<u8> = value.bind(py).extract()?;
            decode_duckdb_bit(&bytes).into_py_any(py)
        }
    }
}

/// Decode DuckDB's BIT wire layout into a '0101…' bitstring. The leading byte is
/// the number of padding bits in the first data byte; the remaining bytes carry
/// the bits MSB-first, with those leading padding bits skipped.
fn decode_duckdb_bit(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        return String::new();
    }
    let padding = bytes[0] as usize;
    let mut s = String::with_capacity((bytes.len().saturating_sub(1)) * 8);
    for &b in &bytes[1..] {
        for i in (0..8).rev() {
            s.push(if (b >> i) & 1 == 1 { '1' } else { '0' });
        }
    }
    if padding <= s.len() {
        s[padding..].to_string()
    } else {
        s
    }
}

/// Map DuckDB DESCRIBE type names to hints, but only for columns whose Arrow
/// type is string/binary (the only kinds a UUID/BIT can hide behind), so the
/// common case costs nothing.
pub fn hints_from_type_names(names: &[String], schema: &SchemaRef) -> Vec<LogicalHint> {
    schema
        .fields()
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let stringy = matches!(
                f.data_type(),
                DataType::Utf8
                    | DataType::LargeUtf8
                    | DataType::Utf8View
                    | DataType::Binary
                    | DataType::LargeBinary
                    | DataType::BinaryView
            );
            if !stringy {
                return LogicalHint::Plain;
            }
            match names.get(i).map(|n| n.to_ascii_uppercase()) {
                Some(u) if u == "UUID" => LogicalHint::Uuid,
                Some(u) if u == "BIT" || u == "BITSTRING" => LogicalHint::Bit,
                _ => LogicalHint::Plain,
            }
        })
        .collect()
}

fn array_value_to_py(array: &Arc<dyn Array>, idx: usize, py: Python<'_>) -> PyResult<Py<PyAny>> {
    use pyo3::conversion::IntoPyObjectExt;
    match array.data_type() {
        DataType::Boolean => array
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap()
            .value(idx)
            .into_py_any(py),
        DataType::Int8 => array
            .as_any()
            .downcast_ref::<Int8Array>()
            .unwrap()
            .value(idx)
            .into_py_any(py),
        DataType::Int16 => array
            .as_any()
            .downcast_ref::<Int16Array>()
            .unwrap()
            .value(idx)
            .into_py_any(py),
        DataType::Int32 => array
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .value(idx)
            .into_py_any(py),
        DataType::Int64 => array
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(idx)
            .into_py_any(py),
        DataType::UInt8 => array
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap()
            .value(idx)
            .into_py_any(py),
        DataType::UInt16 => array
            .as_any()
            .downcast_ref::<UInt16Array>()
            .unwrap()
            .value(idx)
            .into_py_any(py),
        DataType::UInt32 => array
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .value(idx)
            .into_py_any(py),
        DataType::UInt64 => array
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .value(idx)
            .into_py_any(py),
        DataType::Float16 => (array
            .as_any()
            .downcast_ref::<Float16Array>()
            .unwrap()
            .value(idx)
            .to_f32())
        .into_py_any(py),
        DataType::Float32 => array
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .value(idx)
            .into_py_any(py),
        DataType::Float64 => array
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(idx)
            .into_py_any(py),
        DataType::Utf8 => array
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(idx)
            .into_py_any(py),
        DataType::LargeUtf8 => array
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .unwrap()
            .value(idx)
            .into_py_any(py),
        DataType::Utf8View => array
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap()
            .value(idx)
            .into_py_any(py),
        DataType::Binary => array
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap()
            .value(idx)
            .into_py_any(py),
        DataType::LargeBinary => array
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .unwrap()
            .value(idx)
            .into_py_any(py),
        DataType::BinaryView => array
            .as_any()
            .downcast_ref::<BinaryViewArray>()
            .unwrap()
            .value(idx)
            .into_py_any(py),
        DataType::FixedSizeBinary(_) => array
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .unwrap()
            .value(idx)
            .into_py_any(py),
        DataType::Decimal128(_, scale) => {
            let v = array
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .unwrap()
                .value(idx);
            // scale 0 (e.g. HUGEINT, which DuckDB exports as decimal128(38,0)) is
            // an exact integer; scale > 0 is a fractional decimal.
            if *scale == 0 {
                py.import("builtins")?
                    .getattr("int")?
                    .call1((v.to_string(),))?
                    .into_py_any(py)
            } else {
                decimal_to_py(py, &v.to_string(), *scale)
            }
        }
        DataType::Decimal256(_, scale) => {
            let v = array
                .as_any()
                .downcast_ref::<Decimal256Array>()
                .unwrap()
                .value(idx);
            if *scale == 0 {
                py.import("builtins")?
                    .getattr("int")?
                    .call1((v.to_string(),))?
                    .into_py_any(py)
            } else {
                decimal_to_py(py, &v.to_string(), *scale)
            }
        }
        DataType::Date32 => {
            let days = array
                .as_any()
                .downcast_ref::<Date32Array>()
                .unwrap()
                .value(idx);
            let epoch = py
                .import("datetime")?
                .getattr("date")?
                .call1((1970, 1, 1))?;
            let td = py
                .import("datetime")?
                .getattr("timedelta")?
                .call1((days,))?;
            Ok(epoch.call_method1("__add__", (td,))?.into())
        }
        DataType::Time32(unit) => {
            use arrow::datatypes::TimeUnit;
            let us: i64 = match unit {
                TimeUnit::Second => {
                    array
                        .as_any()
                        .downcast_ref::<Time32SecondArray>()
                        .unwrap()
                        .value(idx) as i64
                        * 1_000_000
                }
                _ => {
                    array
                        .as_any()
                        .downcast_ref::<Time32MillisecondArray>()
                        .unwrap()
                        .value(idx) as i64
                        * 1_000
                }
            };
            time_from_micros(py, us)
        }
        DataType::Time64(unit) => {
            use arrow::datatypes::TimeUnit;
            let us: i64 = match unit {
                TimeUnit::Nanosecond => {
                    array
                        .as_any()
                        .downcast_ref::<Time64NanosecondArray>()
                        .unwrap()
                        .value(idx)
                        / 1_000
                }
                _ => array
                    .as_any()
                    .downcast_ref::<Time64MicrosecondArray>()
                    .unwrap()
                    .value(idx),
            };
            time_from_micros(py, us)
        }
        DataType::Timestamp(_, _) => {
            // Represent as microseconds-based datetime via pyarrow scalar for correctness.
            timestamp_to_py(array, idx, py)
        }
        DataType::Interval(unit) => {
            // DuckDB INTERVAL -> datetime.timedelta, using DuckDB's 30-days-per-month
            // convention (matches duckdb-python: months*30 + days, sub-day precision).
            use arrow::datatypes::IntervalUnit;
            let timedelta = py.import("datetime")?.getattr("timedelta")?;
            let kw = pyo3::types::PyDict::new(py);
            match unit {
                IntervalUnit::MonthDayNano => {
                    let a = array
                        .as_any()
                        .downcast_ref::<arrow::array::IntervalMonthDayNanoArray>()
                        .unwrap();
                    let v = a.value(idx);
                    kw.set_item("days", v.months as i64 * 30 + v.days as i64)?;
                    kw.set_item("microseconds", v.nanoseconds / 1_000)?;
                }
                IntervalUnit::DayTime => {
                    let a = array
                        .as_any()
                        .downcast_ref::<arrow::array::IntervalDayTimeArray>()
                        .unwrap();
                    let v = a.value(idx);
                    kw.set_item("days", v.days as i64)?;
                    kw.set_item("milliseconds", v.milliseconds as i64)?;
                }
                IntervalUnit::YearMonth => {
                    let a = array
                        .as_any()
                        .downcast_ref::<arrow::array::IntervalYearMonthArray>()
                        .unwrap();
                    kw.set_item("days", a.value(idx) as i64 * 30)?;
                }
            }
            Ok(timedelta.call((), Some(&kw))?.into())
        }
        DataType::List(_) => {
            let list = array.as_any().downcast_ref::<ListArray>().unwrap();
            let child = list.value(idx);
            list_to_py(&child, py)
        }
        DataType::LargeList(_) => {
            let list = array.as_any().downcast_ref::<LargeListArray>().unwrap();
            let child = list.value(idx);
            list_to_py(&child, py)
        }
        DataType::FixedSizeList(_, _) => {
            // DuckDB's fixed-size ARRAY type returns a tuple (not a list).
            let list = array.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
            let child = list.value(idx);
            list_to_pytuple(&child, py)
        }
        DataType::Dictionary(_, _) => {
            // Enum columns arrive as a dictionary array; return the decoded value.
            let dict = array
                .as_any()
                .downcast_ref::<arrow::array::DictionaryArray<arrow::datatypes::UInt8Type>>();
            if let Some(d) = dict {
                let key = d.keys().value(idx) as usize;
                return array_value_to_py(d.values(), key, py);
            }
            // Fall back to pyarrow for wider index types.
            let dict16 = array
                .as_any()
                .downcast_ref::<arrow::array::DictionaryArray<arrow::datatypes::UInt16Type>>();
            if let Some(d) = dict16 {
                let key = d.keys().value(idx) as usize;
                return array_value_to_py(d.values(), key, py);
            }
            let dict32 = array
                .as_any()
                .downcast_ref::<arrow::array::DictionaryArray<arrow::datatypes::UInt32Type>>();
            if let Some(d) = dict32 {
                let key = d.keys().value(idx) as usize;
                return array_value_to_py(d.values(), key, py);
            }
            Ok(format!("{:?}", array.slice(idx, 1))
                .into_pyobject(py)?
                .into_any()
                .unbind())
        }
        DataType::Union(_, _) => {
            // A UNION row is the value of its active child.
            let u = array
                .as_any()
                .downcast_ref::<arrow::array::UnionArray>()
                .unwrap();
            let child = u.value(idx); // ArrayRef of length 1 (the active value)
            if child.is_null(0) {
                Ok(py.None())
            } else {
                array_value_to_py(&child, 0, py)
            }
        }
        DataType::Struct(fields) => {
            // A STRUCT row becomes a dict {field_name: value} (DuckDB semantics).
            let st = array.as_any().downcast_ref::<StructArray>().unwrap();
            let d = pyo3::types::PyDict::new(py);
            for (fi, field) in fields.iter().enumerate() {
                let col = st.column(fi);
                let v = if col.is_null(idx) {
                    py.None()
                } else {
                    array_value_to_py(col, idx, py)?
                };
                d.set_item(field.name(), v)?;
            }
            Ok(d.into())
        }
        DataType::Map(_, _) => {
            // A MAP row becomes a dict {key: value} (DuckDB semantics).
            let m = array.as_any().downcast_ref::<MapArray>().unwrap();
            let entries = m.value(idx); // a StructArray of {key, value}
            let st = entries.as_any().downcast_ref::<StructArray>().unwrap();
            let keys = st.column(0);
            let vals = st.column(1);
            let d = pyo3::types::PyDict::new(py);
            for r in 0..st.len() {
                let k = array_value_to_py(keys, r, py)?;
                let v = if vals.is_null(r) {
                    py.None()
                } else {
                    array_value_to_py(vals, r, py)?
                };
                d.set_item(k, v)?;
            }
            Ok(d.into())
        }
        _ => {
            // Fallback: stringify unknown types rather than silently returning None,
            // so nothing is lost.
            Ok(format!("{:?}", array.slice(idx, 1))
                .into_pyobject(py)?
                .into_any()
                .unbind())
        }
    }
}

/// Build a Python `datetime.time` from microseconds-since-midnight. Values at or
/// past 24:00:00 (which DuckDB permits) are clamped to 23:59:59.999999, matching
/// how DuckDB-Python surfaces the max TIME value.
fn time_from_micros(py: Python<'_>, mut us: i64) -> PyResult<Py<PyAny>> {
    const DAY_US: i64 = 24 * 3600 * 1_000_000;
    if us >= DAY_US {
        us = DAY_US - 1; // 23:59:59.999999
    }
    if us < 0 {
        us = 0;
    }
    let micro = (us % 1_000_000) as u32;
    let total_secs = us / 1_000_000;
    let sec = (total_secs % 60) as u32;
    let minute = ((total_secs / 60) % 60) as u32;
    let hour = (total_secs / 3600) as u32;
    let time_cls = py.import("datetime")?.getattr("time")?;
    Ok(time_cls.call1((hour, minute, sec, micro))?.into())
}

fn decimal_to_py(py: Python<'_>, unscaled: &str, scale: i8) -> PyResult<Py<PyAny>> {
    let decimal_cls = py.import("decimal")?.getattr("Decimal")?;
    // Build the fixed-point string directly rather than Decimal(unscaled).scaleb():
    // scaleb() rounds to the active decimal context precision (default 28 digits),
    // which silently truncates wide values like DECIMAL(38,10). Constructing the
    // Decimal from an already-pointed string never rounds, matching pyarrow.
    let s = format_scaled_decimal(unscaled, scale);
    Ok(decimal_cls.call1((s,))?.into())
}

/// Insert a decimal point into an unscaled integer string at `scale` places from
/// the right (e.g. "-99999999999" scale 4 -> "-9999999.9999"), padding with
/// leading zeros when the integer is shorter than the scale.
fn format_scaled_decimal(unscaled: &str, scale: i8) -> String {
    if scale <= 0 {
        return unscaled.to_string();
    }
    let scale = scale as usize;
    let (neg, digits) = match unscaled.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, unscaled.strip_prefix('+').unwrap_or(unscaled)),
    };
    let body = if digits.len() <= scale {
        format!("0.{}{}", "0".repeat(scale - digits.len()), digits)
    } else {
        let point = digits.len() - scale;
        format!("{}.{}", &digits[..point], &digits[point..])
    };
    if neg {
        format!("-{body}")
    } else {
        body
    }
}

fn list_to_py(child: &Arc<dyn Array>, py: Python<'_>) -> PyResult<Py<PyAny>> {
    let mut items = Vec::with_capacity(child.len());
    for i in 0..child.len() {
        items.push(if child.is_null(i) {
            py.None()
        } else {
            array_value_to_py(child, i, py)?
        });
    }
    Ok(PyList::new(py, items)?.into())
}

/// Like `list_to_py` but produces a Python tuple — DuckDB's fixed-size ARRAY
/// type surfaces as a tuple, not a list.
fn list_to_pytuple(child: &Arc<dyn Array>, py: Python<'_>) -> PyResult<Py<PyAny>> {
    let mut items = Vec::with_capacity(child.len());
    for i in 0..child.len() {
        items.push(if child.is_null(i) {
            py.None()
        } else {
            array_value_to_py(child, i, py)?
        });
    }
    Ok(pyo3::types::PyTuple::new(py, items)?.into())
}

fn timestamp_to_py(array: &Arc<dyn Array>, idx: usize, py: Python<'_>) -> PyResult<Py<PyAny>> {
    // Extract raw i64 and unit; build a naive datetime.
    let (value, unit) = match array.data_type() {
        DataType::Timestamp(u, _) => (
            array
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .map(|a| a.value(idx))
                .or_else(|| {
                    array
                        .as_any()
                        .downcast_ref::<TimestampNanosecondArray>()
                        .map(|a| a.value(idx))
                })
                .or_else(|| {
                    array
                        .as_any()
                        .downcast_ref::<TimestampMillisecondArray>()
                        .map(|a| a.value(idx))
                })
                .or_else(|| {
                    array
                        .as_any()
                        .downcast_ref::<TimestampSecondArray>()
                        .map(|a| a.value(idx))
                }),
            u.clone(),
        ),
        _ => (None, duckdb::arrow::datatypes::TimeUnit::Microsecond),
    };
    let Some(v) = value else {
        return Ok(py.None());
    };
    let (secs, micros) = match unit {
        duckdb::arrow::datatypes::TimeUnit::Second => (v, 0i64),
        duckdb::arrow::datatypes::TimeUnit::Millisecond => {
            (v.div_euclid(1_000), v.rem_euclid(1_000) * 1_000)
        }
        duckdb::arrow::datatypes::TimeUnit::Microsecond => {
            (v.div_euclid(1_000_000), v.rem_euclid(1_000_000))
        }
        duckdb::arrow::datatypes::TimeUnit::Nanosecond => (
            v.div_euclid(1_000_000_000),
            v.rem_euclid(1_000_000_000) / 1_000,
        ),
    };
    let datetime = py.import("datetime")?.getattr("datetime")?;
    let epoch = datetime.call1((1970, 1, 1))?;
    let timedelta = py.import("datetime")?.getattr("timedelta")?;
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("seconds", secs)?;
    kwargs.set_item("microseconds", micros)?;
    let delta = timedelta.call((), Some(&kwargs))?;
    let naive = epoch.call_method1("__add__", (delta,))?;
    // TIMESTAMP is naive; TIMESTAMPTZ arrives as UTC-normalized micros with a tz
    // in the Arrow type, so surface an aware datetime (tzinfo=UTC).
    let has_tz = matches!(array.data_type(), DataType::Timestamp(_, Some(_)));
    if has_tz {
        let utc = py.import("datetime")?.getattr("timezone")?.getattr("utc")?;
        let kw = pyo3::types::PyDict::new(py);
        kw.set_item("tzinfo", utc)?;
        return Ok(naive.call_method("replace", (), Some(&kw))?.into());
    }
    Ok(naive.into())
}

fn print_batch(batch: &RecordBatch) {
    let schema = batch.schema();
    let col_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    println!("{}", col_names.join(" | "));
    println!(
        "{}",
        col_names
            .iter()
            .map(|_| "---")
            .collect::<Vec<_>>()
            .join("-+-")
    );
    for row_idx in 0..batch.num_rows() {
        let mut row_vals = Vec::new();
        for col_idx in 0..batch.num_columns() {
            let col = batch.column(col_idx);
            let val = if col.is_null(row_idx) {
                "NULL".to_string()
            } else {
                format_array_value(col, row_idx)
            };
            row_vals.push(val);
        }
        println!("{}", row_vals.join(" | "));
    }
}

fn format_array_value(array: &Arc<dyn Array>, idx: usize) -> String {
    match array.data_type() {
        DataType::Boolean => array
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap()
            .value(idx)
            .to_string(),
        DataType::Int32 => array
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .value(idx)
            .to_string(),
        DataType::Int64 => array
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(idx)
            .to_string(),
        DataType::Float64 => array
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(idx)
            .to_string(),
        DataType::Float32 => array
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .value(idx)
            .to_string(),
        DataType::Utf8 => array
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(idx)
            .to_string(),
        dt => format!("[{:?}]", dt),
    }
}
