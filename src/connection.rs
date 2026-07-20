use crate::error::Error;
use crate::relation::{LogicalHint, Relation};
use duckdb::arrow::datatypes::SchemaRef;
use duckdb::arrow::record_batch::RecordBatch;
use duckdb::vtab::arrow::ArrowVTab;
use duckdb::Connection as DuckConnection;
use pyo3::prelude::*;
use std::cell::RefCell;
use std::sync::Arc;

/// A shareable DuckDB connection handle.
///
/// The inner `DuckConnection` is wrapped in an `Arc` so that lazy `Relation`s can
/// hold a back-reference and re-execute their SQL against the same database. All
/// access happens under the Python GIL (the pyclass is `unsendable`), so no extra
/// locking is required — `&DuckConnection` methods only need `&self`.
#[pyclass(name = "Connection", unsendable)]
pub struct Connection {
    pub inner: Arc<DuckConnection>,
    /// Result of the most recent `execute`, for DBAPI-style `conn.execute(...).fetchone()`.
    last_result: RefCell<Option<(Vec<RecordBatch>, SchemaRef)>>,
    /// Per-column DuckDB logical-type hints for `last_result` (UUID/BIT that
    /// Arrow erases to string/binary). Empty = every column is Plain.
    last_hints: RefCell<Vec<LogicalHint>>,
    /// Replacement-scan toggles (DuckDB `SET python_enable_replacements` /
    /// `SET python_scan_all_frames`). Default: enabled, single caller frame.
    enable_replacements: RefCell<bool>,
    scan_all_frames: RefCell<bool>,
}

impl Connection {
    pub fn connect(path: Option<&str>) -> PyResult<Self> {
        let conn = match path {
            None | Some(":memory:") | Some("") => DuckConnection::open_in_memory(),
            Some(p) => DuckConnection::open(p),
        }
        .map_err(Error::DuckDb)?;
        // Register the Arrow table function once so we can ingest RecordBatches
        // zero-copy via `arrow(?, ?)` instead of round-tripping through parquet.
        conn.register_table_function::<ArrowVTab>("arrow")
            .map_err(Error::DuckDb)?;
        // Re-scannable materialized-boundary scan: `jude_scan(<id>)` streams a
        // registered Arc<Vec<RecordBatch>> without a temp-table copy.
        conn.register_table_function::<crate::mat_scan::MaterializedScanVTab>("jude_scan")
            .map_err(Error::DuckDb)?;
        // Auto-install/load known DuckDB extensions on first use, so vector
        // search (vss/HNSW), full-text search (fts), json, icu, etc. work in
        // `jude.sql(...)` without a manual INSTALL/LOAD. Best-effort: ignore
        // failures (e.g. offline) so a plain in-memory connection still opens.
        let _ = conn.execute_batch(
            "SET autoinstall_known_extensions=true; SET autoload_known_extensions=true;",
        );
        Ok(Self {
            inner: Arc::new(conn),
            last_result: RefCell::new(None),
            last_hints: RefCell::new(Vec::new()),
            enable_replacements: RefCell::new(true),
            scan_all_frames: RefCell::new(false),
        })
    }

    /// Wrap an existing shared connection handle (same underlying database).
    pub fn from_arc(inner: Arc<DuckConnection>) -> Self {
        Self {
            inner,
            last_result: RefCell::new(None),
            last_hints: RefCell::new(Vec::new()),
            enable_replacements: RefCell::new(true),
            scan_all_frames: RefCell::new(false),
        }
    }

    /// Clone the shared connection handle (same underlying database).
    pub fn share(&self) -> Arc<DuckConnection> {
        self.inner.clone()
    }

    /// Execute a query and collect all result batches eagerly.
    pub fn run_sql(&self, query: &str) -> Result<Vec<RecordBatch>, Error> {
        let mut stmt = self.inner.prepare(query).map_err(Error::DuckDb)?;
        let batches: Vec<RecordBatch> = stmt.query_arrow([]).map_err(Error::DuckDb)?.collect();
        Ok(batches)
    }

    /// Best-effort replacement scan: register in-scope pandas/polars/Arrow
    /// variables named in the query as temp views. Failures are swallowed so a
    /// normal query with no such variables is unaffected.
    ///
    /// `depth` is how many Python frames above the `register_scan_candidates`
    /// Python function the user's frame sits (2: this Rust frame adds none, but
    /// the sql()/execute() pymethod is one and the user is above it — resolved
    /// on the Python side by walking from the given depth).
    fn register_scan_candidates(&self, py: Python<'_>, query: &str) {
        if !*self.enable_replacements.borrow() {
            return;
        }
        let all_frames = *self.scan_all_frames.borrow();
        let result: PyResult<()> = (|| {
            let module = py.import("jude._replacement")?;
            let conn_obj = Py::new(py, Connection::from_arc(self.inner.clone()))?;
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("all_frames", all_frames)?;
            module.call_method("register_scan_candidates", (conn_obj, query), Some(&kwargs))?;
            Ok(())
        })();
        let _ = result; // best-effort
    }

    /// Intercept DuckDB-Python-only config SETs (`python_enable_replacements`,
    /// `python_scan_all_frames`) that our stock DuckDB doesn't know. Returns
    /// true if `sql` was such a statement (and was applied to our flags), so the
    /// caller can skip handing it to DuckDB.
    fn maybe_intercept_config(&self, sql: &str) -> bool {
        let s = sql.trim().trim_end_matches(';').to_ascii_lowercase();
        let parse_bool = |s: &str| -> Option<bool> {
            let v = s.rsplit('=').next()?.trim();
            match v {
                "true" | "1" => Some(true),
                "false" | "0" => Some(false),
                _ => None,
            }
        };
        if s.starts_with("set python_enable_replacements")
            || s.starts_with("set global python_enable_replacements")
        {
            if let Some(b) = parse_bool(&s) {
                *self.enable_replacements.borrow_mut() = b;
            }
            true
        } else if s.starts_with("set python_scan_all_frames")
            || s.starts_with("set global python_scan_all_frames")
        {
            if let Some(b) = parse_bool(&s) {
                *self.scan_all_frames.borrow_mut() = b;
            }
            true
        } else {
            false
        }
    }

    /// Execute a query and return both the result batches and the result schema.
    ///
    /// The schema is read from the Arrow stream, so it is correct even when the
    /// query returns zero rows (unlike inferring it from the first batch).
    pub fn run_sql_with_schema(
        &self,
        query: &str,
    ) -> Result<(Vec<RecordBatch>, duckdb::arrow::datatypes::SchemaRef), Error> {
        let mut stmt = self.inner.prepare(query).map_err(Error::DuckDb)?;
        let arrow = stmt.query_arrow([]).map_err(Error::DuckDb)?;
        let schema = arrow.get_schema();
        let batches: Vec<RecordBatch> = arrow.collect();
        Ok((batches, schema))
    }

    /// Return only the schema of a query (executes with an implicit LIMIT-free
    /// prepare; the Arrow stream exposes the schema before rows are pulled).
    pub fn schema_of(&self, query: &str) -> Result<duckdb::arrow::datatypes::SchemaRef, Error> {
        let mut stmt = self.inner.prepare(query).map_err(Error::DuckDb)?;
        let arrow = stmt.query_arrow([]).map_err(Error::DuckDb)?;
        Ok(arrow.get_schema())
    }

    /// Execute a parameterized query and return batches + schema.
    pub fn run_sql_with_schema_params(
        &self,
        query: &str,
        params: &[&dyn duckdb::ToSql],
    ) -> Result<(Vec<RecordBatch>, SchemaRef), Error> {
        let mut stmt = self.inner.prepare(query).map_err(Error::DuckDb)?;
        let arrow = stmt.query_arrow(params).map_err(Error::DuckDb)?;
        let schema = arrow.get_schema();
        let batches: Vec<RecordBatch> = arrow.collect();
        Ok((batches, schema))
    }

    /// DuckDB logical type names for each output column of `query`, via DESCRIBE.
    /// These carry information Arrow erases — notably UUID (arrow: string) and
    /// BIT (arrow: binary). Wrapped in a subquery so any SELECT is describable.
    pub fn logical_type_names(&self, query: &str) -> Result<Vec<String>, Error> {
        use duckdb::arrow::array::{Array, StringArray};
        let describe = format!("DESCRIBE SELECT * FROM ({query}) AS _jt_describe");
        let mut stmt = self.inner.prepare(&describe).map_err(Error::DuckDb)?;
        let arrow = stmt.query_arrow([]).map_err(Error::DuckDb)?;
        let batches: Vec<RecordBatch> = arrow.collect();
        let mut out = Vec::new();
        for batch in &batches {
            if let Some(col) = batch.column_by_name("column_type") {
                if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
                    for i in 0..a.len() {
                        out.push(a.value(i).to_string());
                    }
                }
            }
        }
        Ok(out)
    }

    /// Compute per-column logical hints for a result, skipping the DESCRIBE
    /// round-trip entirely unless some column is string/binary (the only kinds a
    /// UUID/BIT can hide behind). Best-effort: on any error, no hints (all Plain).
    fn compute_hints(&self, query: &str, schema: &SchemaRef) -> Vec<LogicalHint> {
        use duckdb::arrow::datatypes::DataType;
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
        match self.logical_type_names(query) {
            Ok(names) => crate::relation::hints_from_type_names(&names, schema),
            Err(_) => Vec::new(),
        }
    }

    /// Build a lazy (or eagerly-executed for DDL) relation from a SQL string,
    /// with no bound parameters. This is the internal path all the readers use;
    /// the `sql` pymethod adds parameter binding on top.
    pub fn sql_lazy(&self, py: Python<'_>, query: &str) -> PyResult<Relation> {
        if self.maybe_intercept_config(query) {
            // DuckDB-Python-only config toggle: applied to our flags; return an
            // empty relation so callers that materialize get a no-op result.
            return Relation::new_lazy_sql(py, self, "SELECT NULL WHERE FALSE".to_string());
        }
        // A side-effecting statement that returns no result set (CREATE/INSERT/
        // UPDATE/DELETE/DROP/ALTER/ATTACH/SET/…) is executed eagerly, matching
        // DuckDB where `con.sql(ddl)` runs immediately. A lazy relation would
        // never run the DDL (nothing materializes it), so a following
        // `con.table(...)` would not see it.
        if is_eager_statement(query) {
            // Register any in-scope df/arrow vars first (a CREATE TABLE AS
            // SELECT * FROM <df> needs the replacement scan), then run the DDL.
            self.register_scan_candidates(py, query);
            self.inner.execute_batch(query).map_err(Error::DuckDb)?;
            return Relation::new_lazy_sql(py, self, "SELECT NULL WHERE FALSE".to_string());
        }
        self.register_scan_candidates(py, query);
        Relation::new_lazy_sql(py, self, query.to_string())
    }
}

#[pymethods]
impl Connection {
    #[new]
    #[pyo3(signature = (path=None))]
    fn new(path: Option<&str>) -> PyResult<Self> {
        Self::connect(path)
    }

    /// Build a lazy relation from a SQL query. Nothing executes until the
    /// relation is materialized (fetch/num_rows/to_arrow/show/…).
    ///
    /// Before building the plan we run a best-effort *replacement scan*: any
    /// in-scope pandas/polars/Arrow variable referenced by name in the query is
    /// registered as a temp view (DuckDB/Vane semantics).
    #[pyo3(signature = (query, params=None))]
    pub fn sql(
        &self,
        py: Python<'_>,
        query: &str,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Relation> {
        // DuckDB: `sql(query, params=…)` binds the parameters and returns a
        // *materialized* relation (the bound result is executed eagerly, since a
        // lazy relation can't carry positional params through re-execution).
        if let Some(p) = params {
            if !p.is_none() {
                self.register_scan_candidates(py, query);
                let params_vec = py_params_to_vec(p)?;
                let params_ref: Vec<&dyn duckdb::ToSql> =
                    params_vec.iter().map(|v| v as &dyn duckdb::ToSql).collect();
                let (batches, _) = self
                    .run_sql_with_schema_params(query, params_ref.as_slice())
                    .map_err(PyErr::from)?;
                return Relation::new_materialized(py, self, batches);
            }
        }
        self.sql_lazy(py, query)
    }

    /// Alias for `sql`, matching DuckDB's `query` entry point.
    fn query(&self, py: Python<'_>, query: &str) -> PyResult<Relation> {
        self.sql_lazy(py, query)
    }

    #[pyo3(signature = (sql, params=None))]
    fn execute(
        slf: PyRef<'_, Self>,
        sql: &str,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<Self>> {
        // Run the statement and stash its result so DBAPI-style
        // `conn.execute(...).fetchone()` works. Statements without a result set
        // (CREATE/INSERT/…) leave an empty result.
        let this = &*slf;
        // Intercept DuckDB-Python-only config SETs before touching DuckDB.
        if this.maybe_intercept_config(sql) {
            *this.last_result.borrow_mut() = None;
            return Ok(slf.into());
        }
        // Replacement scan: register any in-scope pandas/polars/Arrow variable
        // referenced by name (DuckDB semantics — `execute` supports it too).
        this.register_scan_candidates(slf.py(), sql);
        let (batches, schema) = match params {
            None => this.run_sql_with_schema(sql),
            Some(p) => {
                let params_vec = py_params_to_vec(p)?;
                let params_ref: Vec<&dyn duckdb::ToSql> =
                    params_vec.iter().map(|v| v as &dyn duckdb::ToSql).collect();
                this.run_sql_with_schema_params(sql, params_ref.as_slice())
            }
        }
        .map_err(PyErr::from)?;
        *this.last_hints.borrow_mut() = this.compute_hints(sql, &schema);
        *this.last_result.borrow_mut() = Some((batches, schema));
        Ok(slf.into())
    }

    /// Fetch one row from the last `execute` result (or None).
    fn fetchone(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let cache = self.last_result.borrow();
        let Some((batches, _)) = cache.as_ref() else {
            return Ok(py.None());
        };
        crate::relation::first_row_to_py(batches, py, &self.last_hints.borrow())
    }

    /// Fetch all rows from the last `execute` result.
    fn fetchall(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let cache = self.last_result.borrow();
        let Some((batches, _)) = cache.as_ref() else {
            return Ok(pyo3::types::PyList::empty(py).into());
        };
        crate::relation::all_rows_to_py(batches, py, None, &self.last_hints.borrow())
    }

    #[pyo3(signature = (size=1))]
    fn fetchmany(&self, py: Python<'_>, size: usize) -> PyResult<Py<PyAny>> {
        let cache = self.last_result.borrow();
        let Some((batches, _)) = cache.as_ref() else {
            return Ok(pyo3::types::PyList::empty(py).into());
        };
        crate::relation::all_rows_to_py(batches, py, Some(size), &self.last_hints.borrow())
    }

    fn fetchdf(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let table = self.fetch_arrow_table(py)?;
        Ok(table.bind(py).call_method0("to_pandas")?.into())
    }

    /// DuckDB alias: `conn.execute(...).df()` returns the result as a pandas
    /// DataFrame.
    fn df(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.fetchdf(py)
    }

    /// DuckDB alias: `conn.execute(...).to_arrow_table()` (mirrors
    /// `fetch_arrow_table`).
    fn to_arrow_table(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.fetch_arrow_table(py)
    }

    /// DuckDB alias for `fetchdf`.
    fn fetch_df(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.fetchdf(py)
    }

    fn fetch_arrow_table(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let cache = self.last_result.borrow();
        let Some((batches, schema)) = cache.as_ref() else {
            let pyarrow = py.import("pyarrow")?;
            return Ok(pyarrow.call_method0("table")?.into());
        };
        crate::arrow_ffi::batches_to_pyarrow_table(py, batches, schema)
    }

    fn execute_batch(&self, sql: &str) -> PyResult<()> {
        self.inner
            .execute_batch(sql)
            .map_err(|e| Error::DuckDb(e).into())
    }

    #[pyo3(signature = (sql, parameters))]
    fn executemany(&self, sql: &str, parameters: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = parameters.py();
        if let Ok(outer) = parameters.extract::<Vec<Py<PyAny>>>() {
            for inner_py in &outer {
                let inner = inner_py.bind(py);
                let inner_params = py_params_to_vec(inner)?;
                let params_ref: Vec<&dyn duckdb::ToSql> = inner_params
                    .iter()
                    .map(|v| v as &dyn duckdb::ToSql)
                    .collect();
                self.inner
                    .execute(sql, params_ref.as_slice())
                    .map_err(Error::DuckDb)?;
            }
        }
        Ok(())
    }

    fn close(&self) -> PyResult<()> {
        Ok(())
    }

    fn begin(&self) -> PyResult<()> {
        self.inner
            .execute_batch("BEGIN")
            .map_err(|e| Error::DuckDb(e).into())
    }

    fn commit(&self) -> PyResult<()> {
        self.inner
            .execute_batch("COMMIT")
            .map_err(|e| Error::DuckDb(e).into())
    }

    fn rollback(&self) -> PyResult<()> {
        self.inner
            .execute_batch("ROLLBACK")
            .map_err(|e| Error::DuckDb(e).into())
    }

    // ---- File reading ----

    /// DuckDB-compatible `read_csv(path_or_file, **options)`. Accepts a path
    /// string, `pathlib.Path`/os.PathLike, or a list of paths, plus DuckDB's
    /// full CSV option surface (pandas-style aliases mapped to DuckDB option
    /// names). Options are rendered into the `read_csv(...)` table function.
    #[pyo3(signature = (path_or_buffer, **kwargs))]
    pub fn read_csv(
        &self,
        py: Python<'_>,
        path_or_buffer: &Bound<'_, PyAny>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Relation> {
        // DuckDB-Python's read_csv rejects the raw engine option `quote` (the
        // Python surface exposes it as `quotechar`).
        if let Some(kw) = kwargs {
            if kw.contains("quote")? {
                return Err(invalid_input(
                    py,
                    "The methods read_csv and read_csv_auto do not have the \"quote\" argument, use \"quotechar\" instead",
                ));
            }
            // `delimiter` and `sep` are aliases; specifying both is an error.
            if kw.contains("delimiter")? && kw.contains("sep")? {
                return Err(invalid_input(
                    py,
                    "read_csv takes either 'delimiter' or 'sep', not both",
                ));
            }
            // `names` must be a list of strings.
            if let Some(names) = kw.get_item("names")? {
                if !names.is_none() {
                    let ok = (names.is_instance_of::<pyo3::types::PyList>()
                        || names.is_instance_of::<pyo3::types::PyTuple>())
                        && names
                            .try_iter()?
                            .all(|i| i.map(|x| x.extract::<String>().is_ok()).unwrap_or(false));
                    if !ok {
                        return Err(invalid_input(
                            py,
                            "read_csv only accepts 'names' as a list of strings",
                        ));
                    }
                }
            }
        }
        // A file-like object (has .read) is read into memory and scanned via a
        // temp file — matching DuckDB-Python's MemoryFileSystem behavior (the
        // bytes are pulled once with .read). A non-file-like, non-path object
        // (int, None, …) is a TypeError.
        let is_str = path_or_buffer.extract::<String>().is_ok();
        let is_pathlike = path_or_buffer.hasattr("__fspath__").unwrap_or(false);
        let is_seq = path_or_buffer.is_instance_of::<pyo3::types::PyList>()
            || path_or_buffer.is_instance_of::<pyo3::types::PyTuple>();
        if !is_str && !is_pathlike && !is_seq {
            let has_read = path_or_buffer.hasattr("read").unwrap_or(false);
            if !has_read {
                return Err(py_type_error("Can not read from a non file-like object"));
            }
            // Pull the whole content via .read() (propagating the object's own
            // errors, e.g. a ReadError raising ValueError), stage to a temp file.
            let content = path_or_buffer.call_method0("read")?;
            let bytes: Vec<u8> = if let Ok(b) = content.extract::<Vec<u8>>() {
                b
            } else {
                content.extract::<String>()?.into_bytes()
            };
            let tmp = std::env::temp_dir()
                .join(format!("jude_csv_{}.csv", uuid::Uuid::new_v4().simple()));
            std::fs::write(&tmp, &bytes).map_err(Error::Io)?;
            let opts = csv_kwargs_to_sql(kwargs)?;
            let src = format!("'{}'", escape_sql_string(&tmp.to_string_lossy()));
            let sql = if opts.is_empty() {
                format!("SELECT * FROM read_csv({src})")
            } else {
                format!("SELECT * FROM read_csv({src}, {opts})")
            };
            return self.sql_lazy(py, &sql);
        }
        // An empty list of paths is rejected before building any SQL.
        if is_seq && path_or_buffer.len().unwrap_or(0) == 0 {
            return Err(invalid_input(
                py,
                "Please provide a non-empty list of paths or file-like objects",
            ));
        }
        let src = csv_source_sql(path_or_buffer)?;
        let opts = csv_kwargs_to_sql(kwargs)?;
        let sql = if opts.is_empty() {
            format!("SELECT * FROM read_csv({src})")
        } else {
            format!("SELECT * FROM read_csv({src}, {opts})")
        };
        self.sql_lazy(py, &sql)
    }

    pub fn read_json(&self, py: Python<'_>, path: &str) -> PyResult<Relation> {
        self.sql_lazy(
            py,
            &format!("SELECT * FROM read_json('{}')", escape_sql_string(path)),
        )
    }

    pub fn read_parquet(&self, py: Python<'_>, glob: &str) -> PyResult<Relation> {
        self.sql_lazy(
            py,
            &format!("SELECT * FROM read_parquet('{}')", escape_sql_string(glob)),
        )
    }

    /// Read a Hive-partitioned dataset (key=value/ directory layout): partition
    /// columns are derived from the paths. `glob` should match the leaf files
    /// (e.g. '/warehouse/tbl/**/*.parquet'). `union_by_name` aligns schemas that
    /// differ across files. Filters on partition columns prune directories.
    #[pyo3(signature = (glob, hive_partitioning=true, union_by_name=false))]
    pub fn read_hive(
        &self,
        py: Python<'_>,
        glob: &str,
        hive_partitioning: bool,
        union_by_name: bool,
    ) -> PyResult<Relation> {
        let mut opts = format!(
            "hive_partitioning={}",
            if hive_partitioning { "true" } else { "false" }
        );
        if union_by_name {
            opts.push_str(", union_by_name=true");
        }
        self.sql_lazy(
            py,
            &format!(
                "SELECT * FROM read_parquet('{}', {opts})",
                escape_sql_string(glob)
            ),
        )
    }

    /// Read an Apache Iceberg table into a relation via DuckDB's `iceberg`
    /// extension. `snapshot_id` selects a historical snapshot (time travel);
    /// `version` a metadata version. The extension is loaded on demand.
    #[pyo3(signature = (path, snapshot_id=None, version=None))]
    pub fn read_iceberg(
        &self,
        py: Python<'_>,
        path: &str,
        snapshot_id: Option<i64>,
        version: Option<String>,
    ) -> PyResult<Relation> {
        // Load the extension (idempotent; ignore "already loaded").
        let _ = self.inner.execute_batch("INSTALL iceberg; LOAD iceberg;");
        let p = escape_sql_string(path);
        let mut opts = String::new();
        if let Some(sid) = snapshot_id {
            opts.push_str(&format!(", snapshot_from_id => {sid}"));
        }
        if let Some(v) = version {
            opts.push_str(&format!(", version => '{}'", escape_sql_string(&v)));
        }
        self.sql_lazy(py, &format!("SELECT * FROM iceberg_scan('{p}'{opts})"))
    }

    /// List an Iceberg table's snapshots (id, timestamp, manifest) — time-travel
    /// discovery. Backed by DuckDB's `iceberg_snapshots`.
    pub fn iceberg_snapshots(&self, py: Python<'_>, path: &str) -> PyResult<Relation> {
        let _ = self.inner.execute_batch("INSTALL iceberg; LOAD iceberg;");
        self.sql_lazy(
            py,
            &format!(
                "SELECT * FROM iceberg_snapshots('{}')",
                escape_sql_string(path)
            ),
        )
    }

    /// Read a Lance dataset into a relation. Lance is not a DuckDB extension, so
    /// this scans via pylance (Rust-backed) into Arrow and materializes; `columns`
    /// projects and `filter` is a Lance filter expression (both pushed into the
    /// Lance scan). See `docs/storage_design.zh.md`.
    #[pyo3(signature = (path, columns=None, filter=None, version=None))]
    pub fn read_lance(
        &self,
        py: Python<'_>,
        path: &str,
        columns: Option<&Bound<'_, PyAny>>,
        filter: Option<&str>,
        version: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Relation> {
        let helper = py.import("jude._lance")?;
        let cols = columns.map(|c| c.clone().unbind());
        let ver = version.map(|v| v.clone().unbind());
        let table = helper.call_method1("read_table", (path, cols, filter, ver))?;
        let batches = crate::arrow_ffi::py_arrow_to_batches(py, &table)?;
        Relation::new_materialized(py, self, batches)
    }

    /// Version history of a Lance dataset (git log): one row per committed
    /// version. Read a past version via `read_lance(path, version=…)`.
    pub fn lance_versions(&self, py: Python<'_>, path: &str) -> PyResult<Relation> {
        let helper = py.import("jude._lance")?;
        let table = helper.call_method1("list_versions", (path,))?;
        let batches = crate::arrow_ffi::py_arrow_to_batches(py, &table)?;
        Relation::new_materialized(py, self, batches)
    }

    /// Name a Lance version (git tag), readable via `read_lance(path, version=tag)`.
    pub fn lance_tag(
        &self,
        py: Python<'_>,
        path: &str,
        tag: &str,
        version: i64,
    ) -> PyResult<Py<PyAny>> {
        let helper = py.import("jude._lance")?;
        Ok(helper
            .call_method1("create_tag", (path, tag, version))?
            .into())
    }

    /// List a Lance dataset's tags (name -> version).
    pub fn lance_tags(&self, py: Python<'_>, path: &str) -> PyResult<Relation> {
        let helper = py.import("jude._lance")?;
        let table = helper.call_method1("list_tags", (path,))?;
        let batches = crate::arrow_ffi::py_arrow_to_batches(py, &table)?;
        Relation::new_materialized(py, self, batches)
    }

    /// Roll back a Lance dataset to a past `version` (int or tag) — a new commit
    /// that restores the old state (history preserved, git-revert style).
    pub fn lance_restore(
        &self,
        py: Python<'_>,
        path: &str,
        version: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let helper = py.import("jude._lance")?;
        Ok(helper.call_method1("restore", (path, version))?.into())
    }

    /// Build an ANN vector index on a Lance embedding column (IVF_PQ / HNSW /…),
    /// so `lance_vector_search` runs approximate nearest-neighbour. Fills the
    /// vector-search gap neither stock DuckDB nor Vane has natively.
    #[pyo3(signature = (path, column, index_type="IVF_PQ", metric="L2", num_partitions=None, num_sub_vectors=None, replace=true))]
    #[allow(clippy::too_many_arguments)]
    pub fn create_lance_vector_index(
        &self,
        py: Python<'_>,
        path: &str,
        column: &str,
        index_type: &str,
        metric: &str,
        num_partitions: Option<i64>,
        num_sub_vectors: Option<i64>,
        replace: bool,
    ) -> PyResult<Py<PyAny>> {
        let helper = py.import("jude._lance")?;
        Ok(helper
            .call_method1(
                "create_vector_index",
                (
                    path,
                    column,
                    index_type,
                    metric,
                    num_partitions,
                    num_sub_vectors,
                    replace,
                ),
            )?
            .into())
    }

    /// Build a scalar secondary index (BTREE / BITMAP) on a Lance column so
    /// filters skip data instead of scanning.
    #[pyo3(signature = (path, column, index_type="BTREE"))]
    pub fn create_lance_scalar_index(
        &self,
        py: Python<'_>,
        path: &str,
        column: &str,
        index_type: &str,
    ) -> PyResult<Py<PyAny>> {
        let helper = py.import("jude._lance")?;
        Ok(helper
            .call_method1("create_scalar_index", (path, column, index_type))?
            .into())
    }

    /// Approximate nearest-neighbour search over a Lance dataset: the `k` rows
    /// whose `column` vector is closest to `query` (adds a `_distance` column).
    /// `filter` pushes a predicate into the scan for hybrid search. Returns a
    /// jude relation, so ANN results compose with ordinary SQL.
    #[pyo3(signature = (path, column, query, k=10, filter=None, columns=None, nprobes=None, refine_factor=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn lance_vector_search(
        &self,
        py: Python<'_>,
        path: &str,
        column: &str,
        query: &Bound<'_, PyAny>,
        k: i64,
        filter: Option<&str>,
        columns: Option<&Bound<'_, PyAny>>,
        nprobes: Option<i64>,
        refine_factor: Option<i64>,
    ) -> PyResult<Relation> {
        let helper = py.import("jude._lance")?;
        let cols = columns.map(|c| c.clone().unbind());
        let table = helper.call_method1(
            "vector_search",
            (path, column, query, k, filter, cols, nprobes, refine_factor),
        )?;
        let batches = crate::arrow_ffi::py_arrow_to_batches(py, &table)?;
        Relation::new_materialized(py, self, batches)
    }

    /// DuckDB-compatible `from_csv_auto(path_or_file, **options)` — an alias of
    /// `read_csv`/`read_csv_auto` that accepts the same CSV options.
    #[pyo3(signature = (path_or_buffer, **kwargs))]
    pub fn from_csv_auto(
        &self,
        py: Python<'_>,
        path_or_buffer: &Bound<'_, PyAny>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Relation> {
        self.read_csv(py, path_or_buffer, kwargs)
    }

    fn from_parquet(&self, py: Python<'_>, glob: &str) -> PyResult<Relation> {
        self.read_parquet(py, glob)
    }

    /// Register a Python Arrow-like object (Table/RecordBatch/reader) as a view,
    /// ingesting the data zero-copy through the Arrow table function.
    pub fn from_arrow(&self, py: Python<'_>, arrow_obj: &Bound<'_, PyAny>) -> PyResult<Relation> {
        let batches = crate::arrow_ffi::py_arrow_to_batches(py, arrow_obj)?;
        Relation::new_materialized(py, self, batches)
    }

    /// Ingest a Daft DataFrame (or anything with `to_arrow`) as a jude relation,
    /// so Daft's multimodal / embedding ops feed back into jude + SQL + Lance.
    pub fn from_daft(&self, py: Python<'_>, df: &Bound<'_, PyAny>) -> PyResult<Relation> {
        let helper = py.import("jude._daft")?;
        let table = helper.call_method1("from_daft", (df,))?;
        let batches = crate::arrow_ffi::py_arrow_to_batches(py, &table)?;
        Relation::new_materialized(py, self, batches)
    }

    /// Generator / table UDF: call a Python function that produces rows and turn
    /// the result into a relation. `args` are forwarded to `fn_`; the return
    /// value is normalized to Arrow (pyarrow Table/RecordBatch/reader, a frame
    /// with an Arrow C stream, a list of dicts, or rows + a `schema` of names).
    #[pyo3(signature = (fn_, *args, schema=None))]
    fn table_function_udf(
        &self,
        py: Python<'_>,
        fn_: &Bound<'_, PyAny>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        schema: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Relation> {
        let result = fn_.call1(args)?;
        let helper = py.import("jude._table_udf")?;
        let table = helper.call_method1("normalize_to_arrow", (result, schema))?;
        let batches = crate::arrow_ffi::py_arrow_to_batches(py, &table)?;
        Relation::new_materialized(py, self, batches)
    }

    /// DuckDB `from_df(df)`: build a relation from a pandas DataFrame by
    /// converting it to an Arrow table (via pyarrow) and materializing it.
    fn from_df(&self, py: Python<'_>, df: &Bound<'_, PyAny>) -> PyResult<Relation> {
        let pa = py.import("pyarrow")?;
        let table = pa.getattr("Table")?.call_method1("from_pandas", (df,))?;
        self.from_arrow(py, &table)
    }

    /// DuckDB alias for `from_df`.
    fn df_scan(&self, py: Python<'_>, df: &Bound<'_, PyAny>) -> PyResult<Relation> {
        self.from_df(py, df)
    }

    /// Register an Arrow-like object under a persistent view name.
    fn register(
        &self,
        py: Python<'_>,
        view_name: &str,
        python_object: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let batches = crate::arrow_ffi::py_arrow_to_batches(py, python_object)?;
        crate::arrow_ffi::register_batches_as_view(&self.inner, view_name, &batches)?;
        Ok(())
    }

    fn unregister(&self, view_name: &str) -> PyResult<()> {
        // In DuckDB, `DROP VIEW IF EXISTS x` raises if x is a TABLE (and vice
        // versa), so try each drop independently and ignore the type mismatch.
        let ident = quote_ident(view_name);
        let _ = self
            .inner
            .execute_batch(&format!("DROP TABLE IF EXISTS {ident}"));
        let _ = self
            .inner
            .execute_batch(&format!("DROP VIEW IF EXISTS {ident}"));
        Ok(())
    }

    fn cursor(&self) -> PyResult<Connection> {
        // Share the same underlying database (DuckDB connections are multi-cursor safe).
        Ok(Connection::from_arc(self.inner.clone()))
    }

    fn duplicate(&self) -> PyResult<Connection> {
        self.cursor()
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __exit__(
        &self,
        _exc_type: &Bound<'_, PyAny>,
        _exc_val: &Bound<'_, PyAny>,
        _tb: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        Ok(false)
    }

    /// DuckDB-style scalar UDF registration:
    /// `conn.create_function(name, fn, parameters=[...], return_type="...",
    /// type="native"|"arrow")`. `type="arrow"` (or `vectorized=True`) registers a
    /// vectorized UDF that receives whole pyarrow columns (one GIL acquisition
    /// per chunk, full type coverage) instead of the row-by-row native adapter.
    #[pyo3(signature = (name, func, parameters=None, return_type=None, r#type=None, vectorized=false, exception_handling=None, null_handling=None, side_effects=false, **_kwargs))]
    fn create_function(
        &self,
        name: &str,
        func: &Bound<'_, PyAny>,
        parameters: Option<Vec<String>>,
        return_type: Option<String>,
        r#type: Option<String>,
        vectorized: bool,
        exception_handling: Option<String>,
        null_handling: Option<String>,
        side_effects: bool,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<()> {
        let is_arrow = vectorized
            || r#type
                .as_deref()
                .map(|t| t.eq_ignore_ascii_case("arrow"))
                .unwrap_or(false);
        crate::expression_udf::registration::attach_function(
            func,
            Some(name),
            Some(self),
            false,
            parameters,
            return_type,
            is_arrow,
            exception_handling,
            null_handling,
            side_effects,
        )
    }

    fn remove_function(&self, name: &str) -> PyResult<()> {
        crate::expression_udf::registration::detach_function(name, Some(self))
    }

    // ---- Type system helpers (string-based, DuckDB SQL type names) ----

    fn sqltype(&self, type_str: &str) -> PyResult<String> {
        Ok(type_str.to_string())
    }
    fn dtype(&self, type_str: &str) -> PyResult<String> {
        self.sqltype(type_str)
    }
    fn list_type(&self, child_type: &str) -> PyResult<String> {
        Ok(format!("{child_type}[]"))
    }
    /// `struct_type(fields)` — accepts a dict `{name: type}` or a list of
    /// `(name, type)` pairs, producing `STRUCT(name type, ...)`.
    fn struct_type(&self, fields: &Bound<'_, PyAny>) -> PyResult<String> {
        let mut parts: Vec<String> = Vec::new();
        if let Ok(d) = fields.cast::<pyo3::types::PyDict>() {
            for (k, v) in d.iter() {
                parts.push(format!(
                    "{} {}",
                    k.extract::<String>()?,
                    v.extract::<String>()?
                ));
            }
        } else if let Ok(pairs) = fields.extract::<Vec<(String, String)>>() {
            for (n, t) in pairs {
                parts.push(format!("{n} {t}"));
            }
        } else {
            return Err(invalid_input(
                fields.py(),
                "struct_type expects a dict {name: type} or list of (name, type)",
            ));
        }
        Ok(format!("STRUCT({})", parts.join(", ")))
    }
    fn array_type(&self, child_type: &str, size: usize) -> PyResult<String> {
        Ok(format!("{child_type}[{size}]"))
    }
    fn map_type(&self, key_type: &str, value_type: &str) -> PyResult<String> {
        Ok(format!("MAP({key_type}, {value_type})"))
    }

    // ---- Profiling ----

    fn enable_profiling(&self) -> PyResult<()> {
        self.inner
            .execute_batch("PRAGMA enable_profiling")
            .map_err(|e| Error::DuckDb(e).into())
    }
    fn disable_profiling(&self) -> PyResult<()> {
        self.inner
            .execute_batch("PRAGMA disable_profiling")
            .map_err(|e| Error::DuckDb(e).into())
    }

    // ---- Extensions ----

    fn install_extension(&self, name: &str) -> PyResult<()> {
        self.inner
            .execute_batch(&format!("INSTALL {name}"))
            .map_err(|e| Error::DuckDb(e).into())
    }
    fn load_extension(&self, name: &str) -> PyResult<()> {
        self.inner
            .execute_batch(&format!("LOAD {name}"))
            .map_err(|e| Error::DuckDb(e).into())
    }

    // ---- Table/View factory ----

    fn table(&self, py: Python<'_>, name: &str) -> PyResult<Relation> {
        // Eagerly validate the table exists (DuckDB's TableRelation binds on
        // construction), then build a Table-leaf plan so `insert`/`update` can
        // recover the base-table name. The name doubles as the relation alias.
        // Register any in-scope df/arrow variable of this name first, so the
        // replacement scan (`conn.table("some_df")`) still resolves.
        let scan_sql = format!("SELECT * FROM {}", quote_qualified_name(name));
        self.register_scan_candidates(py, &scan_sql);
        self.inner
            .prepare(&scan_sql)
            .map_err(|e| PyErr::from(Error::DuckDb(e)))?;
        let rel = Relation::from_plan(
            py,
            self,
            crate::plan::LogicalPlan::Table {
                name: name.to_string(),
            },
        )?;
        rel.set_alias(py, name)
    }
    fn view(&self, py: Python<'_>, name: &str) -> PyResult<Relation> {
        let rel = self.sql_lazy(py, &format!("SELECT * FROM {}", quote_ident(name)))?;
        rel.set_alias(py, name)
    }

    /// DuckDB `conn.table_function(name, parameters=[...])`: build a relation
    /// from a table-function call, e.g. `table_function("test_all_types")` →
    /// `SELECT * FROM test_all_types()`. `parameters` are rendered as SQL
    /// literals in call order.
    #[pyo3(signature = (name, parameters=None))]
    fn table_function(
        &self,
        py: Python<'_>,
        name: &str,
        parameters: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Relation> {
        let args = match parameters {
            None => String::new(),
            Some(p) => {
                let mut rendered: Vec<String> = Vec::new();
                // Accept a list/tuple of scalar params.
                if let Ok(seq) = p.try_iter() {
                    for item in seq {
                        rendered.push(py_literal_to_sql(&item?)?);
                    }
                }
                rendered.join(", ")
            }
        };
        self.sql_lazy(
            py,
            &format!("SELECT * FROM {}({})", quote_ident(name), args),
        )
    }
    /// DuckDB `values(...)`: build a relation from literal rows. Variadic like
    /// DuckDB: `values(1, 2, 3)` / `values(Expr, Expr, ...)` is a single row;
    /// `values([1, 2, 3])` is a single row from a list; `values([[..],[..]])` is
    /// multiple rows; a lone SQL string is used verbatim. No arguments raises
    /// InvalidInputException, and mixing Expressions with plain values raises
    /// "Please provide arguments of type Expression!".
    #[pyo3(signature = (*args))]
    fn values(&self, py: Python<'_>, args: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Relation> {
        if args.is_empty() {
            return Err(invalid_input(
                py,
                "Could not create a ValueRelation without any inputs",
            ));
        }
        let items: Vec<Bound<'_, PyAny>> = args.iter().collect();

        // Multiple args. If the first is a tuple, they are ROWS (DuckDB's
        // multi-tuple form): every arg must be a tuple, all the same length.
        // Otherwise they are the cells of a single row (bare Expressions/scalars).
        if items.len() > 1 {
            let first_is_tuple = items[0].is_instance_of::<pyo3::types::PyTuple>();
            if first_is_tuple {
                let expected = items[0].len()?;
                let mut rows = Vec::new();
                for a in &items {
                    if !a.is_instance_of::<pyo3::types::PyTuple>() {
                        return Err(invalid_input(py, "Expected objects of type tuple"));
                    }
                    let cells: Vec<Bound<'_, PyAny>> = a.try_iter()?.collect::<PyResult<_>>()?;
                    if cells.len() != expected {
                        return Err(invalid_input(
                            py,
                            &format!(
                            "Mismatch between length of tuples in input, expected {} but found {}",
                            expected, cells.len()
                        ),
                        ));
                    }
                    let rendered: Vec<String> = cells
                        .iter()
                        .map(values_cell_to_sql)
                        .collect::<PyResult<_>>()?;
                    rows.push(format!("({})", rendered.join(", ")));
                }
                return self.sql_lazy(py, &format!("SELECT * FROM (VALUES {})", rows.join(", ")));
            }
            // A single row whose cells are the bare args (Expressions or
            // autocastable scalars); any other type is an error.
            let mut cells = Vec::new();
            for a in &items {
                if let Ok(expr) = a.extract::<crate::expressions::Expression>() {
                    cells.push(expr.render_sql());
                } else if is_autocastable_scalar(a) {
                    cells.push(py_literal_to_sql(a)?);
                } else {
                    return Err(invalid_input(
                        py,
                        "Please provide arguments of type Expression!",
                    ));
                }
            }
            return self.sql_lazy(
                py,
                &format!("SELECT * FROM (VALUES ({}))", cells.join(", ")),
            );
        }

        // Single argument.
        let values = &items[0];
        // An empty tuple/list is rejected with DuckDB's message.
        if (values.is_instance_of::<pyo3::types::PyTuple>()
            || values.is_instance_of::<pyo3::types::PyList>())
            && values.len().unwrap_or(0) == 0
        {
            return Err(invalid_input(py, "Please provide a non-empty tuple"));
        }
        if let Ok(sql) = values.extract::<String>() {
            return self.sql_lazy(py, &format!("SELECT * FROM (VALUES {sql})"));
        }
        if let Ok(expr) = values.extract::<crate::expressions::Expression>() {
            return self.sql_lazy(
                py,
                &format!("SELECT * FROM (VALUES ({}))", expr.render_sql()),
            );
        }
        // A list/tuple: decide row-of-scalars vs list-of-rows by the first item.
        let rows_sql = if values.is_instance_of::<pyo3::types::PyList>()
            || values.is_instance_of::<pyo3::types::PyTuple>()
        {
            let inner: Vec<Bound<'_, PyAny>> = values.try_iter()?.collect::<PyResult<_>>()?;
            let first_is_row = inner.first().map(|i| {
                i.is_instance_of::<pyo3::types::PyList>()
                    || i.is_instance_of::<pyo3::types::PyTuple>()
            });
            match first_is_row {
                Some(true) => {
                    // list of rows
                    let mut rows = Vec::new();
                    for row in &inner {
                        let mut cells = Vec::new();
                        for cell in row.try_iter()? {
                            cells.push(values_cell_to_sql(&cell?)?);
                        }
                        rows.push(format!("({})", cells.join(", ")));
                    }
                    rows.join(", ")
                }
                _ => {
                    // single row of scalars (or Expressions)
                    let mut cells = Vec::new();
                    for cell in &inner {
                        cells.push(values_cell_to_sql(cell)?);
                    }
                    format!("({})", cells.join(", "))
                }
            }
        } else {
            format!("({})", py_literal_to_sql(values)?)
        };
        self.sql_lazy(py, &format!("SELECT * FROM (VALUES {rows_sql})"))
    }

    // ---- Introspection ----

    #[getter]
    fn rowcount(&self) -> i64 {
        -1
    }

    fn __repr__(&self) -> String {
        "Connection".to_string()
    }
}

/// Quote an identifier for use in SQL if it isn't a bare word.
pub fn quote_ident(name: &str) -> String {
    // Quote if not a simple identifier, if it starts with a digit, or if it is a
    // reserved SQL keyword (e.g. a table literally named "table"/"select").
    let simple = !name.is_empty()
        && name.chars().all(|c| c.is_alphanumeric() || c == '_')
        && !name.chars().next().is_some_and(|c| c.is_ascii_digit());
    if simple && !is_reserved_keyword(name) {
        name.to_string()
    } else {
        format!("\"{}\"", name.replace('"', "\"\""))
    }
}

/// Quote a possibly schema/catalog-qualified name (catalog.schema.table) by
/// quoting each dot-separated part independently, so "not_main.tbl" becomes
/// "not_main"."tbl" rather than one quoted identifier. If the name already
/// contains a quote, trust the caller's quoting.
pub fn quote_qualified_name(name: &str) -> String {
    if name.contains('"') {
        return name.to_string();
    }
    name.split('.')
        .map(quote_ident)
        .collect::<Vec<_>>()
        .join(".")
}

/// A conservative set of SQL reserved keywords that must be quoted when used as
/// an identifier. Not exhaustive, but covers the common collisions.
fn is_reserved_keyword(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "select"
            | "from"
            | "where"
            | "table"
            | "update"
            | "insert"
            | "delete"
            | "group"
            | "order"
            | "by"
            | "join"
            | "on"
            | "as"
            | "and"
            | "or"
            | "not"
            | "in"
            | "is"
            | "null"
            | "case"
            | "when"
            | "then"
            | "else"
            | "end"
            | "create"
            | "drop"
            | "into"
            | "values"
            | "distinct"
            | "union"
            | "all"
            | "having"
            | "limit"
            | "offset"
            | "using"
            | "default"
            | "check"
            | "primary"
            | "foreign"
            | "references"
            | "constraint"
            | "unique"
            | "index"
            | "view"
            | "column"
            | "add"
            | "alter"
            | "with"
            | "over"
            | "partition"
            | "window"
            | "asc"
            | "desc"
            | "between"
            | "like"
            | "exists"
            | "cross"
            | "inner"
            | "outer"
            | "left"
            | "right"
            | "full"
            | "natural"
            | "collate"
            | "cast"
    )
}

/// Escape a string literal for single-quoted SQL context.
pub fn escape_sql_string(s: &str) -> String {
    s.replace('\'', "''")
}

/// A scalar that can autocast to a constant Expression (int/float/str/bool/None).
fn is_autocastable_scalar(obj: &Bound<'_, PyAny>) -> bool {
    obj.is_none()
        || obj.is_instance_of::<pyo3::types::PyBool>()
        || obj.is_instance_of::<pyo3::types::PyInt>()
        || obj.is_instance_of::<pyo3::types::PyFloat>()
        || obj.is_instance_of::<pyo3::types::PyString>()
}

/// Build a Python TypeError PyErr.
fn py_type_error(msg: &str) -> PyErr {
    pyo3::exceptions::PyTypeError::new_err(msg.to_string())
}

/// True if `query`'s leading keyword is a side-effecting statement that returns
/// no result set, so `sql()` should execute it eagerly (like DuckDB) rather than
/// build a lazy relation. Result-producing statements (SELECT/WITH/VALUES/TABLE/
/// FROM/DESCRIBE/SUMMARIZE/PRAGMA/CALL/EXPLAIN/SHOW) stay lazy. INSERT is treated
/// as eager unless it has a RETURNING clause.
fn is_eager_statement(query: &str) -> bool {
    let s = query.trim_start();
    // Skip a leading line comment if present.
    let s = s.trim_start();
    let kw: String = s
        .chars()
        .take_while(|c| c.is_ascii_alphabetic())
        .collect::<String>()
        .to_ascii_lowercase();
    match kw.as_str() {
        "create" | "drop" | "alter" | "update" | "delete" | "attach" | "detach" | "begin"
        | "commit" | "rollback" | "truncate" | "checkpoint" | "use" | "install" | "load"
        | "set" | "reset" | "vacuum" | "analyze" | "comment" => true,
        "insert" => !s.to_ascii_lowercase().contains("returning"),
        _ => false,
    }
}

/// Build a jude.exceptions.InvalidInputException PyErr.
fn invalid_input(py: Python<'_>, msg: &str) -> PyErr {
    match py
        .import("jude.exceptions")
        .and_then(|m| m.getattr("InvalidInputException"))
    {
        Ok(exc) => match exc.call1((msg.to_string(),)) {
            Ok(inst) => PyErr::from_value(inst),
            Err(e) => e,
        },
        Err(_) => pyo3::exceptions::PyValueError::new_err(msg.to_string()),
    }
}

/// Render a value-list cell: a jude Expression renders via its SQL, anything
/// else is a scalar literal. So `values(ConstantExpression(5))` yields `5`, not
/// `'5'` (the Expression carries its own type).
fn values_cell_to_sql(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(expr) = obj.extract::<crate::expressions::Expression>() {
        return Ok(expr.render_sql());
    }
    py_literal_to_sql(obj)
}

/// Render a Python scalar as a SQL literal (for table-function arguments).
fn py_literal_to_sql(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    if obj.is_none() {
        return Ok("NULL".to_string());
    }
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(if b {
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
        return Ok(format!("'{}'", escape_sql_string(&s)));
    }
    // Fallback: stringify.
    Ok(format!(
        "'{}'",
        escape_sql_string(&obj.str()?.extract::<String>()?)
    ))
}

/// Render the CSV source argument of `read_csv(...)`: a path string,
/// `pathlib.Path`/os.PathLike, or a list of such. Returns the SQL fragment that
/// goes immediately after `read_csv(` (before any options).
fn csv_source_sql(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    // A list/tuple of paths → ['a.csv', 'b.csv'].
    if obj.is_instance_of::<pyo3::types::PyList>() || obj.is_instance_of::<pyo3::types::PyTuple>() {
        let mut parts = Vec::new();
        for item in obj.try_iter()? {
            let p = os_fspath(&item?)?;
            parts.push(format!("'{}'", escape_sql_string(&p)));
        }
        return Ok(format!("[{}]", parts.join(", ")));
    }
    let path = os_fspath(obj)?;
    Ok(format!("'{}'", escape_sql_string(&path)))
}

/// Coerce a str / pathlib.Path / os.PathLike into a filesystem path string.
fn os_fspath(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(s) = obj.extract::<String>() {
        return Ok(s);
    }
    let py = obj.py();
    let os = py.import("os")?;
    let p = os.call_method1("fspath", (obj,))?;
    p.extract::<String>()
}

/// CSV option keywords whose Python (pandas-style) name differs from DuckDB's
/// `read_csv` option name.
fn csv_alias(key: &str) -> &str {
    match key {
        "skiprows" => "skip",
        "quotechar" => "quote",
        "escapechar" => "escape",
        "lineterminator" => "new_line",
        "delimiter" => "sep",
        "na_values" => "nullstr",
        "dtype" => "dtypes",
        "decimal" => "decimal_separator",
        "date_format" => "dateformat",
        "timestamp_format" => "timestampformat",
        other => other,
    }
}

/// Translate DuckDB `read_csv` keyword options into the SQL option list
/// (`key=value, key=value`). Values are rendered as SQL literals matching their
/// Python type (bool → TRUE/FALSE, list → [...], dict → {...}, enum → its
/// `.value`).
fn csv_kwargs_to_sql(kwargs: Option<&Bound<'_, pyo3::types::PyDict>>) -> PyResult<String> {
    let Some(kwargs) = kwargs else {
        return Ok(String::new());
    };
    let mut parts: Vec<String> = Vec::new();
    for (k, v) in kwargs.iter() {
        let key = k.extract::<String>()?;
        if v.is_none() {
            continue;
        }
        let opt = csv_alias(&key);
        let val = if opt == "new_line" {
            csv_newline_literal(&v)?
        } else if opt == "dtypes" || opt == "types" || opt == "column_types" {
            csv_dtypes_to_sql(&v)?
        } else {
            csv_value_to_sql(&v)?
        };
        parts.push(format!("{opt}={val}"));
    }
    Ok(parts.join(", "))
}

/// Render the `new_line` CSV option: DuckDB accepts only the escape-sequence
/// forms `'\r'`, `'\n'`, `'\r\n'`. Normalize literal newline characters and the
/// `CSVLineTerminator` enum names/values to those forms.
fn csv_newline_literal(v: &Bound<'_, PyAny>) -> PyResult<String> {
    // Unwrap enum → its `.value` (our CSVLineTerminator uses "\n"/"\r\n").
    let raw = if v.getattr("name").is_ok() {
        match v.getattr("value") {
            Ok(inner) => inner
                .extract::<String>()
                .or_else(|_| v.str().and_then(|s| s.extract::<String>()))
                .unwrap_or_default(),
            Err(_) => v.extract::<String>().unwrap_or_default(),
        }
    } else {
        v.extract::<String>()?
    };
    let mapped = match raw.as_str() {
        "\n" | "LINE_FEED" => "\\n",
        "\r" | "CARRIAGE_RETURN" => "\\r",
        "\r\n" | "CARRIAGE_RETURN_LINE_FEED" => "\\r\\n",
        // Already an escape-sequence form ("\\n", "\\r", "\\r\\n") or something
        // else — pass through unchanged.
        other => other,
    };
    Ok(format!("'{}'", escape_sql_string(mapped)))
}

/// Render a Python CSV-option value as a DuckDB SQL literal.
fn csv_value_to_sql(v: &Bound<'_, PyAny>) -> PyResult<String> {
    // Enum (e.g. CSVLineTerminator) → its `.value`.
    if let Ok(inner) = v.getattr("value") {
        // Only unwrap when it's a genuine enum member (has `name` too), to avoid
        // grabbing unrelated `.value` attributes.
        if v.getattr("name").is_ok() && !inner.is_none() {
            return csv_value_to_sql(&inner);
        }
    }
    if v.is_instance_of::<pyo3::types::PyBool>() {
        return Ok(if v.extract::<bool>()? {
            "TRUE".to_string()
        } else {
            "FALSE".to_string()
        });
    }
    if v.is_instance_of::<pyo3::types::PyList>() || v.is_instance_of::<pyo3::types::PyTuple>() {
        let mut items = Vec::new();
        for item in v.try_iter()? {
            items.push(csv_value_to_sql(&item?)?);
        }
        return Ok(format!("[{}]", items.join(", ")));
    }
    if let Ok(dict) = v.cast::<pyo3::types::PyDict>() {
        let mut items = Vec::new();
        for (dk, dv) in dict.iter() {
            let key = dk.extract::<String>()?;
            items.push(format!(
                "'{}': {}",
                escape_sql_string(&key),
                csv_value_to_sql(&dv)?
            ));
        }
        return Ok(format!("{{{}}}", items.join(", ")));
    }
    py_literal_to_sql(v)
}

/// Render a CSV `dtype`/`types` option. Unlike other options, its values are
/// DuckDB *type names*: a `{col: type}` dict, a `[type, …]` list, or a single
/// type. Each type may be a Python builtin (int/float/str/bool), an existing
/// type-name string, or anything else (rendered via `str()` best-effort).
fn csv_dtypes_to_sql(v: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(dict) = v.cast::<pyo3::types::PyDict>() {
        let mut items = Vec::new();
        for (dk, dv) in dict.iter() {
            let key = dk.extract::<String>()?;
            items.push(format!(
                "'{}': {}",
                escape_sql_string(&key),
                csv_type_literal(&dv)?
            ));
        }
        return Ok(format!("{{{}}}", items.join(", ")));
    }
    if v.is_instance_of::<pyo3::types::PyList>() || v.is_instance_of::<pyo3::types::PyTuple>() {
        let mut items = Vec::new();
        for item in v.try_iter()? {
            items.push(csv_type_literal(&item?)?);
        }
        return Ok(format!("[{}]", items.join(", ")));
    }
    csv_type_literal(v)
}

/// Render one dtype value as a single-quoted DuckDB type name.
fn csv_type_literal(v: &Bound<'_, PyAny>) -> PyResult<String> {
    // A string is already a type name.
    if let Ok(s) = v.extract::<String>() {
        return Ok(format!("'{}'", escape_sql_string(&s)));
    }
    // Python builtin type objects map to DuckDB base types (DuckDB-Python parity).
    if let Some(name) = py_type_to_duckdb_name(v) {
        return Ok(format!("'{name}'"));
    }
    // Fallback: str(v) (e.g. a numpy dtype) — best effort.
    let s = v.str()?.extract::<String>()?;
    Ok(format!("'{}'", escape_sql_string(&s)))
}

/// Map a Python builtin type object to a DuckDB base type name, matching
/// DuckDB-Python's read_csv dtype coercion. Returns None for anything else.
fn py_type_to_duckdb_name(v: &Bound<'_, PyAny>) -> Option<String> {
    let py = v.py();
    let builtins = py.import("builtins").ok()?;
    for (name, duck) in [
        ("bool", "BOOLEAN"),
        ("int", "BIGINT"),
        ("float", "DOUBLE"),
        ("str", "VARCHAR"),
    ] {
        if let Ok(t) = builtins.getattr(name) {
            if v.is(&t) {
                return Some(duck.to_string());
            }
        }
    }
    None
}

enum PyParam {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
}

impl duckdb::ToSql for PyParam {
    fn to_sql(&self) -> duckdb::Result<duckdb::types::ToSqlOutput<'_>> {
        use duckdb::types::{ToSqlOutput, Value};
        Ok(match self {
            PyParam::Null => ToSqlOutput::Owned(Value::Null),
            PyParam::Bool(b) => ToSqlOutput::Owned(Value::Boolean(*b)),
            PyParam::Int(i) => ToSqlOutput::Owned(Value::BigInt(*i)),
            PyParam::Float(f) => ToSqlOutput::Owned(Value::Double(*f)),
            PyParam::Str(s) => ToSqlOutput::Owned(Value::Text(s.clone())),
            PyParam::Bytes(b) => ToSqlOutput::Owned(Value::Blob(b.clone())),
        })
    }
}

fn py_params_to_vec(py: &Bound<'_, PyAny>) -> PyResult<Vec<PyParam>> {
    if let Ok(list) = py.extract::<Vec<Py<PyAny>>>() {
        let py_obj = py.py();
        let mut params = Vec::new();
        for item in &list {
            params.push(py_to_param(&item.bind(py_obj))?);
        }
        return Ok(params);
    }
    Ok(vec![py_to_param(py)?])
}

fn py_to_param(v: &Bound<'_, PyAny>) -> PyResult<PyParam> {
    if v.is_none() {
        return Ok(PyParam::Null);
    }
    if let Ok(b) = v.extract::<bool>() {
        return Ok(PyParam::Bool(b));
    }
    if let Ok(i) = v.extract::<i64>() {
        return Ok(PyParam::Int(i));
    }
    if let Ok(f) = v.extract::<f64>() {
        return Ok(PyParam::Float(f));
    }
    if let Ok(s) = v.extract::<String>() {
        return Ok(PyParam::Str(s));
    }
    if let Ok(b) = v.extract::<Vec<u8>>() {
        return Ok(PyParam::Bytes(b));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "Unsupported parameter type: {}",
        v.get_type().name()?
    )))
}
