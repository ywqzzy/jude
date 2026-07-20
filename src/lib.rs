use pyo3::prelude::*;

pub mod ai;
pub mod arrow_ffi;
pub mod config;
pub mod connection;
pub mod curate;
pub mod curate_mm;
pub mod curate_py;
pub mod dist;
pub mod env;
pub mod error;
pub mod expression_udf;
pub mod expressions;
pub mod kmeans;
pub mod mat_scan;
pub mod multimodal;
pub mod observe;
pub mod observe_audit;
pub mod plan;
pub mod relation;
pub mod runners;
pub mod stream;
pub mod typing;
pub mod udf;

pub use config::{configure, current_config, Config};
pub use connection::Connection;
pub use env::EnvRegistry;
pub use expressions::{col, lit, sql_expr, Expression};
pub use relation::Relation;

static VERSION: &str = "0.1.0";

#[pymodule]
fn jude(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();

    m.add("__version__", VERSION)?;
    m.add("apilevel", "2.0")?;
    m.add("paramstyle", "qmark")?;
    m.add("threadsafety", 1)?;

    m.add_class::<Connection>()?;
    m.add_class::<Relation>()?;
    m.add_class::<crate::stream::RecordBatchStream>()?;
    m.add_class::<Expression>()?;
    m.add_class::<Config>()?;
    m.add_class::<EnvRegistry>()?;

    m.add_function(wrap_pyfunction!(connect, m)?)?;
    m.add_function(wrap_pyfunction!(sql, m)?)?;
    m.add_function(wrap_pyfunction!(read_csv, m)?)?;
    m.add_function(wrap_pyfunction!(read_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(read_iceberg, m)?)?;
    m.add_function(wrap_pyfunction!(read_lance, m)?)?;
    m.add_function(wrap_pyfunction!(read_json, m)?)?;
    m.add_function(wrap_pyfunction!(from_csv_auto, m)?)?;
    m.add_function(wrap_pyfunction!(write_csv, m)?)?;
    m.add_function(wrap_pyfunction!(project_df, m)?)?;
    m.add_function(wrap_pyfunction!(filter_df, m)?)?;
    m.add_function(wrap_pyfunction!(order_df, m)?)?;
    m.add_function(wrap_pyfunction!(distinct_df, m)?)?;
    m.add_function(wrap_pyfunction!(alias_df, m)?)?;
    m.add_function(wrap_pyfunction!(col_py, m)?)?;
    m.add_function(wrap_pyfunction!(lit_py, m)?)?;
    m.add_function(wrap_pyfunction!(sql_expr_py, m)?)?;
    m.add_function(wrap_pyfunction!(configure, m)?)?;
    m.add_function(wrap_pyfunction!(current_config, m)?)?;
    m.add_function(wrap_pyfunction!(make_env, m)?)?;

    // UDF registration
    m.add_function(wrap_pyfunction!(
        expression_udf::registration::attach_function,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        expression_udf::registration::detach_function,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(shutdown_udf_pools, m)?)?;

    // AI submodule
    let ai = PyModule::new(py, "ai")?;
    ai::register_bound(&ai)?;
    m.add_submodule(&ai)?;

    // Distributed-orchestration submodule (Rust scheduling brain).
    let dist_mod = PyModule::new(py, "dist")?;
    dist_mod.add_class::<dist::WorkerManager>()?;
    dist_mod.add_class::<dist::resource::ResourceManager>()?;
    dist_mod.add_class::<dist::cluster::ClusterScheduler>()?;
    m.add_submodule(&dist_mod)?;

    // Observability submodule (Rust metrics/progress registry). Named `_observe`
    // so the Python `jude.observe` facade module owns the public name.
    let observe_mod = PyModule::new(py, "_observe")?;
    observe::register_bound(&observe_mod)?;
    observe_audit::register(&observe_mod)?;
    m.add_submodule(&observe_mod)?;

    // Data-curation kernels (chunking / dedup hashing / quality heuristics).
    let curate_mod = PyModule::new(py, "_curate")?;
    curate_py::register(&curate_mod)?;
    m.add_submodule(&curate_mod)?;

    Ok(())
}

#[pyfunction]
fn shutdown_udf_pools() {
    udf::shutdown_all_pools();
}

#[pyfunction]
#[pyo3(signature = (database=None, read_only=false, config=None, **_kwargs))]
fn connect(
    database: Option<&str>,
    read_only: bool,
    config: Option<&Bound<'_, PyAny>>,
    _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<Connection> {
    // DuckDB's entry point is `connect(database=":memory:", read_only=False,
    // config={})`; accept the same keyword surface. `read_only`/`config` are
    // accepted for API compatibility (in-memory DBs ignore them).
    let _ = (read_only, config);
    Connection::connect(database)
}

#[pyfunction]
#[pyo3(signature = (query, params=None))]
fn sql(py: Python<'_>, query: &str, params: Option<&Bound<'_, PyAny>>) -> PyResult<Relation> {
    let conn = Connection::connect(Some(":memory:"))?;
    conn.sql(py, query, params)
}

/// Module-level `duckdb.read_csv(...)`: read against a fresh in-memory
/// connection (DuckDB's connection-wrapper convenience entry point).
#[pyfunction]
#[pyo3(signature = (path_or_buffer, **kwargs))]
fn read_csv(
    py: Python<'_>,
    path_or_buffer: &Bound<'_, PyAny>,
    kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<Relation> {
    let conn = Connection::connect(Some(":memory:"))?;
    conn.read_csv(py, path_or_buffer, kwargs)
}

#[pyfunction]
fn read_parquet(py: Python<'_>, glob: &str) -> PyResult<Relation> {
    let conn = Connection::connect(Some(":memory:"))?;
    conn.read_parquet(py, glob)
}

#[pyfunction]
#[pyo3(signature = (path, snapshot_id=None, version=None))]
fn read_iceberg(
    py: Python<'_>,
    path: &str,
    snapshot_id: Option<i64>,
    version: Option<String>,
) -> PyResult<Relation> {
    let conn = Connection::connect(Some(":memory:"))?;
    conn.read_iceberg(py, path, snapshot_id, version)
}

#[pyfunction]
#[pyo3(signature = (path, columns=None, filter=None, version=None))]
fn read_lance(
    py: Python<'_>,
    path: &str,
    columns: Option<&Bound<'_, PyAny>>,
    filter: Option<&str>,
    version: Option<&Bound<'_, PyAny>>,
) -> PyResult<Relation> {
    let conn = Connection::connect(Some(":memory:"))?;
    conn.read_lance(py, path, columns, filter, version)
}

#[pyfunction]
fn read_json(py: Python<'_>, path: &str) -> PyResult<Relation> {
    let conn = Connection::connect(Some(":memory:"))?;
    conn.read_json(py, path)
}

/// `duckdb.from_csv_auto(path, **options)` — read a CSV via a fresh in-memory connection.
#[pyfunction]
#[pyo3(signature = (path_or_buffer, **kwargs))]
fn from_csv_auto(
    py: Python<'_>,
    path_or_buffer: &Bound<'_, PyAny>,
    kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<Relation> {
    let conn = Connection::connect(Some(":memory:"))?;
    conn.from_csv_auto(py, path_or_buffer, kwargs)
}

/// `duckdb.write_csv(df, path)` — write a DataFrame/relation to CSV.
#[pyfunction]
fn write_csv(py: Python<'_>, df: &Bound<'_, PyAny>, path: &str) -> PyResult<()> {
    df_to_relation(py, df)?.write_csv(py, path)
}

// ---- Module-level DataFrame relational helpers (DuckDB convenience API) ----
// `duckdb.project(df, "i")`, `duckdb.filter(df, "i>1")`, etc. Each builds a
// relation over the DataFrame (via a fresh in-memory connection) and applies
// the operator, matching DuckDB's connection-wrapper functions.

fn df_to_relation(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Relation> {
    // Already a jude relation? use it directly.
    if let Ok(rel) = obj.cast::<Relation>() {
        return Ok(rel.borrow().clone_ref_relation(py));
    }
    let conn = Connection::connect(Some(":memory:"))?;
    conn.from_arrow(py, obj)
}

#[pyfunction]
#[pyo3(name = "project", signature = (df, project_expr, groups=""))]
fn project_df(
    py: Python<'_>,
    df: &Bound<'_, PyAny>,
    project_expr: &Bound<'_, PyAny>,
    groups: &str,
) -> PyResult<Relation> {
    let _ = groups;
    df_to_relation(py, df)?.project(py, project_expr)
}

#[pyfunction]
#[pyo3(name = "filter")]
fn filter_df(
    py: Python<'_>,
    df: &Bound<'_, PyAny>,
    filter_expr: &Bound<'_, PyAny>,
) -> PyResult<Relation> {
    df_to_relation(py, df)?.filter(py, filter_expr)
}

#[pyfunction]
#[pyo3(name = "order")]
fn order_df(
    py: Python<'_>,
    df: &Bound<'_, PyAny>,
    order_expr: &Bound<'_, PyAny>,
) -> PyResult<Relation> {
    df_to_relation(py, df)?.order(py, order_expr)
}

#[pyfunction]
#[pyo3(name = "distinct")]
fn distinct_df(py: Python<'_>, df: &Bound<'_, PyAny>) -> PyResult<Relation> {
    df_to_relation(py, df)?.distinct(py)
}

#[pyfunction]
#[pyo3(name = "alias")]
fn alias_df(py: Python<'_>, df: &Bound<'_, PyAny>, alias: &str) -> PyResult<Relation> {
    df_to_relation(py, df)?.set_alias(py, alias)
}

#[pyfunction(name = "col")]
fn col_py(name: &str) -> Expression {
    col(name)
}

#[pyfunction(name = "lit")]
fn lit_py(value: &Bound<'_, PyAny>) -> PyResult<Expression> {
    expressions::lit_from_py(value)
}

#[pyfunction(name = "sql_expr")]
fn sql_expr_py(sql: &str) -> Expression {
    sql_expr(sql)
}

#[pyfunction]
fn make_env() -> PyResult<EnvRegistry> {
    Ok(EnvRegistry::new())
}
