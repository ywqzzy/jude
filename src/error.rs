use pyo3::prelude::*;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("DuckDB error: {0}")]
    DuckDb(#[from] duckdb::Error),
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("HTTP error: {0}")]
    Http(String),
    #[error("Unsupported provider: {0}")]
    UnsupportedProvider(String),
    #[error("Config error: {0}")]
    Config(String),
    #[error("UDF error: {0}")]
    Udf(String),
    #[error("{0}")]
    Other(String),
}

impl From<Error> for PyErr {
    fn from(e: Error) -> PyErr {
        match &e {
            // Map DuckDB errors to jude's DuckDB-compatible exception classes by
            // parsing the error-kind prefix DuckDB embeds in the message
            // ("Binder Error:", "Parser Error:", "Catalog Error:", ...). This is
            // what lets `jude.BinderException` / `jude.ParserException` etc. be
            // raised the way DuckDB (and Vane's tests) expect.
            Error::DuckDb(_) => duckdb_error_to_pyerr(&e.to_string()),
            _ => pyo3::exceptions::PyRuntimeError::new_err(e.to_string()),
        }
    }
}

/// Map a DuckDB error message to the matching jude.exceptions class.
fn duckdb_error_to_pyerr(msg: &str) -> PyErr {
    // Message looks like: "DuckDB error: <Kind> Error: <detail>" or
    // "DuckDB error: <detail>". Find the kind token.
    let lower = msg.to_ascii_lowercase();
    let exc_name = if lower.contains("parser error") || lower.contains("parse error") {
        "ParserException"
    } else if lower.contains("binder error") {
        "BinderException"
    } else if lower.contains("catalog error") {
        "CatalogException"
    } else if lower.contains("conversion error") {
        "ConversionException"
    } else if lower.contains("constraint error") {
        "ConstraintException"
    } else if lower.contains("out of range") {
        "OutOfRangeException"
    } else if lower.contains("invalid input") || lower.contains("invalid type") {
        "InvalidInputException"
    } else if lower.contains("not implemented") {
        "NotImplementedException"
    } else if lower.contains("io error") || lower.contains("i/o error") {
        "IOException"
    } else if lower.contains("permission") {
        "PermissionException"
    } else if lower.contains("out of memory") {
        "OutOfMemoryException"
    } else if lower.contains("syntax error") {
        // DuckDB reports raw parser syntax errors without the "Parser Error"
        // prefix in some paths.
        "ParserException"
    } else {
        // Generic DuckDB failure.
        "Error"
    };

    Python::attach(|py| match import_jude_exception(py, exc_name) {
        Ok(exc_type) => {
            PyErr::from_value(exc_type.call1((msg.to_string(),)).unwrap_or_else(|_| {
                pyo3::exceptions::PyRuntimeError::new_err(msg.to_string())
                    .value(py)
                    .clone()
                    .into_any()
            }))
        }
        Err(_) => pyo3::exceptions::PyRuntimeError::new_err(msg.to_string()),
    })
}

fn import_jude_exception<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    let module = py.import("jude.exceptions")?;
    module.getattr(name)
}
