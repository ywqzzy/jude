use crate::connection::Connection;
use crate::error::Error;
use pyo3::conversion::IntoPyObjectExt;
use pyo3::prelude::*;
use std::cell::RefCell;

use duckdb::core::Inserter;
use duckdb::core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId};
use duckdb::types::DuckString;
use duckdb::vscalar::{ScalarFunctionSignature, VScalar};
use duckdb::vtab::arrow::WritableVector;
use std::sync::Arc;

thread_local! {
    static PENDING_SIG: RefCell<Option<(Vec<LogicalTypeId>, LogicalTypeId)>> = RefCell::new(None);
}

fn type_str_to_id(s: &str) -> LogicalTypeId {
    match s.to_uppercase().as_str() {
        "BOOLEAN" | "BOOL" => LogicalTypeId::Boolean,
        "INTEGER" | "INT" => LogicalTypeId::Integer,
        "BIGINT" | "LONG" => LogicalTypeId::Bigint,
        "FLOAT" => LogicalTypeId::Float,
        "DOUBLE" | "REAL" => LogicalTypeId::Double,
        "VARCHAR" | "TEXT" | "STRING" => LogicalTypeId::Varchar,
        "BLOB" | "BINARY" => LogicalTypeId::Blob,
        "DATE" => LogicalTypeId::Date,
        "TIMESTAMP" => LogicalTypeId::Timestamp,
        _ => LogicalTypeId::Varchar,
    }
}

#[pyfunction]
#[pyo3(signature = (func, alias=None, connection=None, replace=false, parameters=None, return_dtype=None, vectorized=false, exception_handling=None, null_handling=None, side_effects=false))]
pub fn attach_function(
    func: &Bound<'_, PyAny>,
    alias: Option<&str>,
    connection: Option<&Connection>,
    #[allow(unused_variables)] replace: bool,
    parameters: Option<Vec<String>>,
    return_dtype: Option<String>,
    vectorized: bool,
    exception_handling: Option<String>,
    null_handling: Option<String>,
    side_effects: bool,
) -> PyResult<()> {
    let func_name = alias.map(String::from).unwrap_or_else(|| {
        func.getattr("__name__")
            .ok()
            .and_then(|n| n.extract::<String>().ok())
            .unwrap_or_else(|| "udf".to_string())
    });
    let conn = match connection {
        Some(c) => c,
        None => {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "A connection is required",
            ))
        }
    };
    let param_strs = parameters.unwrap_or_else(|| vec!["VARCHAR".to_string()]);
    let ret_str = return_dtype.unwrap_or_else(|| "VARCHAR".to_string());
    let param_ids: Vec<LogicalTypeId> = param_strs.iter().map(|s| type_str_to_id(s)).collect();
    let ret_id = type_str_to_id(&ret_str);
    // exception_handling: "return_null" (a throwing row/chunk -> NULL, continue)
    // vs "forward"/None (re-raise, the default).
    let on_error = match exception_handling
        .as_deref()
        .map(|s| s.to_ascii_lowercase())
    {
        Some(ref s) if s == "return_null" || s == "null" => OnError::ReturnNull,
        _ => OnError::Forward,
    };
    // null_handling: "default" filters NULL-argument rows out (UDF not called);
    // "special"/None passes NULLs through as Python None (the default).
    let null_handling = match null_handling.as_deref().map(|s| s.to_ascii_lowercase()) {
        Some(ref s) if s == "default" => NullHandling::Default,
        _ => NullHandling::Special,
    };
    let py_func: Py<PyAny> = func.into();
    let state = UdfState {
        func: Arc::new(py_func),
        param_types: param_ids.clone(),
        ret_type: ret_id,
        on_error,
        null_handling,
    };

    PENDING_SIG.with(|cell| {
        *cell.borrow_mut() = Some((param_ids.clone(), ret_id));
    });

    // Pick the adapter by two axes: vectorized (arrow) vs row-by-row, and
    // side_effects (volatile — DuckDB won't fold repeated calls) vs pure.
    // `vectorized=True` (type="arrow") hands the UDF whole pyarrow columns (one
    // GIL grab per chunk, full type coverage); otherwise any-arity row-by-row.
    match (vectorized, side_effects) {
        (true, false) => conn
            .inner
            .register_scalar_function_with_state::<PyArrowUdf>(&func_name, &state),
        (true, true) => conn
            .inner
            .register_scalar_function_with_state::<PyArrowUdfVolatile>(&func_name, &state),
        (false, false) => conn
            .inner
            .register_scalar_function_with_state::<PyUdf>(&func_name, &state),
        (false, true) => conn
            .inner
            .register_scalar_function_with_state::<PyUdfVolatile>(&func_name, &state),
    }
    .map_err(Error::DuckDb)?;
    Ok(())
}

#[pyfunction]
pub fn detach_function(alias: &str, connection: Option<&Connection>) -> PyResult<()> {
    let conn = match connection {
        Some(c) => c,
        None => {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "A connection is required",
            ))
        }
    };
    // DuckDB refuses to DROP scalar functions registered via the C API
    // ("internal catalog entry"). The registration is connection-scoped, so
    // treat a failed drop as a no-op rather than surfacing an error.
    let _ = conn
        .inner
        .execute_batch(&format!("DROP FUNCTION IF EXISTS {alias}"));
    Ok(())
}

#[derive(Clone, Copy, PartialEq)]
enum OnError {
    /// Re-raise the Python exception (default).
    Forward,
    /// A throwing row (native path) or chunk (arrow path) becomes NULL; the scan
    /// continues. For billion-row jobs where one corrupt input mustn't abort all.
    ReturnNull,
}

#[derive(Clone, Copy, PartialEq)]
enum NullHandling {
    /// The UDF sees NULL arguments as Python `None` and decides (default).
    Special,
    /// Rows with ANY NULL argument are skipped — the UDF is not called for them
    /// and their result is NULL. Matches DuckDB's DEFAULT null handling.
    Default,
}

#[derive(Clone)]
struct UdfState {
    func: Arc<Py<PyAny>>,
    param_types: Vec<LogicalTypeId>,
    ret_type: LogicalTypeId,
    on_error: OnError,
    null_handling: NullHandling,
}

/// Set result rows to NULL wherever any input column of `batch` is NULL — the
/// arrow-path implementation of `null_handling=default`. Uses `nullif` against a
/// boolean mask that is true exactly where some argument is null.
fn null_out_arg_null_rows(
    batch: &arrow::array::RecordBatch,
    result: &arrow::array::ArrayRef,
) -> Result<arrow::array::ArrayRef, Box<dyn std::error::Error>> {
    use arrow::array::{Array, BooleanArray};
    let n = result.len();
    if batch.num_columns() == 0 {
        return Ok(result.clone());
    }
    let mut any_null = vec![false; n];
    for c in 0..batch.num_columns() {
        let col = batch.column(c);
        if col.null_count() == 0 {
            continue;
        }
        for (i, slot) in any_null.iter_mut().enumerate() {
            if col.is_null(i) {
                *slot = true;
            }
        }
    }
    if !any_null.iter().any(|b| *b) {
        return Ok(result.clone());
    }
    let mask = BooleanArray::from(any_null);
    // nullif(result, mask): NULL where mask is true, else result.
    Ok(arrow::compute::nullif(result.as_ref(), &mask)?)
}

/// Map a UDF return `LogicalTypeId` to the Arrow `DataType` used to build an
/// all-NULL result when `on_error=return_null` and the whole chunk failed.
fn logical_id_to_arrow(ty: LogicalTypeId) -> arrow::datatypes::DataType {
    use arrow::datatypes::DataType;
    match ty {
        LogicalTypeId::Boolean => DataType::Boolean,
        LogicalTypeId::Tinyint => DataType::Int8,
        LogicalTypeId::Smallint => DataType::Int16,
        LogicalTypeId::Integer => DataType::Int32,
        LogicalTypeId::Bigint => DataType::Int64,
        LogicalTypeId::Float => DataType::Float32,
        LogicalTypeId::Double => DataType::Float64,
        LogicalTypeId::Varchar => DataType::Utf8,
        LogicalTypeId::Blob => DataType::Binary,
        _ => DataType::Null,
    }
}

fn extract_row_value(
    py: Python<'_>,
    input: &mut DataChunkHandle,
    col: usize,
    ty: LogicalTypeId,
    row: usize,
) -> PyResult<Py<PyAny>> {
    let vec = input.flat_vector(col);
    // A SQL NULL argument is passed to the UDF as Python None.
    if vec.row_is_null(row as u64) {
        return Ok(py.None());
    }
    match ty {
        LogicalTypeId::Varchar => {
            let vals =
                unsafe { vec.as_slice_with_len::<duckdb::ffi::duckdb_string_t>(input.len()) };
            let mut ptr = vals[row];
            let s = DuckString::new(&mut ptr).as_str().to_string();
            Ok(s.into_pyobject(py)?.into_any().unbind())
        }
        LogicalTypeId::Integer => {
            let vals = unsafe { vec.as_slice_with_len::<i32>(input.len()) };
            Ok(vals[row].into_pyobject(py)?.into_any().unbind())
        }
        LogicalTypeId::Bigint => {
            let vals = unsafe { vec.as_slice_with_len::<i64>(input.len()) };
            Ok(vals[row].into_pyobject(py)?.into_any().unbind())
        }
        LogicalTypeId::Double => {
            let vals = unsafe { vec.as_slice_with_len::<f64>(input.len()) };
            Ok(vals[row].into_pyobject(py)?.into_any().unbind())
        }
        LogicalTypeId::Boolean => {
            let vals = unsafe { vec.as_slice_with_len::<u8>(input.len()) };
            let v: bool = vals[row] != 0;
            Ok(v.into_py_any(py)?)
        }
        _ => {
            let vals =
                unsafe { vec.as_slice_with_len::<duckdb::ffi::duckdb_string_t>(input.len()) };
            let mut ptr = vals[row];
            let s = DuckString::new(&mut ptr).as_str().to_string();
            Ok(s.into_pyobject(py)?.into_any().unbind())
        }
    }
}

fn write_row_output(
    output: &mut dyn WritableVector,
    row: usize,
    py_val: &Bound<'_, PyAny>,
    ret_type: LogicalTypeId,
    total_len: usize,
) -> PyResult<()> {
    // A UDF returning Python None yields SQL NULL.
    if py_val.is_none() {
        output.flat_vector().set_null(row);
        return Ok(());
    }
    match ret_type {
        LogicalTypeId::Varchar => {
            let s: String = py_val.extract().unwrap_or_default();
            output.flat_vector().insert(row, s.as_str());
        }
        LogicalTypeId::Integer => {
            let v: i32 = py_val.extract().unwrap_or(0);
            let mut vec = output.flat_vector();
            let slice = unsafe { vec.as_mut_slice_with_len::<i32>(total_len) };
            slice[row] = v;
        }
        LogicalTypeId::Bigint => {
            let v: i64 = py_val.extract().unwrap_or(0);
            let mut vec = output.flat_vector();
            let slice = unsafe { vec.as_mut_slice_with_len::<i64>(total_len) };
            slice[row] = v;
        }
        LogicalTypeId::Double => {
            let v: f64 = py_val.extract().unwrap_or(0.0);
            let mut vec = output.flat_vector();
            let slice = unsafe { vec.as_mut_slice_with_len::<f64>(total_len) };
            slice[row] = v;
        }
        LogicalTypeId::Boolean => {
            let v: bool = py_val.extract().unwrap_or(false);
            let mut vec = output.flat_vector();
            let slice = unsafe { vec.as_mut_slice_with_len::<bool>(total_len) };
            slice[row] = v;
        }
        _ => {
            let s: String = py_val.extract().unwrap_or_default();
            output.flat_vector().insert(row, s.as_str());
        }
    }
    Ok(())
}

/// The row-by-row invoke body (shared by the volatile and non-volatile adapters).
fn invoke_row(
    state: &UdfState,
    input: &mut DataChunkHandle,
    output: &mut dyn WritableVector,
) -> Result<(), Box<dyn std::error::Error>> {
    Python::attach(|py| -> Result<(), Box<dyn std::error::Error>> {
        let len = input.len();
        let arity = state.param_types.len();
        for row in 0..len {
            // null_handling=default: skip rows with any NULL argument — the
            // UDF isn't called and the result is NULL.
            if state.null_handling == NullHandling::Default
                && (0..arity).any(|col| input.flat_vector(col).row_is_null(row as u64))
            {
                output.flat_vector().set_null(row);
                continue;
            }
            let mut args = Vec::with_capacity(arity);
            for col in 0..arity {
                args.push(extract_row_value(
                    py,
                    input,
                    col,
                    state.param_types[col],
                    row,
                )?);
            }
            let tuple = pyo3::types::PyTuple::new(py, args)?;
            match state.func.call1(py, &tuple) {
                Ok(result) => {
                    let result_bound = result.bind(py);
                    write_row_output(output, row, result_bound, state.ret_type, len)?;
                }
                Err(e) => {
                    // on_error=return_null: a throwing row becomes NULL, scan continues.
                    if state.on_error == OnError::ReturnNull {
                        output.flat_vector().set_null(row);
                    } else {
                        return Err(e.into());
                    }
                }
            }
        }
        Ok(())
    })
}

/// The vectorized arrow invoke body (shared by the volatile and non-volatile
/// adapters).
fn invoke_arrow(
    state: &UdfState,
    input: &mut DataChunkHandle,
    output: &mut dyn WritableVector,
) -> Result<(), Box<dyn std::error::Error>> {
    use duckdb::vtab::arrow::{data_chunk_to_arrow, write_arrow_array_to_vector};
    let batch = data_chunk_to_arrow(input)?;
    let n_rows = batch.num_rows();
    let out_array_res = Python::attach(|py| -> PyResult<arrow::array::ArrayRef> {
        let table = crate::arrow_ffi::batches_to_pyarrow_table(
            py,
            std::slice::from_ref(&batch),
            &batch.schema(),
        )?;
        let table = table.bind(py);
        let n_cols = batch.num_columns();
        let mut args: Vec<Bound<'_, PyAny>> = Vec::with_capacity(n_cols);
        for c in 0..n_cols {
            args.push(table.call_method1("column", (c,))?);
        }
        let result = state.func.call1(py, pyo3::types::PyTuple::new(py, args)?)?;
        let pa = py.import("pyarrow")?;
        let result_bound = result.bind(py);
        let names = pyo3::types::PyList::new(py, ["__jude_udf_out"])?;
        let out_table = pa
            .getattr("table")?
            .call1((pyo3::types::PyList::new(py, [result_bound])?, names))?;
        let batches = crate::arrow_ffi::py_arrow_to_batches(py, &out_table)?;
        let arrays: Vec<arrow::array::ArrayRef> =
            batches.iter().map(|b| b.column(0).clone()).collect();
        let refs: Vec<&dyn arrow::array::Array> = arrays.iter().map(|a| a.as_ref()).collect();
        Ok(arrow::compute::concat(&refs).map_err(|e| PyErr::from(Error::Arrow(e)))?)
    });
    let out_array = match out_array_res {
        Ok(a) => a,
        Err(e) => {
            if state.on_error == OnError::ReturnNull {
                arrow::array::new_null_array(&logical_id_to_arrow(state.ret_type), n_rows)
            } else {
                return Err(e.into());
            }
        }
    };
    let out_array = if state.null_handling == NullHandling::Default {
        null_out_arg_null_rows(&batch, &out_array)?
    } else {
        out_array
    };
    write_arrow_array_to_vector(&out_array, output)?;
    Ok(())
}

/// The signatures body (identical for every adapter; reads PENDING_SIG).
fn udf_signatures() -> Vec<ScalarFunctionSignature> {
    PENDING_SIG.with(|cell| {
        let (param_ids, ret_id) = cell.borrow_mut().take().unwrap();
        let params: Vec<LogicalTypeHandle> = param_ids
            .iter()
            .map(|id| LogicalTypeHandle::from(*id))
            .collect();
        vec![ScalarFunctionSignature::exact(
            params,
            LogicalTypeHandle::from(ret_id),
        )]
    })
}

/// Generate a `VScalar` adapter struct that delegates to `$body`, with a fixed
/// `volatile()`. We need distinct types because `VScalar::volatile()` is a
/// type-level method — `side_effects=True` selects the volatile variant so
/// DuckDB won't fold repeated calls to a nondeterministic UDF.
macro_rules! udf_adapter {
    ($name:ident, $body:ident, $volatile:literal) => {
        struct $name;
        impl VScalar for $name {
            type State = UdfState;
            fn invoke(
                state: &Self::State,
                input: &mut DataChunkHandle,
                output: &mut dyn WritableVector,
            ) -> Result<(), Box<dyn std::error::Error>> {
                $body(state, input, output)
            }
            fn signatures() -> Vec<ScalarFunctionSignature> {
                udf_signatures()
            }
            fn volatile() -> bool {
                $volatile
            }
        }
    };
}

// Row-by-row (any arity) and vectorized (arrow) adapters, each in a
// non-volatile and a volatile (side_effects=True) flavor.
udf_adapter!(PyUdf, invoke_row, false);
udf_adapter!(PyUdfVolatile, invoke_row, true);
udf_adapter!(PyArrowUdf, invoke_arrow, false);
udf_adapter!(PyArrowUdfVolatile, invoke_arrow, true);
