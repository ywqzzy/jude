pub mod local;

use crate::relation::Relation;
use duckdb::arrow::record_batch::RecordBatch;
use pyo3::prelude::*;

pub trait Runner {
    fn name(&self) -> &str;
    fn run_iter(&self, py: Python<'_>, relation: &Relation) -> PyResult<Vec<MaterializedResult>>;
    fn run_iter_tables(&self, py: Python<'_>, relation: &Relation) -> PyResult<Vec<RecordBatch>>;
}

pub struct MaterializedResult {
    pub batch: RecordBatch,
    pub partition_id: String,
    pub num_rows: usize,
}

#[pyclass(name = "MaterializedResult")]
pub struct PyMaterializedResult {
    #[pyo3(get)]
    pub num_rows: usize,
    #[pyo3(get)]
    pub partition_id: String,
}
