use super::{MaterializedResult, Runner};
use crate::relation::Relation;
use pyo3::prelude::*;
use uuid::Uuid;

pub struct LocalRunner {
    pub num_workers: usize,
}

impl Default for LocalRunner {
    fn default() -> Self {
        Self { num_workers: 1 }
    }
}

impl Runner for LocalRunner {
    fn name(&self) -> &str {
        "local"
    }

    fn run_iter(&self, py: Python<'_>, relation: &Relation) -> PyResult<Vec<MaterializedResult>> {
        let batches = relation.collect_batches(py)?;
        Ok(batches
            .into_iter()
            .map(|batch| {
                let num_rows = batch.num_rows();
                MaterializedResult {
                    partition_id: Uuid::new_v4().to_string(),
                    num_rows,
                    batch,
                }
            })
            .collect())
    }

    fn run_iter_tables(
        &self,
        py: Python<'_>,
        relation: &Relation,
    ) -> PyResult<Vec<duckdb::arrow::record_batch::RecordBatch>> {
        relation.collect_batches(py)
    }
}
