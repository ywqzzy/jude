//! Re-scannable, zero-copy-ish materialized scan — a custom DuckDB table
//! function backing the *output* side of a materialization boundary.
//!
//! When a UDF / multimodal / aggregate-UDF boundary produces batches, the naive
//! lowering copies them into a DuckDB TEMP TABLE so downstream SQL can reference
//! them by name. That doubles memory and forces a full materialize. Instead we
//! register the held `Arc<Vec<RecordBatch>>` under an id and expose it through
//! `jude_scan(<id>)`, a table function that *re-emits* the batches on each scan
//! (`STANDARD_VECTOR_SIZE` slices) without persisting a DuckDB-side copy — so
//! downstream SQL pipelines batch-by-batch (Model-A-over-UDF). It stays
//! re-scannable (each scan re-inits its cursor) and lifetime-safe (the registry
//! holds the `Arc`; the scan source is registered around one execution and
//! unregistered after).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use duckdb::arrow::datatypes::SchemaRef;
use duckdb::arrow::record_batch::RecordBatch;
use duckdb::core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId};
use duckdb::vtab::arrow::{record_batch_to_duckdb_data_chunk, to_duckdb_logical_type};
use duckdb::vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab};
use once_cell::sync::Lazy;

type Entry = (Arc<Vec<RecordBatch>>, SchemaRef);

static REGISTRY: Lazy<Mutex<HashMap<u64, Entry>>> = Lazy::new(|| Mutex::new(HashMap::new()));
static COUNTER: AtomicU64 = AtomicU64::new(1);

/// Register batches for scanning; returns the id to embed in `jude_scan(<id>)`.
pub fn register(batches: Arc<Vec<RecordBatch>>, schema: SchemaRef) -> u64 {
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    REGISTRY.lock().unwrap().insert(id, (batches, schema));
    id
}

/// Drop a registered scan source (call once the query that scanned it is done).
pub fn unregister(id: u64) {
    REGISTRY.lock().unwrap().remove(&id);
}

pub struct MaterializedScanVTab;

pub struct ScanBind {
    batches: Arc<Vec<RecordBatch>>,
}

pub struct ScanInit {
    /// (batch_index, row_offset_within_batch)
    cursor: Mutex<(usize, usize)>,
    vector_size: usize,
}

impl VTab for MaterializedScanVTab {
    type BindData = ScanBind;
    type InitData = ScanInit;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn std::error::Error>> {
        let value = bind.get_parameter(0);
        if value.is_null() {
            return Err("jude_scan: id must not be NULL".into());
        }
        let id = value.to_uint64();
        let (batches, schema) = REGISTRY
            .lock()
            .unwrap()
            .get(&id)
            .cloned()
            .ok_or_else(|| format!("jude_scan: unknown source id {id}"))?;
        for f in schema.fields() {
            bind.add_result_column(f.name(), to_duckdb_logical_type(f.data_type())?);
        }
        Ok(ScanBind { batches })
    }

    fn init(info: &InitInfo) -> Result<Self::InitData, Box<dyn std::error::Error>> {
        // Single-threaded emission keeps the cursor simple.
        info.set_max_threads(1);
        let vector_size = unsafe { duckdb::ffi::duckdb_vector_size() } as usize;
        if vector_size == 0 {
            return Err("DuckDB vector size must be greater than zero".into());
        }
        Ok(ScanInit {
            cursor: Mutex::new((0, 0)),
            vector_size,
        })
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let bind = func.get_bind_data();
        let init = func.get_init_data();
        let batches = &bind.batches;
        let mut cur = init.cursor.lock().unwrap();
        let (mut bi, mut off) = *cur;
        loop {
            if bi >= batches.len() {
                output.set_len(0);
                *cur = (bi, off);
                return Ok(());
            }
            let rb = &batches[bi];
            if off >= rb.num_rows() {
                bi += 1;
                off = 0;
                continue;
            }
            // Emit at most one vector's worth of rows (zero-copy slice).
            let len = (rb.num_rows() - off).min(init.vector_size);
            record_batch_to_duckdb_data_chunk(&rb.slice(off, len), output)?;
            off += len;
            if off >= rb.num_rows() {
                bi += 1;
                off = 0;
            }
            *cur = (bi, off);
            return Ok(());
        }
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![LogicalTypeHandle::from(LogicalTypeId::UBigint)])
    }
}
