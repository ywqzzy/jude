//! Audit log — durable execution records in an embedded redb store.
//!
//! `observe::MetricsRegistry` is in-memory (bounded ring buffers, lost on
//! restart). This module persists *every* execution — queries, distributed
//! stages, pipelines — to a redb file so the history survives restarts and can
//! be inspected from the dashboard (`/api/audit`). Each record is a JSON blob
//! keyed by a monotonic id; a second table indexes ids by finish-time so listing
//! newest-first is cheap.
//!
//! redb is a pure-Rust embedded KV store (no server, single file, ACID). All
//! access goes through one process-global `Database` behind a `Mutex`; every
//! method is a small transaction. Timestamps come from Python (Rust stays
//! clock-free), matching `observe`.

use std::sync::{Mutex, OnceLock};

use pyo3::prelude::*;
use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition};
use serde_json::Value;

// id -> full record JSON
const RECORDS: TableDefinition<u64, &str> = TableDefinition::new("audit_records");
// monotonic sequence counter (single row at key 0)
const META: TableDefinition<&str, u64> = TableDefinition::new("audit_meta");

struct AuditState {
    db: Database,
    seq: u64,
}

static STATE: OnceLock<Mutex<Option<AuditState>>> = OnceLock::new();

fn state() -> &'static Mutex<Option<AuditState>> {
    STATE.get_or_init(|| Mutex::new(None))
}

/// Open (or create) the redb database at `path`, loading the persisted seq.
fn open_db(path: &str) -> Result<AuditState, String> {
    let db = Database::create(path).map_err(|e| format!("redb open {path}: {e}"))?;
    // Ensure tables exist + read back the max id so ids stay monotonic.
    let seq = {
        let wtxn = db.begin_write().map_err(|e| e.to_string())?;
        {
            let _ = wtxn.open_table(RECORDS).map_err(|e| e.to_string())?;
            let _ = wtxn.open_table(META).map_err(|e| e.to_string())?;
        }
        wtxn.commit().map_err(|e| e.to_string())?;
        let rtxn = db.begin_read().map_err(|e| e.to_string())?;
        let meta = rtxn.open_table(META).map_err(|e| e.to_string())?;
        meta.get("seq")
            .map_err(|e| e.to_string())?
            .map(|v| v.value())
            .unwrap_or(0)
    };
    Ok(AuditState { db, seq })
}

/// Durable execution/audit log backed by redb.
#[pyclass(module = "jude.observe")]
pub struct AuditLog;

#[pymethods]
impl AuditLog {
    /// Open the audit store at `path` (created if absent). Idempotent: a second
    /// open with the same/greater scope reuses the process-global handle.
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let mut guard = state().lock().unwrap();
        if guard.is_none() {
            let st = open_db(path).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            *guard = Some(st);
        }
        Ok(AuditLog)
    }

    /// Append a record (a JSON object string). Returns its assigned id. The
    /// record JSON is stored as-is with an injected `audit_id`.
    fn record(&self, record_json: &str) -> PyResult<u64> {
        let mut guard = state().lock().unwrap();
        let st = guard
            .as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("audit store not open"))?;
        st.seq += 1;
        let id = st.seq;
        // Inject audit_id into the stored JSON so readers always have it.
        let mut val: Value = serde_json::from_str(record_json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("bad record json: {e}"))
        })?;
        if let Some(obj) = val.as_object_mut() {
            obj.insert("audit_id".into(), Value::from(id));
        }
        let stored = serde_json::to_string(&val).unwrap_or_else(|_| record_json.to_string());

        let wtxn = st.db.begin_write().map_err(pyerr)?;
        {
            let mut recs = wtxn.open_table(RECORDS).map_err(pyerr)?;
            recs.insert(id, stored.as_str()).map_err(pyerr)?;
            let mut meta = wtxn.open_table(META).map_err(pyerr)?;
            meta.insert("seq", id).map_err(pyerr)?;
        }
        wtxn.commit().map_err(pyerr)?;
        Ok(id)
    }

    /// Get one record by id as a JSON string (None if absent).
    fn get(&self, id: u64) -> PyResult<Option<String>> {
        let guard = state().lock().unwrap();
        let st = guard
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("audit store not open"))?;
        let rtxn = st.db.begin_read().map_err(pyerr)?;
        let recs = rtxn.open_table(RECORDS).map_err(pyerr)?;
        Ok(recs.get(id).map_err(pyerr)?.map(|v| v.value().to_string()))
    }

    /// List up to `limit` records newest-first, optionally filtered by `kind`
    /// and/or `status`. Returns a JSON array string.
    #[pyo3(signature = (limit=100, kind=None, status=None))]
    fn list(&self, limit: usize, kind: Option<&str>, status: Option<&str>) -> PyResult<String> {
        let guard = state().lock().unwrap();
        let st = guard
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("audit store not open"))?;
        let rtxn = st.db.begin_read().map_err(pyerr)?;
        let recs = rtxn.open_table(RECORDS).map_err(pyerr)?;
        let mut out: Vec<Value> = Vec::new();
        // iterate descending by id (id is monotonic with insert/finish order)
        for item in recs.iter().map_err(pyerr)?.rev() {
            let (_k, v) = item.map_err(pyerr)?;
            let val: Value = match serde_json::from_str(v.value()) {
                Ok(x) => x,
                Err(_) => continue,
            };
            if let Some(k) = kind {
                if val.get("kind").and_then(|x| x.as_str()) != Some(k) {
                    continue;
                }
            }
            if let Some(s) = status {
                if val.get("status").and_then(|x| x.as_str()) != Some(s) {
                    continue;
                }
            }
            out.push(val);
            if out.len() >= limit {
                break;
            }
        }
        Ok(serde_json::to_string(&out).unwrap_or_else(|_| "[]".into()))
    }

    /// Total number of records stored.
    fn count(&self) -> PyResult<u64> {
        let guard = state().lock().unwrap();
        let st = guard
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("audit store not open"))?;
        let rtxn = st.db.begin_read().map_err(pyerr)?;
        let recs = rtxn.open_table(RECORDS).map_err(pyerr)?;
        Ok(recs.len().map_err(pyerr)?)
    }

    /// Aggregate stats for the dashboard header: totals by status/kind.
    fn stats_json(&self) -> PyResult<String> {
        let guard = state().lock().unwrap();
        let st = guard
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("audit store not open"))?;
        let rtxn = st.db.begin_read().map_err(pyerr)?;
        let recs = rtxn.open_table(RECORDS).map_err(pyerr)?;
        let (mut total, mut done, mut error, mut rows) = (0u64, 0u64, 0u64, 0u64);
        for item in recs.iter().map_err(pyerr)? {
            let (_k, v) = item.map_err(pyerr)?;
            if let Ok(val) = serde_json::from_str::<Value>(v.value()) {
                total += 1;
                match val.get("status").and_then(|x| x.as_str()) {
                    Some("done") => done += 1,
                    Some("error") => error += 1,
                    _ => {}
                }
                rows += val.get("rows").and_then(|x| x.as_u64()).unwrap_or(0);
            }
        }
        let stats = serde_json::json!({
            "total": total, "done": done, "error": error, "rows_total": rows
        });
        Ok(serde_json::to_string(&stats).unwrap_or_else(|_| "{}".into()))
    }

    /// Delete all records (keeps the file/tables). For tests / retention resets.
    fn clear(&self) -> PyResult<()> {
        let mut guard = state().lock().unwrap();
        let st = guard
            .as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("audit store not open"))?;
        let wtxn = st.db.begin_write().map_err(pyerr)?;
        {
            // redb has no truncate; recreate the table by draining keys.
            let mut recs = wtxn.open_table(RECORDS).map_err(pyerr)?;
            let ids: Vec<u64> = recs
                .iter()
                .map_err(pyerr)?
                .filter_map(|r| r.ok().map(|(k, _v)| k.value()))
                .collect();
            for id in ids {
                recs.remove(id).map_err(pyerr)?;
            }
        }
        wtxn.commit().map_err(pyerr)?;
        Ok(())
    }
}

fn pyerr<E: std::fmt::Display>(e: E) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

/// Register audit classes into a module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AuditLog>()?;
    Ok(())
}
