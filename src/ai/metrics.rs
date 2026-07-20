use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct TokenMetricsEntry {
    pub protocol: String,
    pub model: String,
    pub provider: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub requests: u64,
}

type MetricsKey = (String, String, String);
type Callback = Arc<dyn Fn(&TokenMetricsEntry) + Send + Sync>;

static COUNTERS: Lazy<Mutex<HashMap<MetricsKey, TokenMetricsEntry>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
static CALLBACK: Lazy<Mutex<Option<Callback>>> = Lazy::new(|| Mutex::new(None));

pub fn record_token_metrics(
    protocol: &str,
    model: &str,
    provider: &str,
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    total_tokens: Option<u64>,
) {
    let key = (
        protocol.to_string(),
        model.to_string(),
        provider.to_string(),
    );
    {
        let mut counters = COUNTERS.lock();
        let entry = counters
            .entry(key.clone())
            .or_insert_with(|| TokenMetricsEntry {
                protocol: protocol.into(),
                model: model.into(),
                provider: provider.into(),
                input_tokens: 0,
                output_tokens: 0,
                total_tokens: 0,
                requests: 0,
            });
        if let Some(v) = input_tokens {
            entry.input_tokens += v;
        }
        if let Some(v) = output_tokens {
            entry.output_tokens += v;
        }
        if let Some(v) = total_tokens {
            entry.total_tokens += v;
        }
        entry.requests += 1;
    }
    if let Some(cb) = CALLBACK.lock().as_ref() {
        let entry = COUNTERS.lock().get(&key).cloned();
        if let Some(e) = entry {
            cb(&e);
        }
    }
}

pub fn get_token_metrics() -> Vec<TokenMetricsEntry> {
    COUNTERS.lock().values().cloned().collect()
}

pub fn reset_token_metrics() {
    COUNTERS.lock().clear();
}

pub fn set_token_metrics_callback(callback: Option<Callback>) {
    *CALLBACK.lock() = callback;
}

use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
#[pyo3(name = "get_token_metrics")]
pub fn get_token_metrics_py(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let entries = get_token_metrics();
    let list = pyo3::types::PyList::empty(py);
    for entry in &entries {
        let d = PyDict::new(py);
        d.set_item("protocol", &entry.protocol)?;
        d.set_item("model", &entry.model)?;
        d.set_item("provider", &entry.provider)?;
        d.set_item("input_tokens", entry.input_tokens)?;
        d.set_item("output_tokens", entry.output_tokens)?;
        d.set_item("total_tokens", entry.total_tokens)?;
        d.set_item("requests", entry.requests)?;
        list.append(d)?;
    }
    Ok(list.into())
}

#[pyfunction]
#[pyo3(name = "reset_token_metrics")]
pub fn reset_token_metrics_py() -> PyResult<()> {
    reset_token_metrics();
    Ok(())
}
