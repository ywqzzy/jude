use pyo3::prelude::*;

pub mod batch_wrappers;
pub mod functions;
pub mod metrics;
pub mod options;
pub mod protocols;
pub mod provider;
pub mod providers;
pub mod retry;
pub mod typing;

pub fn register_bound(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(functions::embed_text_py, m)?)?;
    m.add_function(wrap_pyfunction!(functions::classify_text_py, m)?)?;
    m.add_function(wrap_pyfunction!(functions::prompt_py, m)?)?;
    m.add_function(wrap_pyfunction!(functions::embed_py, m)?)?;

    m.add_function(wrap_pyfunction!(provider::load_provider_py, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::get_token_metrics_py, m)?)?;
    m.add_function(wrap_pyfunction!(metrics::reset_token_metrics_py, m)?)?;

    Ok(())
}
