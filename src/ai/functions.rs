use super::batch_wrappers::*;
use super::provider::load_provider;
use super::typing::*;
use crate::relation::Relation;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (rel, column, provider=None, model=None, dimensions=None, output_column="embedding"))]
#[pyo3(name = "embed_text")]
pub fn embed_text_py(
    py: Python<'_>,
    rel: &Relation,
    column: &str,
    provider: Option<&str>,
    model: Option<&str>,
    dimensions: Option<usize>,
    output_column: &str,
) -> PyResult<Relation> {
    embed_text(py, rel, column, provider, model, dimensions, output_column)
}

pub fn embed_text(
    py: Python<'_>,
    rel: &Relation,
    column: &str,
    provider: Option<&str>,
    model: Option<&str>,
    dimensions: Option<usize>,
    output_column: &str,
) -> PyResult<Relation> {
    let prov_name = provider.unwrap_or("transformers");
    let prov = load_provider(prov_name, None, Options::Null)?;
    let descriptor = prov.get_text_embedder(model, dimensions, &Options::Null)?;
    let udf_opts = descriptor.get_udf_options();

    let wrapper = std::cell::RefCell::new(EmbedTextBatch::new(
        descriptor,
        column.to_string(),
        output_column.to_string(),
        udf_opts.max_retries,
        udf_opts.on_error,
    ));

    rel.map_batches(py, |batch| {
        wrapper
            .borrow_mut()
            .process_batch(batch)
            .map_err(pyo3::PyErr::from)
    })
}

#[pyfunction]
#[pyo3(signature = (rel, column, labels, provider=None, model=None, output_column="label"))]
#[pyo3(name = "classify_text")]
pub fn classify_text_py(
    py: Python<'_>,
    rel: &Relation,
    column: &str,
    labels: Vec<String>,
    provider: Option<&str>,
    model: Option<&str>,
    output_column: &str,
) -> PyResult<Relation> {
    classify_text(py, rel, column, &labels, provider, model, output_column)
}

pub fn classify_text(
    py: Python<'_>,
    rel: &Relation,
    column: &str,
    labels: &[String],
    provider: Option<&str>,
    model: Option<&str>,
    output_column: &str,
) -> PyResult<Relation> {
    let prov_name = provider.unwrap_or("transformers");
    let prov = load_provider(prov_name, None, Options::Null)?;
    let descriptor = prov.get_text_classifier(model, &Options::Null)?;
    let udf_opts = descriptor.get_udf_options();

    let wrapper = std::cell::RefCell::new(ClassifyTextBatch::new(
        descriptor,
        column.to_string(),
        output_column.to_string(),
        labels.to_vec(),
        udf_opts.max_retries,
        udf_opts.on_error,
    ));

    rel.map_batches(py, |batch| {
        wrapper
            .borrow_mut()
            .process_batch(batch)
            .map_err(pyo3::PyErr::from)
    })
}

#[pyfunction]
#[pyo3(signature = (rel, column, provider=None, model=None, system_message=None, output_column="response"))]
#[pyo3(name = "prompt")]
pub fn prompt_py(
    py: Python<'_>,
    rel: &Relation,
    column: &str,
    provider: Option<&str>,
    model: Option<&str>,
    system_message: Option<&str>,
    output_column: &str,
) -> PyResult<Relation> {
    prompt_relation(
        py,
        rel,
        column,
        provider,
        model,
        system_message,
        output_column,
    )
}

pub fn prompt_relation(
    py: Python<'_>,
    rel: &Relation,
    column: &str,
    provider: Option<&str>,
    model: Option<&str>,
    system_message: Option<&str>,
    output_column: &str,
) -> PyResult<Relation> {
    let prov_name = provider.unwrap_or("openai");
    let prov = load_provider(prov_name, None, Options::Null)?;
    let descriptor = prov.get_prompter(model, system_message, &Options::Null)?;
    let udf_opts = descriptor.get_udf_options();

    let wrapper = std::cell::RefCell::new(PromptBatch::new(
        descriptor,
        column.to_string(),
        output_column.to_string(),
        udf_opts.max_retries,
        udf_opts.on_error,
        system_message.map(String::from),
    ));

    rel.map_batches(py, |batch| {
        wrapper
            .borrow_mut()
            .process_batch(batch)
            .map_err(pyo3::PyErr::from)
    })
}

#[pyfunction]
#[pyo3(signature = (text, provider="openai", model=None, dimensions=None))]
#[pyo3(name = "embed")]
pub fn embed_py(
    py: Python<'_>,
    text: &str,
    provider: &str,
    model: Option<&str>,
    dimensions: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let prov = load_provider(provider, None, Options::Null)?;
    let descriptor = prov.get_text_embedder(model, dimensions, &Options::Null)?;

    let embedder = descriptor.instantiate()?;
    let rt = super::batch_wrappers::async_rt();
    let embeddings = rt.block_on(async { embedder.embed_text(vec![text.to_string()]).await })?;

    let embedding = &embeddings[0];
    let list = pyo3::types::PyList::new(py, embedding)?;
    Ok(list.into())
}
