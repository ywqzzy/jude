use super::protocols::*;
use super::typing::*;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

pub trait Provider: Send + Sync {
    fn name(&self) -> &str;

    fn get_text_embedder(
        &self,
        model: Option<&str>,
        dimensions: Option<usize>,
        options: &Options,
    ) -> Result<Arc<dyn TextEmbedderDescriptor>, crate::error::Error>;

    fn get_text_classifier(
        &self,
        _model: Option<&str>,
        _options: &Options,
    ) -> Result<Arc<dyn TextClassifierDescriptor>, crate::error::Error> {
        Err(crate::error::Error::Other(format!(
            "Provider '{}' does not support text classification",
            self.name()
        )))
    }

    fn get_prompter(
        &self,
        model: Option<&str>,
        system_message: Option<&str>,
        options: &Options,
    ) -> Result<Arc<dyn PrompterDescriptor>, crate::error::Error>;
}

type ProviderFactory = fn(Option<&str>, Options) -> Result<Arc<dyn Provider>, crate::error::Error>;

static PROVIDERS: Lazy<RwLock<HashMap<String, ProviderFactory>>> = Lazy::new(|| {
    let mut m = HashMap::new();
    #[cfg(feature = "ai-openai")]
    {
        m.insert(
            "openai".into(),
            super::providers::openai::factory as ProviderFactory,
        );
    }
    #[cfg(feature = "ai-anthropic")]
    {
        m.insert(
            "anthropic".into(),
            super::providers::anthropic::factory as ProviderFactory,
        );
    }
    #[cfg(feature = "ai-google")]
    {
        m.insert(
            "google".into(),
            super::providers::google::factory as ProviderFactory,
        );
    }
    m.insert(
        "transformers".into(),
        super::providers::transformers::factory as ProviderFactory,
    );
    RwLock::new(m)
});

pub fn load_provider(
    provider: &str,
    name: Option<&str>,
    options: Options,
) -> Result<Arc<dyn Provider>, crate::error::Error> {
    let registry = PROVIDERS.read();
    let factory = registry
        .get(provider)
        .ok_or_else(|| crate::error::Error::UnsupportedProvider(provider.to_string()))?;
    factory(name, options)
}

pub fn available_providers() -> Vec<String> {
    PROVIDERS.read().keys().cloned().collect()
}

use pyo3::prelude::*;

#[pyfunction]
#[pyo3(name = "load_provider")]
pub fn load_provider_py(py: Python<'_>, provider: &str, name: Option<&str>) -> PyResult<Py<PyAny>> {
    let _prov = load_provider(provider, name, Options::Null)?;
    Ok(py.None())
}
