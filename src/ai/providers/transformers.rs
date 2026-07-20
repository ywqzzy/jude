use super::super::protocols::*;
use super::super::provider::Provider;
use super::super::typing::*;
use crate::error::Error;
use std::sync::Arc;

pub fn factory(name: Option<&str>, _options: Options) -> Result<Arc<dyn Provider>, Error> {
    Ok(Arc::new(TransformersProvider {
        name: name.unwrap_or("transformers").to_string(),
    }))
}

pub struct TransformersProvider {
    name: String,
}

impl Provider for TransformersProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_text_embedder(
        &self,
        model: Option<&str>,
        dimensions: Option<usize>,
        _options: &Options,
    ) -> Result<Arc<dyn TextEmbedderDescriptor>, Error> {
        Ok(Arc::new(TransformersTextEmbedderDescriptor {
            model: model
                .unwrap_or("sentence-transformers/all-MiniLM-L6-v2")
                .to_string(),
            dimensions,
        }))
    }

    fn get_text_classifier(
        &self,
        model: Option<&str>,
        _options: &Options,
    ) -> Result<Arc<dyn TextClassifierDescriptor>, Error> {
        Ok(Arc::new(TransformersTextClassifierDescriptor {
            model: model.unwrap_or("facebook/bart-large-mnli").to_string(),
        }))
    }

    fn get_prompter(
        &self,
        _model: Option<&str>,
        _system_message: Option<&str>,
        _options: &Options,
    ) -> Result<Arc<dyn PrompterDescriptor>, Error> {
        Err(Error::Other(
            "Transformers provider does not support prompting".into(),
        ))
    }
}

pub struct TransformersTextEmbedderDescriptor {
    pub model: String,
    pub dimensions: Option<usize>,
}

impl Descriptor for TransformersTextEmbedderDescriptor {
    type Instance = Arc<dyn TextEmbedder>;

    fn get_provider(&self) -> &str {
        "transformers"
    }
    fn get_model(&self) -> &str {
        &self.model
    }
    fn get_options(&self) -> &Options {
        &Options::Null
    }

    fn instantiate(&self) -> Result<Self::Instance, Error> {
        Err(Error::Other("Transformers provider requires Python runtime for model inference. Use 'openai', 'anthropic', or 'google' provider instead.".into()))
    }
}

impl TextEmbedderDescriptor for TransformersTextEmbedderDescriptor {
    fn get_dimensions(&self) -> Result<EmbeddingDimensions, Error> {
        let size = self.dimensions.unwrap_or(384);
        Ok(EmbeddingDimensions::default_f32(size))
    }
    fn is_async(&self) -> bool {
        false
    }
}

pub struct TransformersTextClassifierDescriptor {
    pub model: String,
}

impl Descriptor for TransformersTextClassifierDescriptor {
    type Instance = Arc<dyn TextClassifier>;

    fn get_provider(&self) -> &str {
        "transformers"
    }
    fn get_model(&self) -> &str {
        &self.model
    }
    fn get_options(&self) -> &Options {
        &Options::Null
    }

    fn instantiate(&self) -> Result<Self::Instance, Error> {
        Err(Error::Other(
            "Transformers provider requires Python runtime for model inference.".into(),
        ))
    }
}

impl TextClassifierDescriptor for TransformersTextClassifierDescriptor {}
