use super::typing::*;
use async_trait::async_trait;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub enum MessagePart {
    Text(String),
    Image { mime_type: String, data: Vec<u8> },
}

#[derive(Clone, Debug)]
pub enum PromptMessages {
    Text(String),
    Multimodal(Vec<MessagePart>),
}

#[async_trait]
pub trait TextEmbedder: Send + Sync {
    async fn embed_text(&self, text: Vec<String>) -> Result<Vec<Embedding>, crate::error::Error>;
}

#[async_trait]
pub trait TextClassifier: Send + Sync {
    async fn classify_text(
        &self,
        text: Vec<String>,
        labels: Vec<Label>,
    ) -> Result<Vec<Option<Label>>, crate::error::Error>;
}

#[async_trait]
pub trait Prompter: Send + Sync {
    async fn prompt(&self, messages: PromptMessages)
        -> Result<Option<String>, crate::error::Error>;
}

pub trait TextEmbedderDescriptor: Descriptor<Instance = Arc<dyn TextEmbedder>> {
    fn get_dimensions(&self) -> Result<EmbeddingDimensions, crate::error::Error>;
    fn is_async(&self) -> bool {
        true
    }
}

pub trait TextClassifierDescriptor: Descriptor<Instance = Arc<dyn TextClassifier>> {}

pub trait PrompterDescriptor: Descriptor<Instance = Arc<dyn Prompter>> {}
