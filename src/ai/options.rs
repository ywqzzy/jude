use super::typing::Options;
use serde::{Deserialize, Serialize};

pub trait ToDescriptorOptions {
    fn to_descriptor_options(&self) -> Options;
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OpenAIProviderOptions {
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub organization: Option<String>,
    pub timeout: Option<f64>,
    pub concurrency: Option<usize>,
    pub max_api_concurrency: Option<usize>,
}

impl ToDescriptorOptions for OpenAIProviderOptions {
    fn to_descriptor_options(&self) -> Options {
        let mut opts = serde_json::Map::new();
        if let Some(ref v) = self.base_url {
            opts.insert("base_url".into(), v.clone().into());
        }
        if let Some(ref v) = self.api_key {
            opts.insert("api_key".into(), v.clone().into());
        }
        if let Some(ref v) = self.organization {
            opts.insert("organization".into(), v.clone().into());
        }
        if let Some(v) = self.timeout {
            opts.insert("timeout".into(), v.into());
        }
        if let Some(v) = self.concurrency {
            opts.insert("actor_number".into(), (v as u64).into());
        }
        if let Some(v) = self.max_api_concurrency {
            opts.insert("max_api_concurrency".into(), (v as u64).into());
        }
        Options::Object(opts)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OpenAIPromptOptions {
    pub use_chat_completions: Option<bool>,
    pub max_output_tokens: Option<u64>,
    pub max_tokens: Option<u64>,
    pub temperature: Option<f64>,
    pub on_error: Option<String>,
}

impl ToDescriptorOptions for OpenAIPromptOptions {
    fn to_descriptor_options(&self) -> Options {
        let mut opts = serde_json::Map::new();
        if let Some(v) = self.use_chat_completions {
            opts.insert("use_chat_completions".into(), v.into());
        }
        if let Some(v) = self.max_output_tokens {
            opts.insert("max_output_tokens".into(), v.into());
        }
        if let Some(v) = self.max_tokens {
            opts.insert("max_tokens".into(), v.into());
        }
        if let Some(v) = self.temperature {
            opts.insert("temperature".into(), v.into());
        }
        if let Some(ref v) = self.on_error {
            opts.insert("on_error".into(), v.clone().into());
        }
        Options::Object(opts)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OpenAIEmbeddingOptions {
    pub encoding_format: Option<String>,
    pub on_error: Option<String>,
}

impl ToDescriptorOptions for OpenAIEmbeddingOptions {
    fn to_descriptor_options(&self) -> Options {
        let mut opts = serde_json::Map::new();
        if let Some(ref v) = self.encoding_format {
            opts.insert("encoding_format".into(), v.clone().into());
        }
        if let Some(ref v) = self.on_error {
            opts.insert("on_error".into(), v.clone().into());
        }
        Options::Object(opts)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AnthropicProviderOptions {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub timeout: Option<f64>,
    pub max_retries: Option<u64>,
    pub concurrency: Option<usize>,
    pub max_api_concurrency: Option<usize>,
}

impl ToDescriptorOptions for AnthropicProviderOptions {
    fn to_descriptor_options(&self) -> Options {
        let mut opts = serde_json::Map::new();
        if let Some(ref v) = self.api_key {
            opts.insert("api_key".into(), v.clone().into());
        }
        if let Some(ref v) = self.base_url {
            opts.insert("base_url".into(), v.clone().into());
        }
        if let Some(v) = self.timeout {
            opts.insert("timeout".into(), v.into());
        }
        if let Some(v) = self.max_retries {
            opts.insert("max_retries".into(), v.into());
        }
        if let Some(v) = self.concurrency {
            opts.insert("actor_number".into(), (v as u64).into());
        }
        if let Some(v) = self.max_api_concurrency {
            opts.insert("max_api_concurrency".into(), (v as u64).into());
        }
        Options::Object(opts)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AnthropicPromptOptions {
    pub max_tokens: Option<u64>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u64>,
    pub stop_sequences: Option<Vec<String>>,
    pub on_error: Option<String>,
}

impl ToDescriptorOptions for AnthropicPromptOptions {
    fn to_descriptor_options(&self) -> Options {
        let mut opts = serde_json::Map::new();
        if let Some(v) = self.max_tokens {
            opts.insert("max_tokens".into(), v.into());
        }
        if let Some(v) = self.temperature {
            opts.insert("temperature".into(), v.into());
        }
        if let Some(v) = self.top_p {
            opts.insert("top_p".into(), v.into());
        }
        if let Some(v) = self.top_k {
            opts.insert("top_k".into(), v.into());
        }
        if let Some(ref v) = self.stop_sequences {
            opts.insert(
                "stop_sequences".into(),
                serde_json::Value::Array(v.iter().map(|s| s.clone().into()).collect()),
            );
        }
        if let Some(ref v) = self.on_error {
            opts.insert("on_error".into(), v.clone().into());
        }
        Options::Object(opts)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GoogleProviderOptions {
    pub api_key: Option<String>,
    pub concurrency: Option<usize>,
    pub max_api_concurrency: Option<usize>,
}

impl ToDescriptorOptions for GoogleProviderOptions {
    fn to_descriptor_options(&self) -> Options {
        let mut opts = serde_json::Map::new();
        if let Some(ref v) = self.api_key {
            opts.insert("api_key".into(), v.clone().into());
        }
        if let Some(v) = self.concurrency {
            opts.insert("actor_number".into(), (v as u64).into());
        }
        if let Some(v) = self.max_api_concurrency {
            opts.insert("max_api_concurrency".into(), (v as u64).into());
        }
        Options::Object(opts)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GooglePromptOptions {
    pub max_output_tokens: Option<u64>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u64>,
    pub on_error: Option<String>,
}

impl ToDescriptorOptions for GooglePromptOptions {
    fn to_descriptor_options(&self) -> Options {
        let mut opts = serde_json::Map::new();
        if let Some(v) = self.max_output_tokens {
            opts.insert("max_output_tokens".into(), v.into());
        }
        if let Some(v) = self.temperature {
            opts.insert("temperature".into(), v.into());
        }
        if let Some(v) = self.top_p {
            opts.insert("top_p".into(), v.into());
        }
        if let Some(v) = self.top_k {
            opts.insert("top_k".into(), v.into());
        }
        if let Some(ref v) = self.on_error {
            opts.insert("on_error".into(), v.clone().into());
        }
        Options::Object(opts)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GoogleEmbeddingOptions {
    pub task_type: Option<String>,
    pub title: Option<String>,
    pub on_error: Option<String>,
}

impl ToDescriptorOptions for GoogleEmbeddingOptions {
    fn to_descriptor_options(&self) -> Options {
        let mut opts = serde_json::Map::new();
        if let Some(ref v) = self.task_type {
            opts.insert("task_type".into(), v.clone().into());
        }
        if let Some(ref v) = self.on_error {
            opts.insert("on_error".into(), v.clone().into());
        }
        if let Some(ref v) = self.title {
            opts.insert("title".into(), v.clone().into());
        }
        Options::Object(opts)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct VllmProviderOptions {
    pub engine_args: Option<serde_json::Value>,
    pub concurrency: Option<usize>,
    pub gpus_per_actor: Option<f64>,
}

impl ToDescriptorOptions for VllmProviderOptions {
    fn to_descriptor_options(&self) -> Options {
        let mut opts = serde_json::Map::new();
        if let Some(ref v) = self.engine_args {
            opts.insert("engine_args".into(), v.clone());
        }
        if let Some(v) = self.concurrency {
            opts.insert("actor_number".into(), (v as u64).into());
        }
        if let Some(v) = self.gpus_per_actor {
            opts.insert("num_gpus".into(), v.into());
        }
        Options::Object(opts)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct VllmPromptOptions {
    pub generate_args: Option<serde_json::Value>,
    pub max_tokens: Option<u64>,
    pub temperature: Option<f64>,
    pub on_error: Option<String>,
}

impl ToDescriptorOptions for VllmPromptOptions {
    fn to_descriptor_options(&self) -> Options {
        let mut opts = serde_json::Map::new();
        if let Some(ref v) = self.generate_args {
            opts.insert("generate_args".into(), v.clone());
        }
        if let Some(v) = self.max_tokens {
            opts.insert("max_tokens".into(), v.into());
        }
        if let Some(v) = self.temperature {
            opts.insert("temperature".into(), v.into());
        }
        if let Some(ref v) = self.on_error {
            opts.insert("on_error".into(), v.clone().into());
        }
        Options::Object(opts)
    }
}
