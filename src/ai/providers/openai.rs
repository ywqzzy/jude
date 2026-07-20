use super::super::protocols::*;
use super::super::provider::Provider;
use super::super::typing::*;
use crate::error::Error;
use async_trait::async_trait;
use std::sync::Arc;

const MODEL_DIMS: &[(&str, usize)] = &[
    ("text-embedding-ada-002", 1536),
    ("text-embedding-3-small", 1536),
    ("text-embedding-3-large", 3072),
];

pub fn factory(name: Option<&str>, options: Options) -> Result<Arc<dyn Provider>, Error> {
    Ok(Arc::new(OpenAIProvider {
        name: name.unwrap_or("openai").to_string(),
        options,
    }))
}

pub struct OpenAIProvider {
    name: String,
    options: Options,
}

impl Provider for OpenAIProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_text_embedder(
        &self,
        model: Option<&str>,
        dimensions: Option<usize>,
        _options: &Options,
    ) -> Result<Arc<dyn TextEmbedderDescriptor>, Error> {
        Ok(Arc::new(OpenAITextEmbedderDescriptor {
            provider_name: self.name.clone(),
            provider_options: self.options.clone(),
            model_name: model.unwrap_or("text-embedding-3-small").to_string(),
            dimensions,
            embed_options: Options::Null,
        }))
    }

    fn get_prompter(
        &self,
        model: Option<&str>,
        system_message: Option<&str>,
        _options: &Options,
    ) -> Result<Arc<dyn PrompterDescriptor>, Error> {
        Ok(Arc::new(OpenAIPrompterDescriptor {
            provider_name: self.name.clone(),
            provider_options: self.options.clone(),
            model_name: model.unwrap_or("gpt-4o-mini").to_string(),
            system_message: system_message.map(String::from),
            prompt_options: Options::Null,
        }))
    }
}

pub struct OpenAITextEmbedderDescriptor {
    pub provider_name: String,
    pub provider_options: Options,
    pub model_name: String,
    pub dimensions: Option<usize>,
    pub embed_options: Options,
}

impl Descriptor for OpenAITextEmbedderDescriptor {
    type Instance = Arc<dyn TextEmbedder>;

    fn get_provider(&self) -> &str {
        &self.provider_name
    }
    fn get_model(&self) -> &str {
        &self.model_name
    }
    fn get_options(&self) -> &Options {
        &self.embed_options
    }

    fn instantiate(&self) -> Result<Self::Instance, Error> {
        let api_key = self
            .provider_options
            .get("api_key")
            .and_then(|v| v.as_str())
            .map(String::from)
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .ok_or_else(|| Error::Other("OpenAI API key not provided".into()))?;

        let base_url = self
            .provider_options
            .get("base_url")
            .and_then(|v| v.as_str())
            .map(String::from)
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());

        Ok(Arc::new(OpenAITextEmbedder {
            client: reqwest::Client::new(),
            model: self.model_name.clone(),
            dimensions: self.dimensions,
            api_key,
            base_url,
        }))
    }
}

impl TextEmbedderDescriptor for OpenAITextEmbedderDescriptor {
    fn get_dimensions(&self) -> Result<EmbeddingDimensions, Error> {
        let size = if let Some(d) = self.dimensions {
            d
        } else {
            MODEL_DIMS
                .iter()
                .find(|(m, _)| *m == self.model_name)
                .map(|(_, d)| *d)
                .unwrap_or(1536)
        };
        Ok(EmbeddingDimensions::default_f32(size))
    }
    fn is_async(&self) -> bool {
        true
    }
}

pub struct OpenAITextEmbedder {
    client: reqwest::Client,
    model: String,
    dimensions: Option<usize>,
    api_key: String,
    base_url: String,
}

#[async_trait]
impl TextEmbedder for OpenAITextEmbedder {
    async fn embed_text(&self, text: Vec<String>) -> Result<Vec<Embedding>, Error> {
        let mut body = serde_json::json!({
            "model": self.model,
            "input": text,
            "encoding_format": "float",
        });
        if let Some(d) = self.dimensions {
            body["dimensions"] = serde_json::json!(d);
        }

        let url = format!("{}/embeddings", self.base_url);
        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Http(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(Error::Http(format!("OpenAI API error {status}: {text}")));
        }

        let resp_json: serde_json::Value =
            resp.json().await.map_err(|e| Error::Http(e.to_string()))?;

        if let Some(usage) = resp_json.get("usage") {
            super::super::metrics::record_token_metrics(
                "embed",
                &self.model,
                "openai",
                usage.get("prompt_tokens").and_then(|v| v.as_u64()),
                None,
                usage.get("total_tokens").and_then(|v| v.as_u64()),
            );
        }

        let data = resp_json
            .get("data")
            .and_then(|v| v.as_array())
            .ok_or_else(|| Error::Other("Missing 'data' in OpenAI response".into()))?;

        let embeddings: Vec<Embedding> = data
            .iter()
            .map(|d| {
                d.get("embedding")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                            .collect()
                    })
                    .unwrap_or_default()
            })
            .collect();

        Ok(embeddings)
    }
}

pub struct OpenAIPrompterDescriptor {
    pub provider_name: String,
    pub provider_options: Options,
    pub model_name: String,
    pub system_message: Option<String>,
    pub prompt_options: Options,
}

impl Descriptor for OpenAIPrompterDescriptor {
    type Instance = Arc<dyn Prompter>;

    fn get_provider(&self) -> &str {
        &self.provider_name
    }
    fn get_model(&self) -> &str {
        &self.model_name
    }
    fn get_options(&self) -> &Options {
        &self.prompt_options
    }

    fn instantiate(&self) -> Result<Self::Instance, Error> {
        let api_key = self
            .provider_options
            .get("api_key")
            .and_then(|v| v.as_str())
            .map(String::from)
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .ok_or_else(|| Error::Other("OpenAI API key not provided".into()))?;

        let base_url = self
            .provider_options
            .get("base_url")
            .and_then(|v| v.as_str())
            .map(String::from)
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());

        let temperature = self
            .prompt_options
            .get("temperature")
            .and_then(|v| v.as_f64());
        let max_tokens = self
            .prompt_options
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .or_else(|| {
                self.prompt_options
                    .get("max_output_tokens")
                    .and_then(|v| v.as_u64())
            });

        Ok(Arc::new(OpenAIPrompter {
            client: reqwest::Client::new(),
            model: self.model_name.clone(),
            system_message: self.system_message.clone(),
            api_key,
            base_url,
            temperature,
            max_tokens,
        }))
    }
}

impl PrompterDescriptor for OpenAIPrompterDescriptor {}

pub struct OpenAIPrompter {
    client: reqwest::Client,
    model: String,
    system_message: Option<String>,
    api_key: String,
    base_url: String,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
}

#[async_trait]
impl Prompter for OpenAIPrompter {
    async fn prompt(&self, messages: PromptMessages) -> Result<Option<String>, Error> {
        let mut messages_arr = Vec::new();

        if let Some(ref sys) = self.system_message {
            messages_arr.push(serde_json::json!({"role": "system", "content": sys}));
        }

        match &messages {
            PromptMessages::Text(text) => {
                messages_arr.push(serde_json::json!({"role": "user", "content": text}));
            }
            PromptMessages::Multimodal(parts) => {
                let mut content_parts = Vec::new();
                for part in parts {
                    match part {
                        MessagePart::Text(t) => {
                            content_parts.push(serde_json::json!({"type": "text", "text": t}));
                        }
                        MessagePart::Image { mime_type, data } => {
                            let b64 = base64_encode(data);
                            let data_url = format!("data:{};base64,{}", mime_type, b64);
                            content_parts.push(serde_json::json!({
                                "type": "image_url",
                                "image_url": {"url": data_url}
                            }));
                        }
                    }
                }
                messages_arr.push(serde_json::json!({"role": "user", "content": content_parts}));
            }
        }

        let mut body = serde_json::json!({
            "model": self.model,
            "messages": messages_arr,
        });
        body["max_tokens"] = serde_json::json!(self.max_tokens.unwrap_or(1024));
        if let Some(t) = self.temperature {
            body["temperature"] = serde_json::json!(t);
        }

        let url = format!("{}/chat/completions", self.base_url);
        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Http(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(Error::Http(format!("OpenAI API error {status}: {text}")));
        }

        let resp_json: serde_json::Value =
            resp.json().await.map_err(|e| Error::Http(e.to_string()))?;

        if let Some(usage) = resp_json.get("usage") {
            super::super::metrics::record_token_metrics(
                "prompt",
                &self.model,
                "openai",
                usage.get("prompt_tokens").and_then(|v| v.as_u64()),
                usage.get("completion_tokens").and_then(|v| v.as_u64()),
                usage.get("total_tokens").and_then(|v| v.as_u64()),
            );
        }

        let content = resp_json
            .get("choices")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .map(String::from);

        Ok(content)
    }
}

fn base64_encode(data: &[u8]) -> String {
    // Simple base64 encoding without external dependency
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as usize;
        let b1 = if chunk.len() > 1 {
            chunk[1] as usize
        } else {
            0
        };
        let b2 = if chunk.len() > 2 {
            chunk[2] as usize
        } else {
            0
        };
        result.push(CHARS[b0 >> 2] as char);
        result.push(CHARS[((b0 & 0x03) << 4) | (b1 >> 4)] as char);
        if chunk.len() > 1 {
            result.push(CHARS[((b1 & 0x0f) << 2) | (b2 >> 6)] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(CHARS[b2 & 0x3f] as char);
        } else {
            result.push('=');
        }
    }
    result
}
