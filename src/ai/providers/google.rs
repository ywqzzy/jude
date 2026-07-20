use super::super::protocols::*;
use super::super::provider::Provider;
use super::super::typing::*;
use crate::error::Error;
use async_trait::async_trait;
use std::sync::Arc;

const EMBEDDING_DIMS: &[(&str, usize)] = &[("text-embedding-004", 768), ("embedding-001", 768)];

pub fn factory(name: Option<&str>, options: Options) -> Result<Arc<dyn Provider>, Error> {
    Ok(Arc::new(GoogleProvider {
        name: name.unwrap_or("google").to_string(),
        options,
    }))
}

pub struct GoogleProvider {
    name: String,
    options: Options,
}

impl Provider for GoogleProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_text_embedder(
        &self,
        model: Option<&str>,
        dimensions: Option<usize>,
        _options: &Options,
    ) -> Result<Arc<dyn TextEmbedderDescriptor>, Error> {
        Ok(Arc::new(GoogleTextEmbedderDescriptor {
            provider_name: self.name.clone(),
            provider_options: self.options.clone(),
            model_name: model.unwrap_or("text-embedding-004").to_string(),
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
        Ok(Arc::new(GooglePrompterDescriptor {
            provider_name: self.name.clone(),
            provider_options: self.options.clone(),
            model_name: model.unwrap_or("gemini-2.0-flash").to_string(),
            system_message: system_message.map(String::from),
            prompt_options: Options::Null,
        }))
    }
}

pub struct GoogleTextEmbedderDescriptor {
    pub provider_name: String,
    pub provider_options: Options,
    pub model_name: String,
    pub dimensions: Option<usize>,
    pub embed_options: Options,
}

impl Descriptor for GoogleTextEmbedderDescriptor {
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
            .or_else(|| std::env::var("GOOGLE_API_KEY").ok())
            .ok_or_else(|| Error::Other("Google API key not provided".into()))?;

        Ok(Arc::new(GoogleTextEmbedder {
            client: reqwest::Client::new(),
            model: self.model_name.clone(),
            api_key,
        }))
    }
}

impl TextEmbedderDescriptor for GoogleTextEmbedderDescriptor {
    fn get_dimensions(&self) -> Result<EmbeddingDimensions, Error> {
        let size = if let Some(d) = self.dimensions {
            d
        } else {
            EMBEDDING_DIMS
                .iter()
                .find(|(m, _)| *m == self.model_name)
                .map(|(_, d)| *d)
                .unwrap_or(768)
        };
        Ok(EmbeddingDimensions::default_f32(size))
    }
    fn is_async(&self) -> bool {
        true
    }
}

pub struct GoogleTextEmbedder {
    client: reqwest::Client,
    model: String,
    api_key: String,
}

#[async_trait]
impl TextEmbedder for GoogleTextEmbedder {
    async fn embed_text(&self, text: Vec<String>) -> Result<Vec<Embedding>, Error> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:batchEmbedContents?key={}",
            self.model, self.api_key
        );

        let requests: Vec<_> = text
            .iter()
            .map(|t| {
                serde_json::json!({
                    "model": format!("models/{}", self.model),
                    "content": {"parts": [{"text": t}]}
                })
            })
            .collect();

        let body = serde_json::json!({"requests": requests});

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Http(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(Error::Http(format!("Google API error {status}: {text}")));
        }

        let resp_json: serde_json::Value =
            resp.json().await.map_err(|e| Error::Http(e.to_string()))?;

        let embeddings = resp_json
            .get("embeddings")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|e| {
                        e.get("values")
                            .and_then(|v| v.as_array())
                            .map(|vals| {
                                vals.iter()
                                    .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                                    .collect()
                            })
                            .unwrap_or_default()
                    })
                    .collect()
            })
            .unwrap_or_default();

        super::super::metrics::record_token_metrics(
            "embed",
            &self.model,
            "google",
            None,
            None,
            None,
        );

        Ok(embeddings)
    }
}

pub struct GooglePrompterDescriptor {
    pub provider_name: String,
    pub provider_options: Options,
    pub model_name: String,
    pub system_message: Option<String>,
    pub prompt_options: Options,
}

impl Descriptor for GooglePrompterDescriptor {
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
            .or_else(|| std::env::var("GOOGLE_API_KEY").ok())
            .ok_or_else(|| Error::Other("Google API key not provided".into()))?;

        Ok(Arc::new(GooglePrompter {
            client: reqwest::Client::new(),
            model: self.model_name.clone(),
            system_message: self.system_message.clone(),
            api_key,
        }))
    }
}

impl PrompterDescriptor for GooglePrompterDescriptor {}

pub struct GooglePrompter {
    client: reqwest::Client,
    model: String,
    system_message: Option<String>,
    api_key: String,
}

#[async_trait]
impl Prompter for GooglePrompter {
    async fn prompt(&self, messages: PromptMessages) -> Result<Option<String>, Error> {
        let user_text = match messages {
            PromptMessages::Text(t) => t,
            PromptMessages::Multimodal(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    MessagePart::Text(t) => Some(t.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(" "),
        };

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            self.model, self.api_key
        );

        let mut body = serde_json::json!({
            "contents": [{"parts": [{"text": user_text}]}],
        });
        if let Some(ref sys) = self.system_message {
            body["systemInstruction"] = serde_json::json!({"parts": [{"text": sys}]});
        }

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Http(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(Error::Http(format!("Google API error {status}: {text}")));
        }

        let resp_json: serde_json::Value =
            resp.json().await.map_err(|e| Error::Http(e.to_string()))?;

        if let Some(usage) = resp_json.get("usageMetadata") {
            super::super::metrics::record_token_metrics(
                "prompt",
                &self.model,
                "google",
                usage.get("promptTokenCount").and_then(|v| v.as_u64()),
                usage.get("candidatesTokenCount").and_then(|v| v.as_u64()),
                usage.get("totalTokenCount").and_then(|v| v.as_u64()),
            );
        }

        let content = resp_json
            .get("candidates")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|c| c.get("content"))
            .and_then(|content| content.get("parts"))
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|p| p.get("text"))
            .and_then(|t| t.as_str())
            .map(String::from);

        Ok(content)
    }
}
