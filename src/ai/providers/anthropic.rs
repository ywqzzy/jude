use super::super::protocols::*;
use super::super::provider::Provider;
use super::super::typing::*;
use crate::error::Error;
use async_trait::async_trait;
use std::sync::Arc;

pub fn factory(name: Option<&str>, options: Options) -> Result<Arc<dyn Provider>, Error> {
    Ok(Arc::new(AnthropicProvider {
        name: name.unwrap_or("anthropic").to_string(),
        options,
    }))
}

pub struct AnthropicProvider {
    name: String,
    options: Options,
}

impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn get_prompter(
        &self,
        model: Option<&str>,
        system_message: Option<&str>,
        _options: &Options,
    ) -> Result<Arc<dyn PrompterDescriptor>, Error> {
        Ok(Arc::new(AnthropicPrompterDescriptor {
            provider_name: self.name.clone(),
            provider_options: self.options.clone(),
            model_name: model.unwrap_or("claude-sonnet-4-20250514").to_string(),
            system_message: system_message.map(String::from),
            prompt_options: Options::Null,
        }))
    }

    fn get_text_embedder(
        &self,
        _model: Option<&str>,
        _dimensions: Option<usize>,
        _options: &Options,
    ) -> Result<Arc<dyn TextEmbedderDescriptor>, Error> {
        Err(Error::Other(
            "Anthropic does not support text embedding".into(),
        ))
    }
}

pub struct AnthropicPrompterDescriptor {
    pub provider_name: String,
    pub provider_options: Options,
    pub model_name: String,
    pub system_message: Option<String>,
    pub prompt_options: Options,
}

impl Descriptor for AnthropicPrompterDescriptor {
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
            .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
            .ok_or_else(|| Error::Other("Anthropic API key not provided".into()))?;

        let base_url = self
            .provider_options
            .get("base_url")
            .and_then(|v| v.as_str())
            .map(String::from)
            .unwrap_or_else(|| "https://api.anthropic.com/v1".to_string());

        Ok(Arc::new(AnthropicPrompter {
            client: reqwest::Client::new(),
            model: self.model_name.clone(),
            system_message: self.system_message.clone(),
            api_key,
            base_url,
        }))
    }
}

impl PrompterDescriptor for AnthropicPrompterDescriptor {}

pub struct AnthropicPrompter {
    client: reqwest::Client,
    model: String,
    system_message: Option<String>,
    api_key: String,
    base_url: String,
}

#[async_trait]
impl Prompter for AnthropicPrompter {
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

        let mut body = serde_json::json!({
            "model": self.model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": user_text}],
        });
        if let Some(ref sys) = self.system_message {
            body["system"] = serde_json::json!(sys);
        }

        let url = format!("{}/messages", self.base_url);
        let resp = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Http(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(Error::Http(format!("Anthropic API error {status}: {text}")));
        }

        let resp_json: serde_json::Value =
            resp.json().await.map_err(|e| Error::Http(e.to_string()))?;

        if let Some(usage) = resp_json.get("usage") {
            super::super::metrics::record_token_metrics(
                "prompt",
                &self.model,
                "anthropic",
                usage.get("input_tokens").and_then(|v| v.as_u64()),
                usage.get("output_tokens").and_then(|v| v.as_u64()),
                None,
            );
        }

        let content = resp_json
            .get("content")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|c| c.get("text"))
            .and_then(|t| t.as_str())
            .map(String::from);

        Ok(content)
    }
}
