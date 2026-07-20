use duckdb::arrow::datatypes::DataType;
use std::sync::Arc;

pub type Embedding = Vec<f32>;
pub type Label = String;
pub type Options = serde_json::Value;

#[derive(Clone, Debug)]
pub struct EmbeddingDimensions {
    pub size: usize,
    pub dtype: DataType,
}

impl EmbeddingDimensions {
    pub fn as_arrow_type(&self) -> DataType {
        DataType::List(Arc::new(arrow::datatypes::Field::new(
            "item",
            self.dtype.clone(),
            true,
        )))
    }

    pub fn default_f32(size: usize) -> Self {
        Self {
            size,
            dtype: DataType::Float32,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct UdfOptions {
    pub actor_number: Option<usize>,
    pub num_gpus: Option<f64>,
    pub max_retries: usize,
    pub on_error: OnError,
    pub batch_size: Option<usize>,
    pub max_api_concurrency: Option<usize>,
}

impl Default for OnError {
    fn default() -> Self {
        OnError::Raise
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OnError {
    Raise,
    Log,
    Ignore,
}

impl OnError {
    pub fn from_str(s: &str) -> Self {
        match s {
            "log" => OnError::Log,
            "ignore" => OnError::Ignore,
            _ => OnError::Raise,
        }
    }
}

pub trait Descriptor: Send + Sync + 'static {
    type Instance: Send + 'static;

    fn get_provider(&self) -> &str;
    fn get_model(&self) -> &str;
    fn get_options(&self) -> &Options;
    fn instantiate(&self) -> Result<Self::Instance, crate::error::Error>;

    fn get_udf_options(&self) -> UdfOptions {
        let opts = self.get_options();
        UdfOptions {
            actor_number: opts
                .get("actor_number")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            num_gpus: opts.get("num_gpus").and_then(|v| v.as_f64()),
            max_retries: opts
                .get("max_retries")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(3),
            on_error: opts
                .get("on_error")
                .and_then(|v| v.as_str())
                .map(OnError::from_str)
                .unwrap_or_default(),
            batch_size: opts
                .get("batch_size")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            max_api_concurrency: opts
                .get("max_api_concurrency")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
        }
    }
}
