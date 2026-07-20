use super::protocols::*;
use super::typing::*;
use crate::error::Error;
use duckdb::arrow::array::*;
use duckdb::arrow::datatypes::{DataType, Field, Schema};
use duckdb::arrow::record_batch::RecordBatch;
use std::sync::Arc;
use std::sync::OnceLock;

static ASYNC_RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

pub fn async_rt() -> &'static tokio::runtime::Runtime {
    ASYNC_RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
    })
}

fn extract_string_column(batch: &RecordBatch, column: &str) -> Result<Vec<Option<String>>, Error> {
    let schema = batch.schema();
    let idx = schema
        .index_of(column)
        .map_err(|e| Error::Other(format!("Column '{column}' not found: {e}")))?;
    let array = batch.column(idx as usize);
    let str_array = array
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| Error::Other(format!("Column '{column}' is not a string column")))?;
    let mut values = Vec::with_capacity(str_array.len());
    for i in 0..str_array.len() {
        if str_array.is_null(i) {
            values.push(None);
        } else {
            values.push(Some(str_array.value(i).to_string()));
        }
    }
    Ok(values)
}

pub struct EmbedTextBatch {
    pub descriptor: Arc<dyn TextEmbedderDescriptor>,
    pub column: String,
    pub output_column: String,
    pub max_chunk_chars: Option<usize>,
    pub chunk_overlap_chars: usize,
    pub max_retries: usize,
    pub on_error: OnError,
    pub normalize: bool,
    embedder: Option<Arc<dyn TextEmbedder>>,
}

impl EmbedTextBatch {
    pub fn new(
        descriptor: Arc<dyn TextEmbedderDescriptor>,
        column: String,
        output_column: String,
        max_retries: usize,
        on_error: OnError,
    ) -> Self {
        Self {
            descriptor,
            column,
            output_column,
            max_chunk_chars: None,
            chunk_overlap_chars: 200,
            max_retries,
            on_error,
            normalize: false,
            embedder: None,
        }
    }

    fn ensure_embedder(&mut self) -> Result<&Arc<dyn TextEmbedder>, Error> {
        if self.embedder.is_none() {
            self.embedder = Some(self.descriptor.instantiate()?);
        }
        Ok(self.embedder.as_ref().unwrap())
    }

    pub fn process_batch(&mut self, batch: &RecordBatch) -> Result<RecordBatch, Error> {
        let texts = extract_string_column(batch, &self.column)?;
        let texts: Vec<String> = texts.into_iter().map(|t| t.unwrap_or_default()).collect();

        let embedder = self.ensure_embedder()?.clone();
        let rt = async_rt();
        let embeddings = rt.block_on(async {
            super::retry::retry_call_async(
                || {
                    let embedder = embedder.clone();
                    let texts = texts.clone();
                    async move { embedder.embed_text(texts).await }
                },
                self.max_retries,
                self.on_error,
            )
            .await
        })?;

        let embeddings = embeddings.unwrap_or_default();
        let dims = self.descriptor.get_dimensions()?;
        let embeddings_array = build_float_list_array(&embeddings, dims.size);

        let schema = Arc::new(Schema::new(vec![Field::new(
            &self.output_column,
            dims.as_arrow_type(),
            true,
        )]));
        Ok(RecordBatch::try_new(
            schema,
            vec![Arc::new(embeddings_array)],
        )?)
    }
}

pub struct ClassifyTextBatch {
    pub descriptor: Arc<dyn TextClassifierDescriptor>,
    pub column: String,
    pub output_column: String,
    pub labels: Vec<Label>,
    pub max_retries: usize,
    pub on_error: OnError,
    classifier: Option<Arc<dyn TextClassifier>>,
}

impl ClassifyTextBatch {
    pub fn new(
        descriptor: Arc<dyn TextClassifierDescriptor>,
        column: String,
        output_column: String,
        labels: Vec<Label>,
        max_retries: usize,
        on_error: OnError,
    ) -> Self {
        Self {
            descriptor,
            column,
            output_column,
            labels,
            max_retries,
            on_error,
            classifier: None,
        }
    }

    fn ensure_classifier(&mut self) -> Result<&Arc<dyn TextClassifier>, Error> {
        if self.classifier.is_none() {
            self.classifier = Some(self.descriptor.instantiate()?);
        }
        Ok(self.classifier.as_ref().unwrap())
    }

    pub fn process_batch(&mut self, batch: &RecordBatch) -> Result<RecordBatch, Error> {
        let texts = extract_string_column(batch, &self.column)?;
        let texts: Vec<String> = texts.into_iter().map(|t| t.unwrap_or_default()).collect();

        let classifier = self.ensure_classifier()?.clone();
        let labels = self.labels.clone();
        let rt = async_rt();
        let results = rt.block_on(async {
            super::retry::retry_call_async(
                || {
                    let classifier = classifier.clone();
                    let texts = texts.clone();
                    let labels = labels.clone();
                    async move { classifier.classify_text(texts, labels).await }
                },
                self.max_retries,
                self.on_error,
            )
            .await
        })?;

        let results = results.unwrap_or_default();
        let mut builder = StringBuilder::new();
        for result in &results {
            match result {
                Some(label) => builder.append_value(label),
                None => builder.append_null(),
            }
        }
        let array = builder.finish();

        let schema = Arc::new(Schema::new(vec![Field::new(
            &self.output_column,
            DataType::Utf8,
            true,
        )]));
        Ok(RecordBatch::try_new(schema, vec![Arc::new(array)])?)
    }
}

pub struct PromptBatch {
    pub descriptor: Arc<dyn PrompterDescriptor>,
    pub column: String,
    pub output_column: String,
    pub max_retries: usize,
    pub on_error: OnError,
    pub max_api_concurrency: Option<usize>,
    pub system_message: Option<String>,
    prompter: Option<Arc<dyn Prompter>>,
}

impl PromptBatch {
    pub fn new(
        descriptor: Arc<dyn PrompterDescriptor>,
        column: String,
        output_column: String,
        max_retries: usize,
        on_error: OnError,
        system_message: Option<String>,
    ) -> Self {
        Self {
            descriptor,
            column,
            output_column,
            max_retries,
            on_error,
            max_api_concurrency: Some(16),
            system_message,
            prompter: None,
        }
    }

    fn ensure_prompter(&mut self) -> Result<&Arc<dyn Prompter>, Error> {
        if self.prompter.is_none() {
            self.prompter = Some(self.descriptor.instantiate()?);
        }
        Ok(self.prompter.as_ref().unwrap())
    }

    pub fn process_batch(&mut self, batch: &RecordBatch) -> Result<RecordBatch, Error> {
        let texts = extract_string_column(batch, &self.column)?;
        let prompter = self.ensure_prompter()?.clone();
        let max_concurrency = self.max_api_concurrency.unwrap_or(16);
        let system_msg = self.system_message.clone();
        let on_error = self.on_error;
        let max_retries = self.max_retries;

        let rt = async_rt();
        let results = rt.block_on(async {
            let sem = Arc::new(tokio::sync::Semaphore::new(max_concurrency));
            let futures: Vec<_> = texts
                .iter()
                .map(|t| {
                    let sem = sem.clone();
                    let prompter = prompter.clone();
                    let text = t.clone().unwrap_or_default();
                    let system_msg = system_msg.clone();
                    let on_error = on_error;
                    let max_retries = max_retries;
                    async move {
                        let _permit = sem.acquire().await.unwrap();
                        let messages = if let Some(sys) = system_msg {
                            PromptMessages::Multimodal(vec![MessagePart::Text(format!(
                                "System: {sys}\n\nUser: {text}"
                            ))])
                        } else {
                            PromptMessages::Text(text)
                        };
                        super::retry::retry_call_async(
                            || {
                                let prompter = prompter.clone();
                                let messages = messages.clone();
                                async move { prompter.prompt(messages).await }
                            },
                            max_retries,
                            on_error,
                        )
                        .await
                    }
                })
                .collect();
            futures::future::join_all(futures).await
        });

        let mut builder = StringBuilder::new();
        for result in &results {
            match result {
                Ok(Some(Some(text))) => builder.append_value(text),
                _ => builder.append_null(),
            }
        }
        let array = builder.finish();

        let schema = Arc::new(Schema::new(vec![Field::new(
            &self.output_column,
            DataType::Utf8,
            true,
        )]));
        Ok(RecordBatch::try_new(schema, vec![Arc::new(array)])?)
    }
}

fn build_float_list_array(embeddings: &[Embedding], expected_size: usize) -> ListArray {
    let mut builder = ListBuilder::new(Float32Builder::with_capacity(
        embeddings.len() * expected_size.max(1),
    ));
    for embedding in embeddings {
        let inner = builder.values();
        for &val in embedding {
            inner.append_value(val);
        }
        builder.append(true);
    }
    builder.finish()
}
