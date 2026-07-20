use super::typing::OnError;
use crate::error::Error;
use std::time::Duration;

#[derive(Debug)]
pub struct RetryAfterError {
    pub retry_after: f64,
    pub message: String,
}

impl std::fmt::Display for RetryAfterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RetryAfterError: retry_after={:.1}s, {}",
            self.retry_after, self.message
        )
    }
}

impl std::error::Error for RetryAfterError {}

pub async fn retry_call_async<F, Fut, T>(
    f: F,
    max_retries: usize,
    on_error: OnError,
) -> Result<Option<T>, Error>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T, Error>>,
{
    let mut last_err: Option<Error> = None;
    for attempt in 0..=max_retries {
        match f().await {
            Ok(result) => return Ok(Some(result)),
            Err(e) => {
                last_err = Some(e);
                if attempt < max_retries {
                    let wait_secs = 2u64.pow(attempt as u32).min(30) as f64;
                    let wait = Duration::from_secs_f64(wait_secs);
                    tokio::time::sleep(wait).await;
                }
            }
        }
    }
    match on_error {
        OnError::Raise => Err(last_err.unwrap_or_else(|| Error::Other("Unknown error".into()))),
        _ => Ok(None),
    }
}

pub fn extract_retry_after(error: &Error) -> Option<f64> {
    match error {
        Error::Http(msg) => {
            if msg.contains("429") || msg.contains("503") {
                Some(5.0)
            } else {
                None
            }
        }
        _ => None,
    }
}
