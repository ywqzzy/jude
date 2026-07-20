//! Subprocess pool for out-of-process UDF execution.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock};

use duckdb::arrow::datatypes::SchemaRef;
use duckdb::arrow::ipc::reader::StreamReader;
use duckdb::arrow::ipc::writer::StreamWriter;
use duckdb::arrow::record_batch::RecordBatch;

use crate::error::Error;

const CTRL: u32 = 0xFFFF_FFFF;

/// A single worker subprocess speaking the Arrow-IPC framing protocol.
struct Worker {
    child: Child,
}

impl Worker {
    fn spawn(python: &str, init_ctrl: &[u8]) -> Result<Self, Error> {
        let mut child = Command::new(python)
            .arg("-m")
            .arg("jude.execution._worker")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| Error::Udf(format!("failed to spawn UDF worker: {e}")))?;

        // Send the init control frame (pickled UDF + config).
        {
            let stdin = child.stdin.as_mut().unwrap();
            write_ctrl(stdin, init_ctrl)?;
        }
        // Wait for the "ready" control frame.
        {
            let stdout = child.stdout.as_mut().unwrap();
            let (kind, payload) = read_frame(stdout)?;
            if kind != CTRL {
                return Err(Error::Udf("worker did not send init ack".into()));
            }
            let msg: serde_json::Value = serde_json::from_slice(&payload)
                .map_err(|e| Error::Udf(format!("bad init ack: {e}")))?;
            if msg.get("status").and_then(|s| s.as_str()) != Some("ready") {
                return Err(Error::Udf(format!("worker init failed: {msg}")));
            }
        }
        Ok(Self { child })
    }

    /// Run one batch through the worker and return the result batches.
    fn run_batch(&mut self, batch: &RecordBatch) -> Result<Vec<RecordBatch>, Error> {
        let ipc = batch_to_ipc(batch)?;
        {
            let stdin = self.child.stdin.as_mut().unwrap();
            write_frame(stdin, &ipc)?;
        }
        let stdout = self.child.stdout.as_mut().unwrap();
        let (kind, payload) = read_frame(stdout)?;
        if kind == CTRL {
            let msg: serde_json::Value = serde_json::from_slice(&payload)
                .map_err(|e| Error::Udf(format!("bad control frame: {e}")))?;
            let detail = msg
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown UDF worker error");
            return Err(Error::Udf(detail.to_string()));
        }
        ipc_to_batches(&payload)
    }

    fn shutdown(&mut self) {
        if let Some(stdin) = self.child.stdin.as_mut() {
            let _ = write_ctrl(stdin, br#"{"cmd":"shutdown"}"#);
        }
        let _ = self.child.wait();
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// A pool of worker subprocesses. Batches are distributed round-robin.
pub struct SubprocessPool {
    workers: Vec<Mutex<Worker>>,
}

impl SubprocessPool {
    /// Spawn `num_workers` workers, each initialized with `init_ctrl` (the JSON
    /// control payload carrying the pickled UDF + call_mode).
    pub fn new(python: &str, num_workers: usize, init_ctrl: &[u8]) -> Result<Self, Error> {
        let n = num_workers.max(1);
        let mut workers = Vec::with_capacity(n);
        for _ in 0..n {
            workers.push(Mutex::new(Worker::spawn(python, init_ctrl)?));
        }
        Ok(Self { workers })
    }

    /// Map `batches` through the pool in parallel and return the concatenated
    /// output batches, preserving input order.
    ///
    /// This is the GIL-free hot path: the caller releases the GIL (via
    /// `Python::allow_threads`) so all worker pipes proceed concurrently and
    /// other Python threads keep running.
    pub fn map_batches(&self, batches: &[RecordBatch]) -> Result<Vec<RecordBatch>, Error> {
        use std::thread;

        if batches.is_empty() {
            return Ok(Vec::new());
        }
        let n = self.workers.len();
        // Assign each input batch index to a worker (round-robin) and run each
        // worker's slice on its own OS thread so pipes overlap.
        let mut results: Vec<Option<Vec<RecordBatch>>> = (0..batches.len()).map(|_| None).collect();

        thread::scope(|scope| -> Result<(), Error> {
            let mut handles = Vec::new();
            for (w, worker) in self.workers.iter().enumerate() {
                let assigned: Vec<usize> = (0..batches.len()).filter(|i| i % n == w).collect();
                if assigned.is_empty() {
                    continue;
                }
                let batches_ref = batches;
                handles.push(scope.spawn(
                    move || -> Result<Vec<(usize, Vec<RecordBatch>)>, Error> {
                        let mut w = worker.lock().unwrap();
                        let mut out = Vec::with_capacity(assigned.len());
                        for i in assigned {
                            out.push((i, w.run_batch(&batches_ref[i])?));
                        }
                        Ok(out)
                    },
                ));
            }
            for h in handles {
                let part = h
                    .join()
                    .map_err(|_| Error::Udf("UDF worker thread panicked".into()))??;
                for (i, batches) in part {
                    results[i] = Some(batches);
                }
            }
            Ok(())
        })?;

        let mut out = Vec::new();
        for r in results.into_iter().flatten() {
            out.extend(r);
        }
        Ok(out)
    }
}

// ---- framing ----

fn write_frame<W: Write>(w: &mut W, payload: &[u8]) -> Result<(), Error> {
    w.write_all(&(payload.len() as u32).to_le_bytes())
        .map_err(Error::Io)?;
    w.write_all(payload).map_err(Error::Io)?;
    w.flush().map_err(Error::Io)?;
    Ok(())
}

fn write_ctrl<W: Write>(w: &mut W, json: &[u8]) -> Result<(), Error> {
    w.write_all(&CTRL.to_le_bytes()).map_err(Error::Io)?;
    w.write_all(&(json.len() as u32).to_le_bytes())
        .map_err(Error::Io)?;
    w.write_all(json).map_err(Error::Io)?;
    w.flush().map_err(Error::Io)?;
    Ok(())
}

fn read_exact<R: Read>(r: &mut R, n: usize) -> Result<Vec<u8>, Error> {
    let mut buf = vec![0u8; n];
    r.read_exact(&mut buf).map_err(Error::Io)?;
    Ok(buf)
}

fn read_frame<R: Read>(r: &mut R) -> Result<(u32, Vec<u8>), Error> {
    let header = read_exact(r, 4)?;
    let len = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
    if len == CTRL {
        let clen_buf = read_exact(r, 4)?;
        let clen =
            u32::from_le_bytes([clen_buf[0], clen_buf[1], clen_buf[2], clen_buf[3]]) as usize;
        return Ok((CTRL, read_exact(r, clen)?));
    }
    Ok((len, read_exact(r, len as usize)?))
}

// ---- Arrow IPC helpers ----

fn batch_to_ipc(batch: &RecordBatch) -> Result<Vec<u8>, Error> {
    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, &batch.schema()).map_err(Error::Arrow)?;
        writer.write(batch).map_err(Error::Arrow)?;
        writer.finish().map_err(Error::Arrow)?;
    }
    Ok(buf)
}

fn ipc_to_batches(data: &[u8]) -> Result<Vec<RecordBatch>, Error> {
    let reader = StreamReader::try_new(std::io::Cursor::new(data), None).map_err(Error::Arrow)?;
    let mut out = Vec::new();
    for b in reader {
        out.push(b.map_err(Error::Arrow)?);
    }
    Ok(out)
}

/// Schema of the first result batch, if any (helper for callers).
pub fn first_schema(batches: &[RecordBatch]) -> Option<SchemaRef> {
    batches.first().map(|b| b.schema())
}

// ---- Persistent pool registry ----
//
// Spawning worker interpreters is expensive (~100ms each for pyarrow import), so
// we cache pools by (python, worker-count, init-payload) and reuse them across
// map_batches calls. This amortizes spawn cost to once per distinct UDF, the
// same idea as Vane's actor pools.

type PoolKey = (String, usize, u64);

fn pool_registry() -> &'static Mutex<HashMap<PoolKey, Arc<SubprocessPool>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<PoolKey, Arc<SubprocessPool>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn hash_bytes(data: &[u8]) -> u64 {
    // FNV-1a — stable, no external dep.
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Get or create a cached pool for this (python, workers, init payload).
pub fn get_or_create_pool(
    python: &str,
    num_workers: usize,
    init_ctrl: &[u8],
) -> Result<Arc<SubprocessPool>, Error> {
    let key: PoolKey = (
        python.to_string(),
        num_workers.max(1),
        hash_bytes(init_ctrl),
    );
    let mut reg = pool_registry().lock().unwrap();
    if let Some(pool) = reg.get(&key) {
        return Ok(pool.clone());
    }
    let pool = Arc::new(SubprocessPool::new(python, num_workers, init_ctrl)?);
    reg.insert(key, pool.clone());
    Ok(pool)
}

/// Tear down all cached pools (e.g. at interpreter shutdown / tests).
pub fn shutdown_all_pools() {
    if let Some(reg) = pool_registry().lock().ok().as_mut() {
        reg.clear();
    }
}
