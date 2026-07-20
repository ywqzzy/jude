//! PyO3 bindings for the data-curation kernels (`crate::curate`). Batch-oriented:
//! functions take Python lists of strings and return lists, so they slot into
//! cosmos pipeline `Stage`s and `Relation` UDFs over Arrow string columns.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::curate;
use crate::curate_mm;
use crate::kmeans;

/// Chunk one string into character chunks with overlap.
#[pyfunction]
#[pyo3(signature = (text, chunk_chars=1024, overlap=0))]
fn chunk_chars(text: &str, chunk_chars: usize, overlap: usize) -> Vec<String> {
    curate::chunk_chars(text, chunk_chars, overlap)
}

/// Recursive-separator chunking of one string.
#[pyfunction]
#[pyo3(signature = (text, chunk_chars=1024, overlap=0, separators=None))]
fn chunk_recursive(
    text: &str,
    chunk_chars: usize,
    overlap: usize,
    separators: Option<Vec<String>>,
) -> Vec<String> {
    let seps = separators.unwrap_or_else(|| {
        vec![
            "\n\n".to_string(),
            "\n".to_string(),
            ". ".to_string(),
            " ".to_string(),
        ]
    });
    curate::chunk_recursive(text, chunk_chars, overlap, &seps)
}

/// Normalize text (lowercase + collapse whitespace) for dedup keying.
#[pyfunction]
fn normalize_text(text: &str) -> String {
    curate::normalize_text(text)
}

/// SHA-256 hex content hash (optionally after normalization) for exact dedup.
#[pyfunction]
#[pyo3(signature = (text, normalize=true))]
fn content_hash(text: &str, normalize: bool) -> String {
    curate::content_hash(text, normalize)
}

/// Vectorized content-hash over a column of strings (None -> None). This is the
/// hot path for exact-dedup: compute a hash column, then DISTINCT on it.
#[pyfunction]
#[pyo3(signature = (texts, normalize=true))]
fn content_hash_batch(texts: Vec<Option<String>>, normalize: bool) -> Vec<Option<String>> {
    texts
        .into_iter()
        .map(|t| t.map(|s| curate::content_hash(&s, normalize)))
        .collect()
}

fn signals_to_dict<'py>(
    py: Python<'py>,
    s: &curate::QualitySignals,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("char_count", s.char_count)?;
    d.set_item("word_count", s.word_count)?;
    d.set_item("mean_word_len", s.mean_word_len)?;
    d.set_item("alpha_ratio", s.alpha_ratio)?;
    d.set_item("digit_ratio", s.digit_ratio)?;
    d.set_item("symbol_ratio", s.symbol_ratio)?;
    d.set_item("alpha_word_ratio", s.alpha_word_ratio)?;
    d.set_item("dup_line_ratio", s.dup_line_ratio)?;
    d.set_item("hash_line_ratio", s.hash_line_ratio)?;
    d.set_item("top_word_ratio", s.top_word_ratio)?;
    Ok(d)
}

/// Quality signals for one document as a dict.
#[pyfunction]
fn quality_signals<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyDict>> {
    signals_to_dict(py, &curate::quality_signals(text))
}

fn thresholds_from_kwargs(
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<curate::QualityThresholds> {
    let mut t = curate::QualityThresholds::default();
    if let Some(kw) = kwargs {
        if let Some(v) = kw.get_item("min_words")? {
            t.min_words = v.extract()?;
        }
        if let Some(v) = kw.get_item("max_words")? {
            t.max_words = v.extract()?;
        }
        if let Some(v) = kw.get_item("min_mean_word_len")? {
            t.min_mean_word_len = v.extract()?;
        }
        if let Some(v) = kw.get_item("max_mean_word_len")? {
            t.max_mean_word_len = v.extract()?;
        }
        if let Some(v) = kw.get_item("max_symbol_ratio")? {
            t.max_symbol_ratio = v.extract()?;
        }
        if let Some(v) = kw.get_item("min_alpha_word_ratio")? {
            t.min_alpha_word_ratio = v.extract()?;
        }
        if let Some(v) = kw.get_item("max_dup_line_ratio")? {
            t.max_dup_line_ratio = v.extract()?;
        }
        if let Some(v) = kw.get_item("max_top_word_ratio")? {
            t.max_top_word_ratio = v.extract()?;
        }
    }
    Ok(t)
}

/// Quality reject reason for one document (None if it passes). Thresholds via
/// kwargs override the Gopher-ish defaults.
#[pyfunction]
#[pyo3(signature = (text, **kwargs))]
fn quality_reject_reason(
    text: &str,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Option<String>> {
    let t = thresholds_from_kwargs(kwargs)?;
    let sig = curate::quality_signals(text);
    Ok(curate::quality_reject_reason(&sig, &t))
}

/// Vectorized quality gate over a column of strings: returns a list of
/// (keep: bool, reason: Option<str>) — the hot path for the quality-filter
/// stage. None text -> (False, "null").
#[pyfunction]
#[pyo3(signature = (texts, **kwargs))]
fn quality_gate_batch<'py>(
    py: Python<'py>,
    texts: Vec<Option<String>>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Bound<'py, PyList>> {
    let t = thresholds_from_kwargs(kwargs)?;
    let out = PyList::empty(py);
    for text in texts {
        let tup = match text {
            None => (false, Some("null".to_string())),
            Some(s) => {
                let sig = curate::quality_signals(&s);
                match curate::quality_reject_reason(&sig, &t) {
                    None => (true, None),
                    Some(r) => (false, Some(r)),
                }
            }
        };
        out.append(tup)?;
    }
    Ok(out)
}

// ---- C1. MinHash / LSH bindings ----

/// MinHash signature (length num_hashes) of one text over word n-gram shingles.
#[pyfunction]
#[pyo3(signature = (text, num_hashes=128, ngram=2, seed=1))]
fn minhash_signature(text: &str, num_hashes: usize, ngram: usize, seed: u64) -> Vec<u64> {
    curate::minhash_signature(text, num_hashes, ngram, seed)
}

/// Vectorized MinHash over a column of texts (None -> empty signature).
#[pyfunction]
#[pyo3(signature = (texts, num_hashes=128, ngram=2, seed=1))]
fn minhash_signature_batch(
    texts: Vec<Option<String>>,
    num_hashes: usize,
    ngram: usize,
    seed: u64,
) -> Vec<Vec<u64>> {
    texts
        .into_iter()
        .map(|t| {
            t.map(|s| curate::minhash_signature(&s, num_hashes, ngram, seed))
                .unwrap_or_default()
        })
        .collect()
}

/// LSH band keys for one signature (near-dups share >=1 key).
#[pyfunction]
#[pyo3(signature = (signature, bands=16))]
fn lsh_band_keys(signature: Vec<u64>, bands: usize) -> Vec<String> {
    curate::lsh_band_keys(&signature, bands)
}

/// Vectorized LSH band keys over many signatures.
#[pyfunction]
#[pyo3(signature = (signatures, bands=16))]
fn lsh_band_keys_batch(signatures: Vec<Vec<u64>>, bands: usize) -> Vec<Vec<String>> {
    signatures
        .into_iter()
        .map(|s| curate::lsh_band_keys(&s, bands))
        .collect()
}

/// Estimated Jaccard similarity of two signatures (fraction of equal entries).
#[pyfunction]
fn signature_similarity(a: Vec<u64>, b: Vec<u64>) -> f64 {
    curate::signature_similarity(&a, &b)
}

/// Given `n` items and candidate duplicate `pairs` (list of (i, j)), return the
/// component representative (min id) for each item — items with the same rep are
/// duplicates; keep the rep, drop the rest. The reducer of a distributed
/// fuzzy-dedup after LSH bucketing produces the pairs.
#[pyfunction]
fn connected_components(n: usize, pairs: Vec<(usize, usize)>) -> Vec<usize> {
    let mut uf = curate::UnionFind::new(n);
    for (a, b) in pairs {
        if a < n && b < n {
            uf.union(a, b);
        }
    }
    uf.representatives(n)
}

/// Cosine similarity of two float vectors.
#[pyfunction]
fn cosine_similarity(a: Vec<f32>, b: Vec<f32>) -> f32 {
    curate::cosine_similarity(&a, &b)
}

/// Semantic-dedup clustering: given embedding vectors, union any pair with
/// cosine >= threshold and return each vector's cluster representative (min id).
/// O(n^2) — call within an ANN/LSH candidate bucket so n stays small.
#[pyfunction]
#[pyo3(signature = (embeddings, threshold=0.9))]
fn semantic_clusters(embeddings: Vec<Vec<f32>>, threshold: f32) -> Vec<usize> {
    curate::semantic_clusters(&embeddings, threshold)
}

/// Detect (language_code, confidence) for one text (heuristic, no model).
#[pyfunction]
fn detect_language(text: &str) -> (String, f64) {
    curate::detect_language(text)
}

/// Vectorized language detection over a column of texts.
#[pyfunction]
fn detect_language_batch(texts: Vec<Option<String>>) -> Vec<(String, f64)> {
    texts
        .into_iter()
        .map(|t| {
            t.map(|s| curate::detect_language(&s))
                .unwrap_or(("und".to_string(), 0.0))
        })
        .collect()
}

// ---- C10. PII ----

/// Redact PII in one text; returns (redacted, count).
#[pyfunction]
fn redact_pii(text: &str) -> (String, usize) {
    curate::redact_pii(text)
}

/// Vectorized PII redaction over a column (None -> None).
#[pyfunction]
fn redact_pii_batch(texts: Vec<Option<String>>) -> Vec<Option<String>> {
    texts
        .into_iter()
        .map(|t| t.map(|s| curate::redact_pii(&s).0))
        .collect()
}

/// Detected PII spans for one text: list of (kind, start, end).
#[pyfunction]
fn detect_pii(text: &str) -> Vec<(String, usize, usize)> {
    curate::detect_pii(text)
        .into_iter()
        .map(|s| (s.kind, s.start, s.end))
        .collect()
}

// ---- C11. Task decontamination ----

/// Build a benchmark n-gram fingerprint set from benchmark texts (for
/// contamination checks). Returns a flat list of n-gram hashes (dedup on the
/// Python side into a set).
#[pyfunction]
#[pyo3(signature = (texts, ngram=8))]
fn benchmark_ngrams(texts: Vec<String>, ngram: usize) -> Vec<u64> {
    let mut out = Vec::new();
    for t in &texts {
        out.extend(curate::ngram_hashes(t, ngram));
    }
    out
}

/// Contamination ratio of each doc vs a benchmark n-gram set: fraction of the
/// doc's n-grams present in the benchmark. `benchmark` is the hash list from
/// `benchmark_ngrams`.
#[pyfunction]
#[pyo3(signature = (texts, benchmark, ngram=8))]
fn contamination_batch(texts: Vec<Option<String>>, benchmark: Vec<u64>, ngram: usize) -> Vec<f64> {
    let bset: std::collections::HashSet<u64> = benchmark.into_iter().collect();
    texts
        .into_iter()
        .map(|t| {
            t.map(|s| curate::contamination_ratio(&curate::ngram_hashes(&s, ngram), &bset))
                .unwrap_or(0.0)
        })
        .collect()
}

/// Per-example benchmark n-gram sets (one hash list per benchmark example), for
/// dilution-resistant contamination scoring via `contamination_coverage_batch`.
#[pyfunction]
#[pyo3(signature = (texts, ngram=8))]
fn benchmark_ngram_sets(texts: Vec<String>, ngram: usize) -> Vec<Vec<u64>> {
    texts
        .iter()
        .map(|t| curate::ngram_hashes(t, ngram))
        .collect()
}

/// Dilution-resistant contamination of each doc: the max fraction of ANY single
/// benchmark example's n-grams present in the doc (so a benchmark question
/// buried in a long doc still scores ~1.0). `example_sets` is from
/// `benchmark_ngram_sets`.
#[pyfunction]
#[pyo3(signature = (texts, example_sets, ngram=8))]
fn contamination_coverage_batch(
    texts: Vec<Option<String>>,
    example_sets: Vec<Vec<u64>>,
    ngram: usize,
) -> Vec<f64> {
    texts
        .into_iter()
        .map(|t| {
            t.map(|s| curate::contamination_coverage(&curate::ngram_hashes(&s, ngram), &example_sets))
                .unwrap_or(0.0)
        })
        .collect()
}

// ---- C17. Multimodal (image) curation ----

/// Perceptual hash of one decoded image (H*W*C u8 flat). `kind` = "phash" |
/// "ahash" | "dhash". Near-duplicate images have small Hamming distance.
#[pyfunction]
#[pyo3(signature = (pixels, height, width, channels, kind="phash"))]
fn image_hash(pixels: Vec<u8>, height: usize, width: usize, channels: usize, kind: &str) -> String {
    match kind {
        "ahash" => curate_mm::ahash(&pixels, height, width, channels),
        "dhash" => curate_mm::dhash(&pixels, height, width, channels),
        _ => curate_mm::phash(&pixels, height, width, channels),
    }
}

/// Hamming distance between two hex image-hash strings.
#[pyfunction]
fn image_hash_distance(a: &str, b: &str) -> u32 {
    curate_mm::hamming_hex(a, b)
}

/// Image quality signals as a dict (width/height/aspect_ratio/brightness/
/// sharpness/extreme_ratio) from decoded H*W*C u8 pixels.
#[pyfunction]
fn image_quality<'py>(
    py: Python<'py>,
    pixels: Vec<u8>,
    height: usize,
    width: usize,
    channels: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let q = curate_mm::image_quality(&pixels, height, width, channels);
    let d = PyDict::new(py);
    d.set_item("width", q.width)?;
    d.set_item("height", q.height)?;
    d.set_item("aspect_ratio", q.aspect_ratio)?;
    d.set_item("brightness", q.brightness)?;
    d.set_item("sharpness", q.sharpness)?;
    d.set_item("extreme_ratio", q.extreme_ratio)?;
    Ok(d)
}

// ---- k-means (distributed clustering) ----

/// One map step of Lloyd's k-means over a shard: assign points (flat n*dim) to
/// the nearest centroid + accumulate per-cluster (sum, count). Returns
/// (sums[k][dim], counts[k], inertia) for the driver to merge across shards.
#[pyfunction]
fn kmeans_assign_accumulate(
    points: Vec<f32>,
    n: usize,
    dim: usize,
    centroids: Vec<Vec<f32>>,
) -> (Vec<Vec<f64>>, Vec<u64>, f64) {
    kmeans::assign_accumulate(&points, n, dim, &centroids)
}

/// Assign each point (flat n*dim) to its nearest centroid; returns labels.
#[pyfunction]
fn kmeans_assign_labels(
    points: Vec<f32>,
    n: usize,
    dim: usize,
    centroids: Vec<Vec<f32>>,
) -> Vec<u32> {
    kmeans::assign_labels(&points, n, dim, &centroids)
}

/// Register the `jude.curate` submodule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(chunk_chars, m)?)?;
    m.add_function(wrap_pyfunction!(chunk_recursive, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_text, m)?)?;
    m.add_function(wrap_pyfunction!(content_hash, m)?)?;
    m.add_function(wrap_pyfunction!(content_hash_batch, m)?)?;
    m.add_function(wrap_pyfunction!(quality_signals, m)?)?;
    m.add_function(wrap_pyfunction!(quality_reject_reason, m)?)?;
    m.add_function(wrap_pyfunction!(quality_gate_batch, m)?)?;
    m.add_function(wrap_pyfunction!(minhash_signature, m)?)?;
    m.add_function(wrap_pyfunction!(minhash_signature_batch, m)?)?;
    m.add_function(wrap_pyfunction!(lsh_band_keys, m)?)?;
    m.add_function(wrap_pyfunction!(lsh_band_keys_batch, m)?)?;
    m.add_function(wrap_pyfunction!(signature_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(connected_components, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(semantic_clusters, m)?)?;
    m.add_function(wrap_pyfunction!(detect_language, m)?)?;
    m.add_function(wrap_pyfunction!(detect_language_batch, m)?)?;
    m.add_function(wrap_pyfunction!(image_hash, m)?)?;
    m.add_function(wrap_pyfunction!(image_hash_distance, m)?)?;
    m.add_function(wrap_pyfunction!(image_quality, m)?)?;
    m.add_function(wrap_pyfunction!(redact_pii, m)?)?;
    m.add_function(wrap_pyfunction!(redact_pii_batch, m)?)?;
    m.add_function(wrap_pyfunction!(detect_pii, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_ngrams, m)?)?;
    m.add_function(wrap_pyfunction!(contamination_batch, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_ngram_sets, m)?)?;
    m.add_function(wrap_pyfunction!(contamination_coverage_batch, m)?)?;
    m.add_function(wrap_pyfunction!(kmeans_assign_accumulate, m)?)?;
    m.add_function(wrap_pyfunction!(kmeans_assign_labels, m)?)?;
    Ok(())
}
