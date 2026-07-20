//! LLM data-curation kernels — the compute-heavy cores for jude's positioning
//! as a large-model data-processing engine. Pure Rust (no PyO3 in the logic),
//! exposed as PyO3 functions and consumed by both `Relation` methods and cosmos
//! pipeline `Stage`s (batch, multi-stage — not streaming).
//!
//! Covers the small/high-value Phase-1 operators from
//! `docs/llm_data_engine_plan.zh.md`:
//! - text chunking (character + recursive-separator) — C5
//! - content normalization + hashing for exact dedup — C2
//! - quality-filter heuristics (Gopher/C4-style) — C3
//!
//! Everything here is deterministic and unit-tested without a Python runtime.

use serde::Serialize;
use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// C1. MinHash signatures + LSH banding (fuzzy near-duplicate detection)
// ---------------------------------------------------------------------------

/// Tokenize `text` into word n-gram shingles (lowercased), returning the set of
/// distinct shingle strings joined by a unit separator. n>=1.
pub fn shingles(text: &str, n: usize) -> Vec<String> {
    let n = n.max(1);
    let words: Vec<String> = text.split_whitespace().map(|w| w.to_lowercase()).collect();
    if words.len() < n {
        // whole doc is one shingle (or empty)
        return if words.is_empty() {
            Vec::new()
        } else {
            vec![words.join(" ")]
        };
    }
    let mut out = Vec::with_capacity(words.len() - n + 1);
    for i in 0..=words.len() - n {
        out.push(words[i..i + n].join(" "));
    }
    out
}

/// A fast 64-bit hash of a shingle (FNV-1a — deterministic, no external dep).
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Compute a MinHash signature of length `num_hashes` for `text` using word
/// `n`-gram shingles. Each hash function is `h_i(x) = (a_i * x + b_i) mod p`
/// over the shingle's base hash; the signature entry is the min over shingles.
/// `seed` makes the (a_i, b_i) deterministic across workers.
pub fn minhash_signature(text: &str, num_hashes: usize, ngram: usize, seed: u64) -> Vec<u64> {
    const MERSENNE_P: u64 = (1 << 61) - 1; // large prime
    let sh = shingles(text, ngram);
    if sh.is_empty() {
        return vec![0; num_hashes];
    }
    // Derive deterministic coefficients from seed via a splitmix64 stream.
    let mut state = seed ^ 0x9e3779b97f4a7c15;
    let mut next = || {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    };
    let coeffs: Vec<(u64, u64)> = (0..num_hashes)
        .map(|_| ((next() % (MERSENNE_P - 1)) + 1, next() % MERSENNE_P))
        .collect();

    let base: Vec<u64> = sh.iter().map(|s| fnv1a(s) % MERSENNE_P).collect();
    let mut sig = vec![u64::MAX; num_hashes];
    for &x in &base {
        for (i, &(a, b)) in coeffs.iter().enumerate() {
            let hv = ((a as u128 * x as u128 + b as u128) % MERSENNE_P as u128) as u64;
            if hv < sig[i] {
                sig[i] = hv;
            }
        }
    }
    sig
}

/// Estimated Jaccard similarity of two signatures = fraction of equal entries.
pub fn signature_similarity(a: &[u64], b: &[u64]) -> f64 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let eq = a.iter().zip(b).filter(|(x, y)| x == y).count();
    eq as f64 / a.len() as f64
}

/// LSH band keys for a signature: split `num_hashes` into `bands` bands of
/// `rows = num_hashes/bands` rows each; each band's key is a hash of its rows,
/// prefixed by band index so keys from different bands never collide. Two docs
/// sharing ANY band key are near-duplicate candidates. Returns one string key
/// per band.
pub fn lsh_band_keys(signature: &[u64], bands: usize) -> Vec<String> {
    let bands = bands.max(1).min(signature.len().max(1));
    let rows = signature.len() / bands;
    if rows == 0 {
        return Vec::new();
    }
    let mut keys = Vec::with_capacity(bands);
    for b in 0..bands {
        let start = b * rows;
        let slice = &signature[start..start + rows];
        // hash the band's rows
        let mut h: u64 = 0xcbf29ce484222325;
        for &v in slice {
            h ^= v;
            h = h.wrapping_mul(0x100000001b3);
        }
        keys.push(format!("{b}:{h:016x}"));
    }
    keys
}

// ---------------------------------------------------------------------------
// C4. Language identification (heuristic, no model dependency)
// ---------------------------------------------------------------------------

/// Detect the dominant writing system / language of `text` with a lightweight
/// heuristic (no model): first by Unicode script majority (CJK/Cyrillic/Arabic/
/// Hangul/Hiragana-Katakana/Greek/Devanagari), then for Latin script by a small
/// stopword vote across common European languages. Returns an ISO-639-1-ish
/// code and a confidence in [0,1]. Intended for coarse language *routing/
/// filtering* of a corpus, not fine-grained LID — swap in fastText lid.176 via a
/// UDF when precision matters.
pub fn detect_language(text: &str) -> (String, f64) {
    let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    let mut letters = 0usize;
    for c in text.chars() {
        let script = match c {
            '\u{4E00}'..='\u{9FFF}' | '\u{3400}'..='\u{4DBF}' => "zh",
            '\u{3040}'..='\u{309F}' | '\u{30A0}'..='\u{30FF}' => "ja",
            '\u{AC00}'..='\u{D7AF}' => "ko",
            '\u{0400}'..='\u{04FF}' => "ru",
            '\u{0600}'..='\u{06FF}' => "ar",
            '\u{0370}'..='\u{03FF}' => "el",
            '\u{0900}'..='\u{097F}' => "hi",
            'a'..='z' | 'A'..='Z' => "latin",
            _ => continue,
        };
        *counts.entry(script).or_insert(0) += 1;
        letters += 1;
    }
    if letters == 0 {
        return ("und".to_string(), 0.0);
    }
    // Non-Latin scripts: majority script wins directly.
    let (top_script, top_n) = counts
        .iter()
        .max_by_key(|(_, n)| **n)
        .map(|(s, n)| (*s, *n))
        .unwrap();
    if top_script != "latin" {
        return (top_script.to_string(), top_n as f64 / letters as f64);
    }
    // Latin script: stopword vote among a few languages.
    lang_by_stopwords(text)
}

fn lang_by_stopwords(text: &str) -> (String, f64) {
    // Tiny high-frequency stopword sets — enough for coarse en/es/fr/de/it/pt routing.
    const SETS: &[(&str, &[&str])] = &[
        (
            "en",
            &[
                "the", "and", "of", "to", "in", "is", "that", "it", "for", "was", "with", "as",
            ],
        ),
        (
            "es",
            &[
                "el", "la", "de", "que", "y", "en", "los", "las", "una", "por", "con", "para",
            ],
        ),
        (
            "fr",
            &[
                "le", "la", "de", "et", "les", "des", "un", "une", "que", "pour", "dans", "est",
            ],
        ),
        (
            "de",
            &[
                "der", "die", "und", "das", "den", "von", "mit", "ist", "nicht", "ein", "eine",
                "auch",
            ],
        ),
        (
            "it",
            &[
                "il", "di", "che", "la", "il", "un", "per", "con", "non", "una", "sono", "gli",
            ],
        ),
        (
            "pt",
            &[
                "de", "que", "os", "as", "um", "uma", "para", "com", "nao", "por", "dos", "das",
            ],
        ),
    ];
    let words: Vec<String> = text
        .split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphabetic())
                .flat_map(|c| c.to_lowercase())
                .collect::<String>()
        })
        .filter(|w| !w.is_empty())
        .collect();
    if words.is_empty() {
        return ("und".to_string(), 0.0);
    }
    let wordset: std::collections::HashSet<&str> = words.iter().map(|s| s.as_str()).collect();
    let mut best = ("en", 0usize);
    let mut total_hits = 0usize;
    for (lang, stops) in SETS {
        let hits = stops.iter().filter(|s| wordset.contains(**s)).count();
        total_hits += hits;
        if hits > best.1 {
            best = (lang, hits);
        }
    }
    // confidence: how DOMINANT the winner is among stopword hits (margin), not
    // the absolute fraction of one language's sentinel words — a clearly-English
    // sentence with few of the 12 sentinels should still score high, not ~0.08.
    let conf = if best.1 == 0 {
        0.0
    } else {
        (best.1 as f64 / total_hits as f64).min(1.0)
    };
    (best.0.to_string(), conf)
}

// ---------------------------------------------------------------------------
// C10. PII detection + redaction (regex-free, dependency-light)
// ---------------------------------------------------------------------------

/// A detected PII span: (kind, start_byte, end_byte).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PiiSpan {
    pub kind: String,
    pub start: usize,
    pub end: usize,
}

/// Detect common PII in `text` with hand-rolled scanners (no regex dep):
/// email, URL, IPv4, phone (US-ish), credit-card-like (13-16 digit runs), and
/// long digit runs (SSN-ish 9). Returns spans in start order. Coarse but fast
/// and dependency-free — swap Presidio via a UDF for higher precision.
pub fn detect_pii(text: &str) -> Vec<PiiSpan> {
    let b = text.as_bytes();
    let n = b.len();
    let mut spans = Vec::new();
    let is_word = |c: u8| {
        c.is_ascii_alphanumeric() || c == b'.' || c == b'_' || c == b'%' || c == b'+' || c == b'-'
    };

    let mut i = 0;
    while i < n {
        // email: local@domain.tld
        if b[i] == b'@' {
            // walk back for local part
            let mut s = i;
            while s > 0 && is_word(b[s - 1]) {
                s -= 1;
            }
            // walk forward for domain incl a dot
            let mut e = i + 1;
            let mut saw_dot = false;
            while e < n && (b[e].is_ascii_alphanumeric() || b[e] == b'.' || b[e] == b'-') {
                if b[e] == b'.' {
                    saw_dot = true;
                }
                e += 1;
            }
            if s < i && saw_dot && e > i + 2 {
                spans.push(PiiSpan {
                    kind: "email".into(),
                    start: s,
                    end: e,
                });
                i = e;
                continue;
            }
        }
        // URL: http:// or https://
        if text[i..].starts_with("http://") || text[i..].starts_with("https://") {
            let mut e = i;
            while e < n && !b[e].is_ascii_whitespace() {
                e += 1;
            }
            spans.push(PiiSpan {
                kind: "url".into(),
                start: i,
                end: e,
            });
            i = e;
            continue;
        }
        // digit runs -> classify (ipv4 / phone / card / ssn / generic)
        if b[i].is_ascii_digit() {
            let start = i;
            let mut digits = 0usize;
            let mut e = i;
            // allow separators . - ( ) space within a token, count digits
            while e < n
                && (b[e].is_ascii_digit() || matches!(b[e], b'.' | b'-' | b'(' | b')' | b' '))
            {
                if b[e].is_ascii_digit() {
                    digits += 1;
                } else if e > start && !b[e - 1].is_ascii_digit() && b[e] != b'.' {
                    // avoid runaway on consecutive separators (keep simple)
                }
                e += 1;
            }
            // trim trailing separators
            while e > start && !b[e - 1].is_ascii_digit() {
                e -= 1;
            }
            let tok = &text[start..e];
            let kind = classify_digit_token(tok, digits);
            if let Some(k) = kind {
                spans.push(PiiSpan {
                    kind: k.into(),
                    start,
                    end: e,
                });
            }
            i = e.max(start + 1);
            continue;
        }
        i += 1;
    }
    spans
}

fn luhn_ok(tok: &str) -> bool {
    // Luhn checksum over the ascii digits in `tok` (ignores separators). A real
    // credit-card number passes; a random 13-16 digit run almost never does, so
    // this removes the bulk of credit_card false positives.
    let digits: Vec<u32> = tok
        .bytes()
        .filter(|b| b.is_ascii_digit())
        .map(|b| (b - b'0') as u32)
        .collect();
    if digits.len() < 13 {
        return false;
    }
    // standard Luhn: walk right-to-left, double every SECOND digit (the check
    // digit itself is not doubled), subtract 9 if the doubled value exceeds 9.
    let mut sum = 0u32;
    let mut double = false;
    for &d in digits.iter().rev() {
        let mut v = d;
        if double {
            v *= 2;
            if v > 9 {
                v -= 9;
            }
        }
        sum += v;
        double = !double;
    }
    sum % 10 == 0
}

fn classify_digit_token(tok: &str, digits: usize) -> Option<&'static str> {
    // IPv4: 4 dot-separated groups, each 1-3 digits <=255
    let dot_parts: Vec<&str> = tok.split('.').collect();
    if dot_parts.len() == 4
        && dot_parts.iter().all(|p| {
            !p.is_empty()
                && p.chars().all(|c| c.is_ascii_digit())
                && p.parse::<u16>().map(|v| v <= 255).unwrap_or(false)
        })
    {
        return Some("ipv4");
    }
    match digits {
        // a 13-16 digit run is a credit card only if it passes the Luhn check
        // (else it's just a long number — don't flag it).
        13..=16 if luhn_ok(tok) => Some("credit_card"),
        11..=12 => Some("phone"),
        10 => Some("phone"),
        9 => Some("ssn"),
        _ => None,
    }
}

/// Replace every detected PII span with a `[KIND]` tag. Returns (redacted_text,
/// count). Non-overlapping, left-to-right.
pub fn redact_pii(text: &str) -> (String, usize) {
    let mut spans = detect_pii(text);
    spans.sort_by_key(|s| s.start);
    // drop overlaps (keep earliest)
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0usize;
    let mut count = 0usize;
    for sp in &spans {
        if sp.start < cursor {
            continue; // overlap with a prior span
        }
        out.push_str(&text[cursor..sp.start]);
        out.push('[');
        out.push_str(&sp.kind.to_uppercase());
        out.push(']');
        cursor = sp.end;
        count += 1;
    }
    out.push_str(&text[cursor..]);
    (out, count)
}

// ---------------------------------------------------------------------------
// C11. Task decontamination (n-gram overlap vs a benchmark set)
// ---------------------------------------------------------------------------

/// Collect the set of word n-grams in `text` (lowercased) as hashes — the
/// contamination fingerprint of a document / benchmark example.
pub fn ngram_hashes(text: &str, n: usize) -> Vec<u64> {
    shingles(text, n).iter().map(|s| fnv1a(s)).collect()
}

/// Given a document's n-gram hashes and a set of benchmark n-gram hashes,
/// return the contamination ratio = fraction of the doc's n-grams that appear
/// in the benchmark set. A doc is "contaminated" if this exceeds a threshold.
pub fn contamination_ratio(
    doc_ngrams: &[u64],
    benchmark_set: &std::collections::HashSet<u64>,
) -> f64 {
    if doc_ngrams.is_empty() {
        return 0.0;
    }
    let hit = doc_ngrams
        .iter()
        .filter(|h| benchmark_set.contains(h))
        .count();
    hit as f64 / doc_ngrams.len() as f64
}

/// Dilution-resistant contamination: the maximum fraction of ANY single
/// benchmark example's n-grams that appear in the doc. Unlike
/// `contamination_ratio` (doc-side: matches / doc n-grams — which a long doc
/// dilutes toward 0), this is benchmark-side, so a doc that CONTAINS a full
/// benchmark question scores ~1.0 regardless of how much other text surrounds
/// it. Each element of `example_sets` is one benchmark example's n-gram hashes.
pub fn contamination_coverage(doc_ngrams: &[u64], example_sets: &[Vec<u64>]) -> f64 {
    if doc_ngrams.is_empty() || example_sets.is_empty() {
        return 0.0;
    }
    let doc: std::collections::HashSet<u64> = doc_ngrams.iter().copied().collect();
    let mut best = 0.0f64;
    for ex in example_sets {
        if ex.is_empty() {
            continue; // example shorter than n -> can't fingerprint at this n
        }
        let hit = ex.iter().filter(|h| doc.contains(h)).count();
        let cov = hit as f64 / ex.len() as f64;
        if cov > best {
            best = cov;
        }
    }
    best
}

/// Union-Find (disjoint set) for clustering near-duplicate ids into components.
pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    pub fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]]; // path halving
            x = self.parent[x];
        }
        x
    }

    pub fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] += 1;
        }
    }

    /// Component representative for each element (the min-id in its component),
    /// so the "keep" decision is deterministic: keep the representative.
    pub fn representatives(&mut self, n: usize) -> Vec<usize> {
        // First map root -> min member.
        let mut root_min: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        for i in 0..n {
            let r = self.find(i);
            let e = root_min.entry(r).or_insert(i);
            if i < *e {
                *e = i;
            }
        }
        (0..n).map(|i| root_min[&self.find(i)]).collect()
    }
}

// ---------------------------------------------------------------------------
// C7. Semantic dedup helpers (embedding-space near-duplicate detection)
// ---------------------------------------------------------------------------

/// Cosine similarity of two equal-length float vectors. Returns 0.0 if either
/// is degenerate (zero norm) or lengths mismatch.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0f32;
    let mut na = 0f32;
    let mut nb = 0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Cluster embedding vectors into semantic near-duplicate groups: any pair with
/// cosine similarity >= `threshold` is unioned; returns the component
/// representative (min id) per vector. `embeddings[i]` is the i-th doc's vector;
/// all must share a length (ragged/empty rows are treated as singletons).
///
/// This is the O(n^2) core used *within an LSH/ANN bucket* — the caller narrows
/// candidates first (Lance ANN or a coarse cluster) so n stays small per group.
pub fn semantic_clusters(embeddings: &[Vec<f32>], threshold: f32) -> Vec<usize> {
    let n = embeddings.len();
    let mut uf = UnionFind::new(n);
    for i in 0..n {
        if embeddings[i].is_empty() {
            continue;
        }
        for j in (i + 1)..n {
            if embeddings[j].len() != embeddings[i].len() {
                continue;
            }
            if cosine_similarity(&embeddings[i], &embeddings[j]) >= threshold {
                uf.union(i, j);
            }
        }
    }
    uf.representatives(n)
}

// ---------------------------------------------------------------------------

/// Split `text` into chunks of at most `chunk_chars` characters, with
/// `overlap` characters carried between consecutive chunks. Operates on
/// Unicode scalar values (chars), not bytes, so multi-byte text is safe.
pub fn chunk_chars(text: &str, chunk_chars: usize, overlap: usize) -> Vec<String> {
    let chunk = chunk_chars.max(1);
    let ov = overlap.min(chunk.saturating_sub(1));
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return Vec::new();
    }
    let step = chunk - ov;
    let mut out = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let end = (start + chunk).min(chars.len());
        out.push(chars[start..end].iter().collect());
        if end == chars.len() {
            break;
        }
        start += step;
    }
    out
}

/// Recursive-separator chunking (LangChain-style): try to split on the first
/// separator that keeps pieces under `chunk_chars`, falling back to finer
/// separators, and finally to a hard char split. `separators` are tried in
/// order (e.g. ["\n\n", "\n", ". ", " "]). Adjacent small pieces are greedily
/// merged up to `chunk_chars`, with `overlap` chars carried between chunks.
pub fn chunk_recursive(
    text: &str,
    chunk_chars: usize,
    overlap: usize,
    separators: &[String],
) -> Vec<String> {
    let chunk = chunk_chars.max(1);
    if text.chars().count() <= chunk {
        return if text.is_empty() {
            Vec::new()
        } else {
            vec![text.to_string()]
        };
    }
    // Produce atomic pieces by recursively splitting on separators.
    let pieces = split_recursive(text, chunk, separators, 0);
    // Greedily merge pieces into chunks up to `chunk`, with char overlap.
    merge_pieces(&pieces, chunk, overlap)
}

fn split_recursive(text: &str, chunk: usize, seps: &[String], depth: usize) -> Vec<String> {
    if text.chars().count() <= chunk {
        return vec![text.to_string()];
    }
    if depth >= seps.len() {
        // No separators left: hard char split (no overlap here; merge adds it).
        return chunk_chars(text, chunk, 0);
    }
    let sep = &seps[depth];
    if sep.is_empty() || !text.contains(sep.as_str()) {
        return split_recursive(text, chunk, seps, depth + 1);
    }
    let mut out = Vec::new();
    for part in text.split(sep.as_str()) {
        if part.chars().count() > chunk {
            out.extend(split_recursive(part, chunk, seps, depth + 1));
        } else if !part.is_empty() {
            out.push(part.to_string());
        }
    }
    out
}

fn merge_pieces(pieces: &[String], chunk: usize, overlap: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for p in pieces {
        if cur.is_empty() {
            cur = p.clone();
        } else if cur.chars().count() + 1 + p.chars().count() <= chunk {
            cur.push(' ');
            cur.push_str(p);
        } else {
            out.push(cur.clone());
            // carry `overlap` trailing chars from the finished chunk
            if overlap > 0 {
                let tail: String = {
                    let cs: Vec<char> = cur.chars().collect();
                    let s = cs.len().saturating_sub(overlap);
                    cs[s..].iter().collect()
                };
                cur = if tail.is_empty() {
                    p.clone()
                } else {
                    format!("{tail} {p}")
                };
            } else {
                cur = p.clone();
            }
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

// ---------------------------------------------------------------------------
// C2. Content normalization + hashing (exact dedup)
// ---------------------------------------------------------------------------

/// Normalize text for exact-dedup keying: lowercase, collapse all whitespace
/// runs to a single space, and trim. Two documents that differ only in casing
/// or whitespace hash equal.
pub fn normalize_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut prev_space = true; // leading trim
    for ch in text.chars() {
        if ch.is_whitespace() {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
        } else {
            for lc in ch.to_lowercase() {
                out.push(lc);
            }
            prev_space = false;
        }
    }
    if out.ends_with(' ') {
        out.pop();
    }
    out
}

/// SHA-256 hex of `text` after normalization — the exact-dedup key.
pub fn content_hash(text: &str, normalize: bool) -> String {
    let s = if normalize {
        normalize_text(text)
    } else {
        text.to_string()
    };
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    let digest = h.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for b in digest.iter() {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

// ---------------------------------------------------------------------------
// C3. Quality-filter heuristics (Gopher/C4-style)
// ---------------------------------------------------------------------------

/// Per-document quality signals used by heuristic filters. All ratios are in
/// [0,1]. Computed in one pass where possible.
#[derive(Clone, Debug, Default, Serialize)]
pub struct QualitySignals {
    pub char_count: usize,
    pub word_count: usize,
    pub mean_word_len: f64,
    /// fraction of chars that are alphabetic
    pub alpha_ratio: f64,
    /// fraction of chars that are digits
    pub digit_ratio: f64,
    /// fraction of chars that are symbols/punctuation (non-alnum, non-space)
    pub symbol_ratio: f64,
    /// fraction of words that contain at least one alphabetic char
    pub alpha_word_ratio: f64,
    /// fraction of lines that are duplicates of an earlier line
    pub dup_line_ratio: f64,
    /// '#' (bullet/hash) line fraction — cheap boilerplate signal
    pub hash_line_ratio: f64,
    /// ratio of the most-common word's count to total words (repetition)
    pub top_word_ratio: f64,
    /// fraction of words that are common English stopwords (real prose has
    /// stopwords; keyword spam / boilerplate lists do not) — Gopher signal
    pub stopword_ratio: f64,
    /// fraction of word 3-grams that are duplicates of an earlier 3-gram
    /// (1 - unique/total); high = repetitive/templated text — Gopher signal
    pub dup_ngram_ratio: f64,
}

/// Small English stopword set for the Gopher stopword gate.
const EN_STOPWORDS: &[&str] = &[
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on",
    "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say",
    "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their",
];

/// Compute quality signals for one document.
pub fn quality_signals(text: &str) -> QualitySignals {
    use std::collections::HashMap;

    let mut sig = QualitySignals::default();
    let chars: Vec<char> = text.chars().collect();
    sig.char_count = chars.len();
    if chars.is_empty() {
        return sig;
    }
    let (mut alpha, mut digit, mut symbol) = (0usize, 0usize, 0usize);
    for &c in &chars {
        if c.is_alphabetic() {
            alpha += 1;
        } else if c.is_ascii_digit() {
            digit += 1;
        } else if !c.is_whitespace() {
            symbol += 1;
        }
    }
    let n = chars.len() as f64;
    sig.alpha_ratio = alpha as f64 / n;
    sig.digit_ratio = digit as f64 / n;
    sig.symbol_ratio = symbol as f64 / n;

    let words: Vec<&str> = text.split_whitespace().collect();
    sig.word_count = words.len();
    if !words.is_empty() {
        let total_len: usize = words.iter().map(|w| w.chars().count()).sum();
        sig.mean_word_len = total_len as f64 / words.len() as f64;
        let alpha_words = words
            .iter()
            .filter(|w| w.chars().any(|c| c.is_alphabetic()))
            .count();
        sig.alpha_word_ratio = alpha_words as f64 / words.len() as f64;
        let mut freq: HashMap<&str, usize> = HashMap::new();
        for w in &words {
            *freq.entry(*w).or_insert(0) += 1;
        }
        let top = freq.values().copied().max().unwrap_or(0);
        sig.top_word_ratio = top as f64 / words.len() as f64;

        // Gopher stopword gate: fraction of words that are common stopwords.
        let stop: std::collections::HashSet<&str> = EN_STOPWORDS.iter().copied().collect();
        let sw = words
            .iter()
            .filter(|w| {
                let lw: String = w.chars().flat_map(|c| c.to_lowercase()).collect();
                stop.contains(lw.as_str())
            })
            .count();
        sig.stopword_ratio = sw as f64 / words.len() as f64;

        // Gopher repetition: duplicate word 3-gram fraction (1 - unique/total).
        if words.len() >= 3 {
            let total = words.len() - 2;
            let mut seen3 = std::collections::HashSet::new();
            let mut dup3 = 0usize;
            for w in words.windows(3) {
                if !seen3.insert((w[0], w[1], w[2])) {
                    dup3 += 1;
                }
            }
            sig.dup_ngram_ratio = dup3 as f64 / total as f64;
        }
    }

    let lines: Vec<&str> = text.lines().collect();
    if !lines.is_empty() {
        let mut seen = std::collections::HashSet::new();
        let mut dup = 0usize;
        let mut hash_lines = 0usize;
        for l in &lines {
            let t = l.trim();
            if !seen.insert(t) {
                dup += 1;
            }
            if t.starts_with('#') {
                hash_lines += 1;
            }
        }
        sig.dup_line_ratio = dup as f64 / lines.len() as f64;
        sig.hash_line_ratio = hash_lines as f64 / lines.len() as f64;
    }
    sig
}

/// A Gopher/C4-style pass/fail verdict with the reason on failure. `None`
/// thresholds are skipped. Defaults mirror common heuristic filters.
#[derive(Clone, Debug)]
pub struct QualityThresholds {
    pub min_words: usize,
    pub max_words: usize,
    pub min_mean_word_len: f64,
    pub max_mean_word_len: f64,
    pub max_symbol_ratio: f64,
    pub min_alpha_word_ratio: f64,
    pub max_dup_line_ratio: f64,
    pub max_top_word_ratio: f64,
    pub max_digit_ratio: f64,
    pub min_stopword_ratio: f64,
    pub max_dup_ngram_ratio: f64,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        // Gopher-ish defaults.
        QualityThresholds {
            min_words: 50,
            max_words: 100_000,
            min_mean_word_len: 3.0,
            max_mean_word_len: 10.0,
            max_symbol_ratio: 0.30,
            min_alpha_word_ratio: 0.60,
            max_dup_line_ratio: 0.30,
            max_top_word_ratio: 0.30,
            max_digit_ratio: 0.30,
            min_stopword_ratio: 0.06,
            max_dup_ngram_ratio: 0.30,
        }
    }
}

/// Returns None if the doc passes, else Some(reason) for the first failed rule.
pub fn quality_reject_reason(sig: &QualitySignals, t: &QualityThresholds) -> Option<String> {
    if sig.word_count < t.min_words {
        return Some(format!("too_few_words:{}<{}", sig.word_count, t.min_words));
    }
    if sig.word_count > t.max_words {
        return Some(format!("too_many_words:{}>{}", sig.word_count, t.max_words));
    }
    if sig.mean_word_len < t.min_mean_word_len {
        return Some(format!("mean_word_len_low:{:.2}", sig.mean_word_len));
    }
    if sig.mean_word_len > t.max_mean_word_len {
        return Some(format!("mean_word_len_high:{:.2}", sig.mean_word_len));
    }
    if sig.symbol_ratio > t.max_symbol_ratio {
        return Some(format!("symbol_ratio_high:{:.2}", sig.symbol_ratio));
    }
    if sig.alpha_word_ratio < t.min_alpha_word_ratio {
        return Some(format!("alpha_word_ratio_low:{:.2}", sig.alpha_word_ratio));
    }
    if sig.dup_line_ratio > t.max_dup_line_ratio {
        return Some(format!("dup_line_ratio_high:{:.2}", sig.dup_line_ratio));
    }
    if sig.top_word_ratio > t.max_top_word_ratio {
        return Some(format!("top_word_ratio_high:{:.2}", sig.top_word_ratio));
    }
    if sig.digit_ratio > t.max_digit_ratio {
        return Some(format!("digit_ratio_high:{:.2}", sig.digit_ratio));
    }
    // Gopher stopword gate: only apply to docs with enough words to be prose
    // (short snippets legitimately have few stopwords).
    if sig.word_count >= t.min_words && sig.stopword_ratio < t.min_stopword_ratio {
        return Some(format!("stopword_ratio_low:{:.3}", sig.stopword_ratio));
    }
    if sig.dup_ngram_ratio > t.max_dup_ngram_ratio {
        return Some(format!("dup_ngram_ratio_high:{:.2}", sig.dup_ngram_ratio));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_chars_overlap() {
        let c = chunk_chars("abcdefghij", 4, 1);
        // step = 3: [0..4]=abcd, [3..7]=defg, [6..10]=ghij (end==len, stop)
        assert_eq!(c, vec!["abcd", "defg", "ghij"]);
    }

    #[test]
    fn chunk_chars_no_overlap_exact() {
        assert_eq!(chunk_chars("abcdef", 3, 0), vec!["abc", "def"]);
        assert_eq!(chunk_chars("", 3, 0), Vec::<String>::new());
        assert_eq!(chunk_chars("ab", 5, 0), vec!["ab"]);
    }

    #[test]
    fn chunk_chars_unicode_safe() {
        let c = chunk_chars("日本語テスト", 2, 0);
        assert_eq!(c, vec!["日本", "語テ", "スト"]);
    }

    #[test]
    fn recursive_prefers_separators() {
        let text = "para one here.\n\npara two here.\n\npara three here.";
        let seps = vec!["\n\n".to_string(), "\n".to_string(), " ".to_string()];
        let c = chunk_recursive(text, 20, 0, &seps);
        // each paragraph is <20 chars, so they stay whole-ish (merged up to 20)
        assert!(c.iter().all(|s| s.chars().count() <= 20 + 5));
        assert!(c.len() >= 2);
    }

    #[test]
    fn recursive_small_text_one_chunk() {
        let seps = vec!["\n".to_string()];
        assert_eq!(chunk_recursive("short", 100, 0, &seps), vec!["short"]);
    }

    #[test]
    fn normalize_collapses_ws_and_case() {
        assert_eq!(normalize_text("  Hello   WORLD\n\tfoo "), "hello world foo");
        assert_eq!(normalize_text(""), "");
    }

    #[test]
    fn content_hash_dedups_whitespace_case() {
        let a = content_hash("Hello World", true);
        let b = content_hash("  hello    world  ", true);
        assert_eq!(a, b);
        let c = content_hash("hello world!", true);
        assert_ne!(a, c);
    }

    #[test]
    fn content_hash_raw_differs() {
        assert_ne!(content_hash("Hello", false), content_hash("hello", false));
    }

    #[test]
    fn quality_signals_basic() {
        let s = quality_signals("the quick brown fox jumps over the lazy dog");
        assert_eq!(s.word_count, 9);
        assert!(s.alpha_ratio > 0.7);
        assert_eq!(s.digit_ratio, 0.0);
        // "the" appears twice of 9 words
        assert!((s.top_word_ratio - 2.0 / 9.0).abs() < 1e-9);
    }

    #[test]
    fn quality_rejects_symbol_spam() {
        let sig = quality_signals("!@#$ %^&* !@#$ %^&* !@#$ %^&*");
        let reason = quality_reject_reason(&sig, &QualityThresholds::default());
        assert!(reason.is_some(), "symbol spam should be rejected");
    }

    #[test]
    fn quality_stopword_gate_and_repetition() {
        // Gopher signals: a real prose paragraph has stopwords + low repetition.
        let prose = "the study shows that the results are consistent with the theory \
                     and the data supports this conclusion for the most part in every case \
                     that we have examined so far across the many different samples we ran";
        let s = quality_signals(prose);
        assert!(
            s.stopword_ratio > 0.1,
            "prose should have stopwords: {}",
            s.stopword_ratio
        );
        assert!(
            s.dup_ngram_ratio < 0.2,
            "prose isn't repetitive: {}",
            s.dup_ngram_ratio
        );

        // keyword-spam list: 60 words, almost no stopwords -> stopword gate fires.
        let spam = (0..60)
            .map(|i| format!("keyword{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let ss = quality_signals(&spam);
        let r = quality_reject_reason(&ss, &QualityThresholds::default());
        assert_eq!(
            r.as_deref(),
            Some(&format!("stopword_ratio_low:{:.3}", ss.stopword_ratio)[..])
        );

        // highly repetitive text -> high dup_ngram_ratio, and it is rejected.
        let rep = "the cat sat on the mat ".repeat(12);
        let rs = quality_signals(&rep);
        assert!(
            rs.dup_ngram_ratio > 0.5,
            "repetitive dup_ngram: {}",
            rs.dup_ngram_ratio
        );
        let rr = quality_reject_reason(&rs, &QualityThresholds::default());
        assert!(rr.is_some(), "repetitive text rejected");
    }

    #[test]
    fn quality_digit_gate() {
        // a 60-"word" doc that is mostly digits -> digit_ratio gate fires.
        let digits = (0..60).map(|_| "1234567").collect::<Vec<_>>().join(" ");
        let s = quality_signals(&digits);
        let r = quality_reject_reason(&s, &QualityThresholds::default());
        assert!(r.is_some(), "digit-heavy doc should be rejected: {r:?}");
    }

    #[test]
    fn quality_rejects_too_short() {
        let sig = quality_signals("just a few words here");
        let reason = quality_reject_reason(&sig, &QualityThresholds::default());
        assert!(reason.unwrap().starts_with("too_few_words"));
    }

    #[test]
    fn quality_passes_normal_prose() {
        // ~60 words of normal prose
        let text = "The history of natural language processing began in the nineteen fifties, \
            although work can be found from earlier periods. In nineteen fifty, Alan Turing \
            published an article titled Computing Machinery and Intelligence which proposed \
            what is now called the Turing test as a criterion of intelligence, a task that \
            involves the automated interpretation and generation of natural human language.";
        let sig = quality_signals(text);
        let reason = quality_reject_reason(&sig, &QualityThresholds::default());
        assert!(reason.is_none(), "normal prose should pass, got {reason:?}");
    }

    #[test]
    fn quality_rejects_duplicate_lines() {
        let text = "same line\n".repeat(100);
        let sig = quality_signals(&text);
        assert!(sig.dup_line_ratio > 0.9);
        let reason = quality_reject_reason(&sig, &QualityThresholds::default());
        assert!(reason.is_some());
    }

    // ---- MinHash / LSH ----

    #[test]
    fn shingles_ngrams() {
        assert_eq!(
            shingles("the quick brown fox", 2),
            vec!["the quick", "quick brown", "brown fox"]
        );
        assert_eq!(shingles("one", 2), vec!["one"]); // shorter than n
        assert_eq!(shingles("", 2), Vec::<String>::new());
    }

    #[test]
    fn minhash_identical_texts_match() {
        let a = minhash_signature("the quick brown fox jumps", 64, 2, 42);
        let b = minhash_signature("the quick brown fox jumps", 64, 2, 42);
        assert_eq!(a, b);
        assert!((signature_similarity(&a, &b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn minhash_near_duplicates_high_sim() {
        // differ by one word out of ~12 -> Jaccard of 2-shingles still high
        let a = minhash_signature(
            "the quick brown fox jumps over the lazy dog in the yard",
            128,
            2,
            7,
        );
        let b = minhash_signature(
            "the quick brown fox jumps over the lazy dog in the park",
            128,
            2,
            7,
        );
        let sim = signature_similarity(&a, &b);
        assert!(sim > 0.6, "near-dup similarity too low: {sim}");
    }

    #[test]
    fn minhash_unrelated_low_sim() {
        let a = minhash_signature("the quick brown fox jumps over the lazy dog", 128, 2, 7);
        let b = minhash_signature(
            "completely different content about databases and rust",
            128,
            2,
            7,
        );
        let sim = signature_similarity(&a, &b);
        assert!(sim < 0.2, "unrelated similarity too high: {sim}");
    }

    #[test]
    fn lsh_bands_collide_for_near_dups() {
        let a = minhash_signature(
            "the quick brown fox jumps over the lazy dog in the yard today",
            128,
            2,
            7,
        );
        let b = minhash_signature(
            "the quick brown fox jumps over the lazy dog in the yard now",
            128,
            2,
            7,
        );
        let ka = lsh_band_keys(&a, 32);
        let kb = lsh_band_keys(&b, 32);
        let shared = ka.iter().filter(|k| kb.contains(k)).count();
        assert!(shared >= 1, "near-dups should share at least one band key");
    }

    #[test]
    fn lsh_bands_isolate_unrelated() {
        let a = minhash_signature("the quick brown fox jumps over the lazy dog", 128, 2, 7);
        let b = minhash_signature(
            "rust systems programming with zero cost abstractions everywhere",
            128,
            2,
            7,
        );
        let ka = lsh_band_keys(&a, 32);
        let kb = lsh_band_keys(&b, 32);
        let shared = ka.iter().filter(|k| kb.contains(k)).count();
        assert_eq!(shared, 0, "unrelated docs should not share band keys");
    }

    #[test]
    fn union_find_components() {
        let mut uf = UnionFind::new(5);
        uf.union(0, 1);
        uf.union(1, 2);
        uf.union(3, 4);
        let reps = uf.representatives(5);
        // {0,1,2} -> rep 0 ; {3,4} -> rep 3
        assert_eq!(reps[0], reps[1]);
        assert_eq!(reps[1], reps[2]);
        assert_eq!(reps[0], 0);
        assert_eq!(reps[3], reps[4]);
        assert_eq!(reps[3], 3);
        assert_ne!(reps[0], reps[3]);
    }

    // ---- Semantic dedup ----

    #[test]
    fn cosine_basics() {
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert!((cosine_similarity(&[1.0, 1.0], &[2.0, 2.0]) - 1.0).abs() < 1e-6); // scale-invariant
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0); // degenerate
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0); // mismatched len
    }

    #[test]
    fn semantic_clusters_groups_similar() {
        // 0 and 1 are nearly identical directions; 2 is orthogonal.
        let embs = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.99, 0.01, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let reps = semantic_clusters(&embs, 0.9);
        assert_eq!(reps[0], reps[1]); // near-dups grouped
        assert_ne!(reps[0], reps[2]); // orthogonal separate
    }

    #[test]
    fn semantic_clusters_all_distinct() {
        let embs = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0]];
        let reps = semantic_clusters(&embs, 0.9);
        // all mutually dissimilar -> each its own rep
        assert_eq!(reps, vec![0, 1, 2]);
    }

    // ---- Language ID ----

    #[test]
    fn detect_language_scripts() {
        assert_eq!(detect_language("这是一段中文文本内容").0, "zh");
        assert_eq!(detect_language("これは日本語のテキストです").0, "ja");
        assert_eq!(detect_language("이것은 한국어 텍스트입니다").0, "ko");
        assert_eq!(detect_language("это русский текст").0, "ru");
        assert_eq!(detect_language("").0, "und");
        assert_eq!(detect_language("12345 !!!").0, "und");
    }

    #[test]
    fn detect_language_latin_stopwords() {
        assert_eq!(
            detect_language("the quick brown fox and the lazy dog is in the yard").0,
            "en"
        );
        assert_eq!(
            detect_language("le chat et le chien sont dans la maison des amis").0,
            "fr"
        );
        assert_eq!(
            detect_language("der Hund und die Katze sind nicht im Haus von mir").0,
            "de"
        );
        // confidence is > 0 when stopwords hit
        assert!(detect_language("the and of to in is").1 > 0.0);
    }

    // ---- PII ----

    #[test]
    fn detect_pii_email_and_url() {
        let t = "contact me at john.doe@example.com or visit https://example.com/page now";
        let spans = detect_pii(t);
        let kinds: Vec<&str> = spans.iter().map(|s| s.kind.as_str()).collect();
        assert!(kinds.contains(&"email"), "kinds={kinds:?}");
        assert!(kinds.contains(&"url"), "kinds={kinds:?}");
    }

    #[test]
    fn detect_pii_ipv4_and_digits() {
        let spans = detect_pii("server at 192.168.1.1 ssn 123456789 card 4111111111111111");
        let kinds: Vec<&str> = spans.iter().map(|s| s.kind.as_str()).collect();
        assert!(kinds.contains(&"ipv4"), "kinds={kinds:?}");
        assert!(kinds.contains(&"ssn"), "kinds={kinds:?}");
        assert!(kinds.contains(&"credit_card"), "kinds={kinds:?}");
    }

    #[test]
    fn credit_card_requires_luhn() {
        // 4111111111111111 passes Luhn -> credit_card; 1234567812345678 fails ->
        // not flagged (no false positive on an arbitrary 16-digit run).
        assert!(luhn_ok("4111111111111111"));
        assert!(!luhn_ok("1234567812345678"));
        let good_spans = detect_pii("pay 4111 1111 1111 1111");
        let good: Vec<&str> = good_spans.iter().map(|s| s.kind.as_str()).collect();
        assert!(good.contains(&"credit_card"), "kinds={good:?}");
        let bad_spans = detect_pii("order 1234567812345678");
        let bad: Vec<&str> = bad_spans.iter().map(|s| s.kind.as_str()).collect();
        assert!(!bad.contains(&"credit_card"), "kinds={bad:?}");
    }

    #[test]
    fn redact_pii_replaces() {
        let (out, count) = redact_pii("email a@b.com and ip 10.0.0.1 here");
        assert!(count >= 2, "count={count} out={out}");
        assert!(out.contains("[EMAIL]"));
        assert!(out.contains("[IPV4]"));
        assert!(!out.contains("a@b.com"));
    }

    #[test]
    fn redact_pii_clean_text_unchanged() {
        let (out, count) = redact_pii("just some ordinary words with no personal info");
        assert_eq!(count, 0);
        assert_eq!(out, "just some ordinary words with no personal info");
    }

    // ---- decontamination ----

    #[test]
    fn contamination_detects_overlap() {
        use std::collections::HashSet;
        let bench = "what is the capital of france paris";
        let bset: HashSet<u64> = ngram_hashes(bench, 3).into_iter().collect();
        // a doc containing the benchmark question -> high contamination
        let doc = "here we ask what is the capital of france paris is the answer";
        let ratio = contamination_ratio(&ngram_hashes(doc, 3), &bset);
        assert!(ratio > 0.2, "ratio={ratio}");
        // an unrelated doc -> ~0
        let clean = "rust systems programming with memory safety and zero cost abstractions";
        let r2 = contamination_ratio(&ngram_hashes(clean, 3), &bset);
        assert!(r2 < 0.05, "r2={r2}");
    }

    #[test]
    fn contamination_coverage_resists_dilution() {
        let bench = "what is the capital of france";
        let example = vec![ngram_hashes(bench, 3)];
        // the benchmark question buried in a LONG doc: doc-side ratio dilutes to
        // ~0, but coverage stays ~1.0 (the full question's n-grams are present).
        let padding = "lorem ipsum dolor sit amet ".repeat(200);
        let long_doc = format!("{bench} {padding}");
        let dn = ngram_hashes(&long_doc, 3);
        let ratio = contamination_ratio(&dn, &example[0].iter().copied().collect());
        let cov = contamination_coverage(&dn, &example);
        assert!(
            ratio < 0.2,
            "diluted doc-side ratio should be small: {ratio}"
        );
        assert!(
            cov > 0.9,
            "coverage should catch the buried question: {cov}"
        );
        // an unrelated long doc -> coverage ~0
        let clean = "rust systems programming ".repeat(50);
        assert!(contamination_coverage(&ngram_hashes(&clean, 3), &example) < 0.05);
    }
}
