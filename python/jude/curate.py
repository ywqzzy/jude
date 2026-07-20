"""jude.curate — LLM data-curation operators (batch, stage-based).

jude's positioning is a **large-model data-processing engine**: the operators
that prepare/clean/dedup text for LLM training / RAG. These are the compute
cores (Rust, in ``jude.jude._curate``) wrapped as Arrow-table transforms, usable
three ways:

1. directly on a pyarrow Table / jude Relation column,
2. as cosmos pipeline ``Stage`` s (``jude.pipeline`` over cosmos-xenna) — data
   curation is multi-stage BATCH, not streaming,
3. as ``map_batches`` UDFs.

Phase-1 operators (see docs/llm_data_engine_plan.zh.md):
- ``chunk_text``      (C5) — split a text column into chunks (1 row -> many)
- ``add_content_hash``(C2) — add a normalized SHA-256 column for exact dedup
- ``exact_dedup``     (C2) — drop duplicate rows by content hash
- ``quality_signals`` (C3) — add per-doc quality metric columns
- ``quality_filter``  (C3) — keep rows passing Gopher/C4-style heuristics
"""

from __future__ import annotations

import functools
from typing import Any

import pyarrow as pa

from .jude import _curate


# --- LSH band calibration (C3) ----------------------------------------------


def optimal_lsh_bands(threshold: float, num_hashes: int, *,
                      fp_weight: float = 0.5, fn_weight: float = 0.5) -> int:
    """Pick the number of LSH bands ``b`` (with rows ``r = num_hashes // b``)
    whose S-curve ``p(s) = 1 - (1 - s**r)**b`` best matches the Jaccard
    ``threshold`` — minimizing the weighted false-positive + false-negative area
    (the datasketch calibration). A fixed band count (the old default 16) is only
    well-tuned near threshold 0.7; at other thresholds it silently loses recall
    (too few candidates) or wastes work. Deriving ``b`` from the threshold fixes
    that. Returns ``b`` in ``[1, num_hashes]``.
    """
    if num_hashes <= 1:
        return 1
    threshold = min(max(float(threshold), 1e-6), 1.0 - 1e-6)

    def _area(f, lo, hi, steps=64):  # trapezoidal integral of f over [lo, hi]
        if hi <= lo:
            return 0.0
        h = (hi - lo) / steps
        total = 0.5 * (f(lo) + f(hi))
        for i in range(1, steps):
            total += f(lo + i * h)
        return total * h

    best_b, best_err = 1, float("inf")
    seen_r: set = set()
    for b in range(1, num_hashes + 1):
        r = num_hashes // b
        if r < 1:
            break
        if (b, r) in seen_r:
            continue
        seen_r.add((b, r))
        def p(s, _b=b, _r=r):
            return 1.0 - (1.0 - s ** _r) ** _b
        fp = _area(lambda s, _p=p: _p(s), 0.0, threshold)          # candidates below threshold
        fn = _area(lambda s, _p=p: 1.0 - _p(s), threshold, 1.0)    # misses above threshold
        err = fp_weight * fp + fn_weight * fn
        if err < best_err:
            best_err, best_b = err, b
    return best_b


def _observed(op: str):
    """Decorator: record a row-reducing curation op's rows_in→rows_out into the
    observability registry (data-quality keep-rate), guarded so it's a no-op when
    observe is unavailable or the first arg isn't an Arrow table. Transparent —
    the wrapped function's return value is unchanged."""

    def deco(fn):
        @functools.wraps(fn)
        def wrap(table, *a, **kw):
            nin = getattr(table, "num_rows", None)
            if nin is None:
                return fn(table, *a, **kw)
            try:
                from jude import observe
            except Exception:  # noqa: BLE001 — observe optional
                return fn(table, *a, **kw)
            with observe.curate(op, rows_in=int(nin)) as c:
                out = fn(table, *a, **kw)
                c.done(rows_out=int(getattr(out, "num_rows", nin)))
                return out

        return wrap

    return deco


__all__ = [
    "chunk_text",
    "add_content_hash",
    "exact_dedup",
    "quality_signals",
    "quality_filter",
    "minhash_signatures",
    "fuzzy_dedup",
    "semantic_dedup",
    "detect_language",
    "language_filter",
    "blend_datasets",
    "global_shuffle",
    "redact_pii",
    "detect_pii",
    "decontaminate",
    # cosmos stages
    "ChunkStage",
    "QualityFilterStage",
    "ContentHashStage",
]


def _col(table: pa.Table, column: str) -> list:
    if column not in table.column_names:
        raise KeyError(f"column {column!r} not in table (have {table.column_names})")
    return table.column(column).to_pylist()


# --- C5. chunking (1 row -> many) -------------------------------------------


def chunk_text(
    table: pa.Table,
    *,
    column: str = "text",
    out_column: str = "chunk",
    chunk_chars: int = 1024,
    overlap: int = 0,
    recursive: bool = True,
    separators: list[str] | None = None,
    index_column: str = "chunk_index",
) -> pa.Table:
    """Split ``column`` into chunks, emitting one row per chunk (other columns
    replicated). Adds ``out_column`` (the chunk text) and ``index_column`` (its
    0-based position within the source row). Recursive chunking respects
    separators (paragraph/line/sentence/word); char chunking is a hard split.
    """
    texts = _col(table, column)
    # Build the exploded row-index map + chunk lists in one pass.
    take_idx: list[int] = []
    chunks: list[str] = []
    chunk_idx: list[int] = []
    for row, t in enumerate(texts):
        if t is None:
            pieces: list[str] = []
        elif recursive:
            pieces = _curate.chunk_recursive(t, chunk_chars, overlap, separators)
        else:
            pieces = _curate.chunk_chars(t, chunk_chars, overlap)
        for j, piece in enumerate(pieces):
            take_idx.append(row)
            chunks.append(piece)
            chunk_idx.append(j)
    if not take_idx:
        # empty result: keep schema + the two new columns
        base = table.slice(0, 0)
        base = base.append_column(out_column, pa.array([], type=pa.string()))
        base = base.append_column(index_column, pa.array([], type=pa.int32()))
        return base
    # Replicate the source columns via take, then append chunk + index.
    out = table.take(pa.array(take_idx, type=pa.int64()))
    out = out.append_column(out_column, pa.array(chunks, type=pa.string()))
    out = out.append_column(index_column, pa.array(chunk_idx, type=pa.int32()))
    return out


# --- C2. exact dedup ---------------------------------------------------------


def add_content_hash(
    table: pa.Table, *, column: str = "text", out_column: str = "content_hash", normalize: bool = True
) -> pa.Table:
    """Add a SHA-256 content-hash column (normalized by default: lowercase +
    collapsed whitespace) — the key for exact dedup."""
    hashes = _curate.content_hash_batch(_col(table, column), normalize)
    return table.append_column(out_column, pa.array(hashes, type=pa.string()))


@_observed("exact_dedup")
def exact_dedup(
    table: pa.Table, *, column: str = "text", normalize: bool = True, keep_hash: bool = False
) -> pa.Table:
    """Drop duplicate rows by normalized content hash, keeping the first
    occurrence (stable). ``keep_hash`` retains the ``content_hash`` column."""
    hashes = _curate.content_hash_batch(_col(table, column), normalize)
    seen: set = set()
    keep: list[int] = []
    for i, h in enumerate(hashes):
        if h is None or h in seen:
            continue
        seen.add(h)
        keep.append(i)
    out = table.take(pa.array(keep, type=pa.int64()))
    if keep_hash:
        kept_hashes = [hashes[i] for i in keep]
        out = out.append_column("content_hash", pa.array(kept_hashes, type=pa.string()))
    return out


# --- C3. quality --------------------------------------------------------------

_QUALITY_FIELDS = [
    ("char_count", pa.int64()),
    ("word_count", pa.int64()),
    ("mean_word_len", pa.float64()),
    ("alpha_ratio", pa.float64()),
    ("digit_ratio", pa.float64()),
    ("symbol_ratio", pa.float64()),
    ("alpha_word_ratio", pa.float64()),
    ("dup_line_ratio", pa.float64()),
    ("hash_line_ratio", pa.float64()),
    ("top_word_ratio", pa.float64()),
]


def quality_signals(table: pa.Table, *, column: str = "text", prefix: str = "q_") -> pa.Table:
    """Add per-document quality-signal columns (prefixed) for inspection or
    custom filtering: word_count, mean_word_len, symbol_ratio, dup_line_ratio,
    top_word_ratio, etc."""
    texts = _col(table, column)
    cols: dict[str, list] = {name: [] for name, _ in _QUALITY_FIELDS}
    for t in texts:
        sig = _curate.quality_signals(t) if t is not None else {}
        for name, _ in _QUALITY_FIELDS:
            cols[name].append(sig.get(name) if t is not None else None)
    out = table
    for name, typ in _QUALITY_FIELDS:
        out = out.append_column(prefix + name, pa.array(cols[name], type=typ))
    return out


@_observed("quality_filter")
def quality_filter(
    table: pa.Table,
    *,
    column: str = "text",
    reason_column: str | None = None,
    **thresholds: Any,
) -> pa.Table:
    """Keep rows passing Gopher/C4-style heuristics. Threshold kwargs override
    defaults (min_words, max_words, min_mean_word_len, max_mean_word_len,
    max_symbol_ratio, min_alpha_word_ratio, max_dup_line_ratio,
    max_top_word_ratio). If ``reason_column`` is set, instead of dropping,
    keep ALL rows and add a column with the reject reason (None = kept)."""
    verdicts = _curate.quality_gate_batch(_col(table, column), **thresholds)
    if reason_column is not None:
        reasons = [r for (_keep, r) in verdicts]
        return table.append_column(reason_column, pa.array(reasons, type=pa.string()))
    keep = [i for i, (k, _r) in enumerate(verdicts) if k]
    return table.take(pa.array(keep, type=pa.int64()))


# --- C1. fuzzy dedup (MinHash + LSH) -----------------------------------------


def minhash_signatures(
    table: pa.Table, *, column: str = "text", num_hashes: int = 128, ngram: int = 2, seed: int = 1
) -> list:
    """MinHash signatures (list[list[int]]) for a text column. Near-duplicate
    documents have high signature agreement (estimated Jaccard)."""
    return _curate.minhash_signature_batch(_col(table, column), num_hashes, ngram, seed)


@_observed("fuzzy_dedup")
def fuzzy_dedup(
    table: pa.Table,
    *,
    column: str = "text",
    num_hashes: int = 128,
    ngram: int = 2,
    bands: int | None = None,
    threshold: float = 0.7,
    seed: int = 1,
    keep_cluster: bool = False,
) -> pa.Table:
    """Near-duplicate removal via MinHash + LSH (C1), keeping one document per
    near-dup cluster (the lowest row index — deterministic).

    Pipeline: MinHash signature per doc → LSH band buckets (docs sharing a band
    key are candidates) → verify candidate pairs by estimated Jaccard ≥
    ``threshold`` → union-find into clusters → keep one per cluster. This is the
    single-node form; ``RayRunner.distributed_fuzzy_dedup`` shuffles the LSH
    buckets across workers for scale. ``keep_cluster`` adds a ``dup_cluster``
    column (the representative row id) instead of dropping.

    ``bands`` defaults to a value CALIBRATED to ``threshold`` (see
    ``optimal_lsh_bands``): the S-curve crossover is aligned to the threshold so
    recall isn't lost to a mis-tuned band count. Pass an explicit ``bands`` to
    override.
    """
    texts = _col(table, column)
    n = len(texts)
    if n == 0:
        return table
    if bands is None:
        bands = optimal_lsh_bands(threshold, num_hashes)
    sigs = _curate.minhash_signature_batch(texts, num_hashes, ngram, seed)
    band_keys = _curate.lsh_band_keys_batch(sigs, bands)

    # Collapse EXACT-identical signatures first: heavy-dup corpora otherwise
    # blow up the per-bucket O(n^2) pair scan. Map each row to a "leader" (the
    # first row with its signature); only leaders enter the candidate scan, and
    # every row is unioned to its leader at the end.
    sig_leader: dict[tuple, int] = {}
    leader_of: list[int] = [0] * n
    for i, s in enumerate(sigs):
        key = tuple(s)
        lead = sig_leader.setdefault(key, i)
        leader_of[i] = lead

    # bucket LEADER ids by band key; leaders in the same bucket are candidates
    buckets: dict[str, list[int]] = {}
    for i, keys in enumerate(band_keys):
        if leader_of[i] != i:
            continue  # non-leader: identical to its leader, skip
        for k in keys:
            buckets.setdefault(k, []).append(i)

    # candidate pairs among leaders, verified by estimated Jaccard >= threshold
    pairs: list[tuple[int, int]] = [(i, leader_of[i]) for i in range(n) if leader_of[i] != i]
    checked: set = set()
    for ids in buckets.values():
        if len(ids) < 2:
            continue
        for a_i in range(len(ids)):
            for b_i in range(a_i + 1, len(ids)):
                a, b = ids[a_i], ids[b_i]
                key = (a, b)
                if key in checked:
                    continue
                checked.add(key)
                if _curate.signature_similarity(sigs[a], sigs[b]) >= threshold:
                    pairs.append((a, b))

    reps = _curate.connected_components(n, pairs)
    if keep_cluster:
        return table.append_column("dup_cluster", pa.array(reps, type=pa.int64()))
    keep = [i for i in range(n) if reps[i] == i]
    return table.take(pa.array(keep, type=pa.int64()))


# --- C7. semantic dedup (embedding + clustering) ----------------------------


@_observed("semantic_dedup")
def semantic_dedup(
    table: pa.Table,
    *,
    embedding_column: str = "embedding",
    threshold: float = 0.9,
    keep_cluster: bool = False,
) -> pa.Table:
    """Remove semantic near-duplicates (SemDeDup): documents whose *embeddings*
    are cosine-similar >= ``threshold`` are collapsed to one (lowest row index).

    Unlike MinHash fuzzy dedup (lexical overlap), this catches "same meaning,
    different words" — the stronger dedup SOTA datasets use. Operates on an
    existing embedding column (produce it with ``jude.ai.embed_text`` or any
    embedder). ``keep_cluster`` annotates ``sem_cluster`` instead of dropping.

    This runs the O(n^2) clustering over the WHOLE table; for scale, first bucket
    by a coarse cluster / Lance ANN (jude's vector stack) so each group is small,
    then apply per-group. jude's Lance IVF/HNSW index makes that bucketing cheap
    — the semantic-dedup advantage over engines without a vector stack.
    """
    if embedding_column not in table.column_names:
        raise KeyError(f"embedding column {embedding_column!r} not in table")
    raw = table.column(embedding_column).to_pylist()
    embs = [list(v) if v is not None else [] for v in raw]
    n = len(embs)
    if n == 0:
        return table
    reps = _curate.semantic_clusters(embs, threshold)
    if keep_cluster:
        return table.append_column("sem_cluster", pa.array(reps, type=pa.int64()))
    keep = [i for i in range(n) if reps[i] == i]
    return table.take(pa.array(keep, type=pa.int64()))


# --- C4. language identification ---------------------------------------------


def detect_language(table: pa.Table, *, column: str = "text", lang_column: str = "lang",
                    conf_column: str = "lang_conf") -> pa.Table:
    """Add detected-language columns (heuristic, no model): ``lang_column`` (an
    ISO-639-1-ish code: en/zh/ja/ko/ru/ar/es/fr/de/... or 'und') and
    ``conf_column`` (confidence 0..1). Swap in fastText lid.176 via a UDF when
    precision matters; this is for coarse corpus routing/filtering."""
    verdicts = _curate.detect_language_batch(_col(table, column))
    langs = [v[0] for v in verdicts]
    confs = [v[1] for v in verdicts]
    out = table.append_column(lang_column, pa.array(langs, type=pa.string()))
    out = out.append_column(conf_column, pa.array(confs, type=pa.float64()))
    return out


@_observed("language_filter")
def language_filter(table: pa.Table, *, column: str = "text", keep: list[str] | str = "en",
                    min_confidence: float = 0.0) -> pa.Table:
    """Keep only rows whose detected language is in ``keep`` (a code or list) and
    whose confidence >= ``min_confidence``. For building mono-/multi-lingual
    subsets."""
    keep_set = {keep} if isinstance(keep, str) else set(keep)
    verdicts = _curate.detect_language_batch(_col(table, column))
    idx = [i for i, (lang, conf) in enumerate(verdicts) if lang in keep_set and conf >= min_confidence]
    return table.take(pa.array(idx, type=pa.int64()))


# --- C9. dataset blending / global shuffle -----------------------------------


def blend_datasets(tables: list[pa.Table], weights: list[float] | None = None,
                   *, total_rows: int | None = None, seed: int = 0) -> pa.Table:
    """Mix multiple datasets by weight (e.g. 50% web + 30% code + 20% books).

    Samples rows from each input table proportional to ``weights`` (normalized;
    default equal), producing ``total_rows`` rows (default: sum of inputs), then
    shuffles. Sampling is with-replacement when a source is smaller than its
    quota, else without. Deterministic given ``seed``. All tables must share a
    schema.
    """
    import numpy as np

    tables = [t for t in tables if t.num_rows > 0]
    if not tables:
        raise ValueError("blend_datasets: no non-empty tables")
    k = len(tables)
    w = weights if weights is not None else [1.0] * k
    if len(w) != k:
        raise ValueError("weights length must match number of tables")
    s = float(sum(w))
    w = [x / s for x in w]
    target = total_rows if total_rows is not None else sum(t.num_rows for t in tables)
    rng = np.random.default_rng(seed)
    parts = []
    for t, frac in zip(tables, w):
        quota = int(round(target * frac))
        if quota <= 0:
            continue
        replace = quota > t.num_rows
        idx = rng.choice(t.num_rows, size=quota, replace=replace)
        parts.append(t.take(pa.array(idx.tolist(), type=pa.int64())))
    if not parts:
        return tables[0].slice(0, 0)
    blended = pa.concat_tables(parts)
    # global shuffle of the blended result
    perm = rng.permutation(blended.num_rows)
    return blended.take(pa.array(perm.tolist(), type=pa.int64()))


def global_shuffle(table: pa.Table, *, seed: int = 0) -> pa.Table:
    """Randomly permute all rows (deterministic given ``seed``). Training data
    should be globally shuffled so batches aren't correlated by source order."""
    import numpy as np

    if table.num_rows == 0:
        return table
    perm = np.random.default_rng(seed).permutation(table.num_rows)
    return table.take(pa.array(perm.tolist(), type=pa.int64()))


# --- C10. PII detection & redaction ------------------------------------------


def redact_pii(table: pa.Table, *, column: str = "text", out_column: str | None = None) -> pa.Table:
    """Redact PII (email/url/ipv4/phone/ssn/credit-card) in a text column,
    replacing each match with a ``[KIND]`` tag. Writes back to ``column`` (or
    ``out_column`` if given). Compliance cleanup for training/RAG corpora.
    Coarse dependency-free scanners; swap Presidio via a UDF for higher recall."""
    redacted = _curate.redact_pii_batch(_col(table, column))
    dst = out_column or column
    arr = pa.array(redacted, type=pa.string())
    if dst in table.column_names:
        idx = table.column_names.index(dst)
        return table.set_column(idx, dst, arr)
    return table.append_column(dst, arr)


def detect_pii(table: pa.Table, *, column: str = "text", count_column: str = "pii_count") -> pa.Table:
    """Add a column counting detected PII spans per row (for auditing/filtering
    without modifying the text)."""
    counts = [len(_curate.detect_pii(t)) if t is not None else 0 for t in _col(table, column)]
    return table.append_column(count_column, pa.array(counts, type=pa.int64()))


# --- C11. task decontamination -----------------------------------------------


@_observed("decontaminate")
def decontaminate(
    table: pa.Table,
    benchmark_texts: list[str],
    *,
    column: str = "text",
    ngram: int = 8,
    threshold: float = 0.2,
    reason_column: str | None = None,
) -> pa.Table:
    """Remove training docs contaminated by benchmark/eval examples (C11): a doc
    whose n-gram overlap with the benchmark set exceeds ``threshold`` is dropped
    (prevents eval leakage / benchmark gaming). ``benchmark_texts`` are the eval
    questions/answers. If ``reason_column`` is set, keep all rows and annotate
    the contamination ratio instead of dropping."""
    bench = _curate.benchmark_ngrams(list(benchmark_texts), ngram)
    ratios = _curate.contamination_batch(_col(table, column), bench, ngram)
    if reason_column is not None:
        return table.append_column(reason_column, pa.array(ratios, type=pa.float64()))
    keep = [i for i, r in enumerate(ratios) if r <= threshold]
    return table.take(pa.array(keep, type=pa.int64()))


# --- cosmos pipeline stages --------------------------------------------------


def _arrow_stage_base():
    from jude.pipeline._multimodal import ArrowStage

    return ArrowStage


def ChunkStage(*, cpus: float = 1.0, **kwargs: Any):  # noqa: N802 — factory
    """cosmos Stage that chunks a text column (1 shard -> exploded shard)."""
    Base = _arrow_stage_base()

    class _ChunkStage(Base):
        def __init__(self):
            super().__init__(cpus=cpus)
            self._kw = kwargs

        def transform(self, table):
            return chunk_text(table, **self._kw)

    return _ChunkStage()


def QualityFilterStage(*, cpus: float = 1.0, **kwargs: Any):  # noqa: N802
    """cosmos Stage that drops (or annotates) low-quality rows."""
    Base = _arrow_stage_base()

    class _QualityFilterStage(Base):
        def __init__(self):
            super().__init__(cpus=cpus)
            self._kw = kwargs

        def transform(self, table):
            return quality_filter(table, **self._kw)

    return _QualityFilterStage()


def ContentHashStage(*, cpus: float = 1.0, **kwargs: Any):  # noqa: N802
    """cosmos Stage that adds a content-hash column (dedup key)."""
    Base = _arrow_stage_base()

    class _ContentHashStage(Base):
        def __init__(self):
            super().__init__(cpus=cpus)
            self._kw = kwargs

        def transform(self, table):
            return add_content_hash(table, **self._kw)

    return _ContentHashStage()
