"""LLM data-curation operators: chunking, exact dedup, quality filtering.

Rust cores exercised via the Python facade + as cosmos pipeline stages
(local engine — the same Stage API cosmos runs)."""

from __future__ import annotations

import pyarrow as pa

from jude import curate


# --- C5. chunking ------------------------------------------------------------


def test_chunk_text_explodes_rows():
    t = pa.table({"id": [1, 2], "text": ["a" * 25, "short"]})
    out = curate.chunk_text(t, chunk_chars=10, overlap=0, recursive=False)
    # row 1: 25 chars / 10 -> 3 chunks; row 2: 1 chunk => 4 rows
    assert out.num_rows == 4
    assert set(out.column_names) >= {"id", "text", "chunk", "chunk_index"}
    # ids replicated
    assert out.column("id").to_pylist() == [1, 1, 1, 2]
    assert out.column("chunk_index").to_pylist() == [0, 1, 2, 0]
    # chunk lengths bounded
    assert all(len(c) <= 10 for c in out.column("chunk").to_pylist())


def test_chunk_text_recursive_respects_paragraphs():
    text = "para one.\n\npara two.\n\npara three."
    t = pa.table({"text": [text]})
    out = curate.chunk_text(t, chunk_chars=15, overlap=0, recursive=True)
    assert out.num_rows >= 2
    assert all(len(c) <= 25 for c in out.column("chunk").to_pylist())


def test_chunk_empty_text():
    t = pa.table({"text": [None, ""]})
    out = curate.chunk_text(t, chunk_chars=10)
    assert out.num_rows == 0
    assert "chunk" in out.column_names


# --- C2. exact dedup ---------------------------------------------------------


def test_add_content_hash_normalizes():
    t = pa.table({"text": ["Hello World", "  hello   world  ", "different"]})
    out = curate.add_content_hash(t)
    hs = out.column("content_hash").to_pylist()
    assert hs[0] == hs[1]  # case/whitespace-insensitive
    assert hs[0] != hs[2]


def test_exact_dedup_keeps_first():
    t = pa.table({"id": [1, 2, 3, 4], "text": ["A", "a", "B", "  a  "]})
    out = curate.exact_dedup(t)
    # "A"/"a"/"  a  " all normalize equal -> keep first (id 1); "B" -> id 3
    assert out.column("id").to_pylist() == [1, 3]


def test_exact_dedup_raw():
    t = pa.table({"text": ["A", "a"]})
    out = curate.exact_dedup(t, normalize=False)
    assert out.num_rows == 2  # case-sensitive: both kept


# --- C3. quality -------------------------------------------------------------


def test_quality_signals_columns():
    t = pa.table({"text": ["the quick brown fox jumps over the lazy dog"]})
    out = curate.quality_signals(t)
    assert "q_word_count" in out.column_names
    assert out.column("q_word_count").to_pylist()[0] == 9


def test_quality_filter_drops_bad():
    good = ("The history of natural language processing began in the nineteen fifties. "
            "In nineteen fifty Alan Turing published an article proposing what is now "
            "called the Turing test as a criterion of intelligence involving the automated "
            "interpretation and generation of natural human language across many domains.")
    t = pa.table({"id": [1, 2, 3], "text": [good, "too short", "!@#$ %^&* " * 20]})
    out = curate.quality_filter(t, min_words=30)
    assert out.column("id").to_pylist() == [1]


def test_quality_filter_annotate_mode():
    t = pa.table({"text": ["too short", "x"]})
    out = curate.quality_filter(t, reason_column="reject", min_words=5)
    assert out.num_rows == 2  # nothing dropped
    reasons = out.column("reject").to_pylist()
    assert all(r is not None for r in reasons)


# --- cosmos pipeline (local engine) ------------------------------------------


def test_curation_pipeline_local():
    import jude.pipeline as jp

    docs = [
        "The quick brown fox jumps over the lazy dog and then runs across the wide green field alone.",
        "The quick brown fox jumps over the lazy dog and then runs across the wide green field alone.",  # dup
        "bad",  # too short
    ]
    t = pa.table({"text": docs})
    # load(table) -> quality_filter -> content_hash : a multi-stage BATCH pipeline
    out = (
        jp.RelationPipeline.from_table(t, rows_per_shard=3, engine="local")
        .quality_filter(min_words=10)
        .content_hash()
        .run()
    )
    # both good docs survive quality; hash column present
    assert "content_hash" in out.column_names
    assert out.num_rows == 2  # the "bad" doc filtered
    # the two survivors are duplicates -> same hash
    hs = out.column("content_hash").to_pylist()
    assert hs[0] == hs[1]


def test_chunk_stage_in_pipeline():
    import jude.pipeline as jp

    t = pa.table({"id": [1], "text": ["a" * 50]})
    out = (
        jp.RelationPipeline.from_table(t, engine="local")
        .chunk(chunk_chars=10, overlap=0, recursive=False)
        .run()
    )
    assert out.num_rows == 5  # 50 / 10
    assert "chunk" in out.column_names


# --- C1. fuzzy dedup (MinHash + LSH) -----------------------------------------


def test_minhash_signature_shape():
    t = pa.table({"text": ["the quick brown fox", "another document here"]})
    sigs = curate.minhash_signatures(t, num_hashes=64)
    assert len(sigs) == 2
    assert all(len(s) == 64 for s in sigs)


def test_fuzzy_dedup_removes_near_duplicates():
    docs = [
        "the quick brown fox jumps over the lazy dog in the yard today with friends",
        "the quick brown fox jumps over the lazy dog in the yard today with buddies",  # near-dup
        "rust systems programming with zero cost abstractions and memory safety guarantees",
    ]
    t = pa.table({"id": [1, 2, 3], "text": docs})
    out = curate.fuzzy_dedup(t, threshold=0.6)
    kept = out.column("id").to_pylist()
    assert 1 in kept and 3 in kept  # one of the near-dup pair + the unrelated
    assert 2 not in kept
    assert out.num_rows == 2


def test_fuzzy_dedup_keeps_distinct():
    docs = [
        "apples and oranges grow on trees in the orchard every summer season",
        "rust systems programming with zero cost abstractions and safety",
        "the ocean tides rise and fall with the gravitational pull of the moon",
    ]
    t = pa.table({"id": [1, 2, 3], "text": docs})
    out = curate.fuzzy_dedup(t, threshold=0.7)
    assert out.num_rows == 3  # all distinct


def test_fuzzy_dedup_cluster_annotation():
    docs = [
        "the quick brown fox jumps over the lazy dog in the yard today with friends",
        "the quick brown fox jumps over the lazy dog in the yard today with buddies",
        "totally unrelated content about database systems and query planning here",
    ]
    t = pa.table({"text": docs})
    out = curate.fuzzy_dedup(t, threshold=0.6, keep_cluster=True)
    clusters = out.column("dup_cluster").to_pylist()
    assert clusters[0] == clusters[1]  # near-dups same cluster
    assert clusters[2] != clusters[0]


def test_fuzzy_dedup_empty():
    t = pa.table({"text": pa.array([], type=pa.string())})
    out = curate.fuzzy_dedup(t)
    assert out.num_rows == 0


# --- C7. semantic dedup (embedding + clustering) -----------------------------


def test_semantic_dedup_collapses_similar_embeddings():
    embs = [[1.0, 0.0, 0.0], [0.98, 0.02, 0.0], [0.0, 0.0, 1.0]]
    t = pa.table({"id": [1, 2, 3], "embedding": embs})
    out = curate.semantic_dedup(t, threshold=0.9)
    kept = out.column("id").to_pylist()
    assert kept == [1, 3]  # 2 is a semantic near-dup of 1


def test_semantic_dedup_all_distinct():
    embs = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
    t = pa.table({"id": [1, 2, 3], "embedding": embs})
    out = curate.semantic_dedup(t, threshold=0.9)
    assert out.num_rows == 3


def test_semantic_dedup_cluster_annotation():
    embs = [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]
    t = pa.table({"embedding": embs})
    out = curate.semantic_dedup(t, threshold=0.9, keep_cluster=True)
    clusters = out.column("sem_cluster").to_pylist()
    assert clusters[0] == clusters[1]
    assert clusters[2] != clusters[0]


# --- C4. language ID ---------------------------------------------------------


def test_detect_language_columns():
    t = pa.table({"text": [
        "the quick brown fox and the lazy dog is in the yard for a while now",
        "这是一段中文文本内容",
    ]})
    out = curate.detect_language(t)
    langs = out.column("lang").to_pylist()
    assert langs[0] == "en"
    assert langs[1] == "zh"
    assert "lang_conf" in out.column_names


def test_language_filter_keeps_english():
    t = pa.table({"id": [1, 2, 3], "text": [
        "the quick brown fox and the lazy dog is in the yard for a while now",
        "这是一段中文文本内容需要过滤",
        "le chat et le chien sont dans la maison des amis proches ici",
    ]})
    out = curate.language_filter(t, keep="en")
    assert out.column("id").to_pylist() == [1]
    multi = curate.language_filter(t, keep=["en", "fr"])
    assert set(multi.column("id").to_pylist()) == {1, 3}


# --- C9. blending / shuffle --------------------------------------------------


def test_blend_datasets_weighted_mix():
    from collections import Counter

    a = pa.table({"x": list(range(20)), "src": ["a"] * 20})
    b = pa.table({"x": list(range(100, 120)), "src": ["b"] * 20})
    out = curate.blend_datasets([a, b], [0.75, 0.25], total_rows=20, seed=1)
    assert out.num_rows == 20
    mix = Counter(out.column("src").to_pylist())
    assert mix["a"] == 15 and mix["b"] == 5  # 75/25 of 20


def test_global_shuffle_permutes():
    t = pa.table({"x": list(range(100))})
    out = curate.global_shuffle(t, seed=7)
    assert out.num_rows == 100
    assert sorted(out.column("x").to_pylist()) == list(range(100))  # same rows
    assert out.column("x").to_pylist() != list(range(100))  # but reordered


def test_blend_deterministic():
    a = pa.table({"x": list(range(30))})
    b = pa.table({"x": list(range(50, 80))})
    o1 = curate.blend_datasets([a, b], [0.5, 0.5], seed=3)
    o2 = curate.blend_datasets([a, b], [0.5, 0.5], seed=3)
    assert o1.column("x").to_pylist() == o2.column("x").to_pylist()



