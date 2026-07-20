"""L3.2: model-as-a-stage — batched model_score / model_filter, GPU-free.

The engine batches + preserves order; the "model" is any callable (mock here;
in production a CPU fastText/ONNX classifier or a remote vLLM/API endpoint). No
model dependency, no GPU — verifies the engine-side contract.
"""

from __future__ import annotations

import pyarrow as pa

from jude import model_stage as ms


def test_model_score_annotates_all():
    t = pa.table({"text": ["short", "a much longer document here", "mid one"]})
    # mock "quality" model: longer text scores higher
    out = ms.model_score(t, lambda batch: [len(x) for x in batch])
    assert out.num_rows == 3
    assert out.column("model_score").to_pylist() == [5.0, 27.0, 7.0]


def test_model_filter_threshold():
    t = pa.table({"text": ["a", "bbbbbb", "cc", "dddddddd"]})
    kept = ms.model_filter(t, lambda b: [len(x) for x in b], threshold=3, keep=">=")
    assert kept.column("text").to_pylist() == ["bbbbbb", "dddddddd"]


def test_model_filter_annotate_mode():
    t = pa.table({"text": ["x", "yy"]})
    out = ms.model_filter(t, lambda b: [len(x) for x in b], score_column="q")
    assert out.num_rows == 2                       # annotate keeps all
    assert out.column("q").to_pylist() == [1.0, 2.0]


def test_batching_preserves_order_and_covers_all():
    # 100 rows, small batches: order + coverage must hold across batch seams
    texts = [f"doc-{i}" for i in range(100)]
    seen_batches = []

    def score(batch):
        seen_batches.append(len(batch))
        return [float(len(x)) for x in batch]

    out = ms.model_score(pa.table({"text": texts}), score, batch_size=16)
    assert out.num_rows == 100
    assert out.column("model_score").to_pylist() == [float(len(x)) for x in texts]
    assert sum(seen_batches) == 100 and max(seen_batches) <= 16   # really batched


def test_score_fn_arity_mismatch_raises():
    import pytest
    t = pa.table({"text": ["a", "b"]})
    with pytest.raises(ValueError):
        ms.model_score(t, lambda b: [1.0])          # wrong number of scores


def test_model_scorer_stage_callable_with_lazy_setup():
    # setup_fn loads the "model" once (per worker in a real pipeline)
    loads = []

    def setup():
        loads.append(1)
        return lambda batch: [len(x) for x in batch]

    scorer = ms.ModelScorer(setup_fn=setup, out_column="q", batch_size=4)
    t = pa.table({"text": ["aa", "bbb"]})
    out = scorer(t)
    assert out.column("q").to_pylist() == [2.0, 3.0]
    scorer(t)                                        # second call reuses the model
    assert len(loads) == 1                           # setup ran once, not per batch
