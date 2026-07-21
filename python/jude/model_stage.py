"""jude.model_stage — run a user-supplied model as a batched curation stage.

jude does NOT implement inference; it *schedules* a stage that calls a model the
user brings. That model can be:
- a **CPU classifier** — fastText / ONNX / sklearn (FineWeb-edu quality filtering
  is literally a fastText model on CPU) — no GPU needed;
- a **remote endpoint** — a vLLM / TGI / OpenAI-compatible server (GPU elsewhere,
  or none);
- a **mock / any callable** — for tests and offline development.

The engine's job is what a data engine should own: **micro-batching**, a stable
row order, and (via the pipeline stage) resource scheduling + backpressure. The
model's job (scoring/labeling) stays the user's — and GPU is just a
``Resources(gpus=N)`` annotation, defaulting to 0 (CPU / remote).

    # quality filter with a CPU fastText model (no GPU):
    clf = load_fasttext("quality.bin")
    keep = model_filter(table, lambda texts: clf.scores(texts), threshold=0.5)

    # as a scaled pipeline stage (cosmos/Ray schedules it; gpus optional):
    pipe.map_batches(ModelScorer(score_fn, batch_size=256), gpus=0)
"""

from __future__ import annotations

from typing import Callable

import pyarrow as pa

Scorer = Callable[[list[str]], list[float]]  # batch text -> batch score


def _batched_scores(texts: list, score_fn: Scorer, batch_size: int) -> list[float]:
    """Apply a BATCH scoring fn over ``texts`` in chunks of ``batch_size``,
    preserving order. Batching is the engine's job so the model amortizes
    per-call overhead (tokenization / GPU launch / HTTP round-trip)."""
    out: list[float] = []
    n = len(texts)
    bs = max(1, int(batch_size))
    for i in range(0, n, bs):
        chunk = [t if t is not None else "" for t in texts[i : i + bs]]
        scores = list(score_fn(chunk))
        if len(scores) != len(chunk):
            raise ValueError(f"score_fn returned {len(scores)} scores for {len(chunk)} inputs")
        out.extend(float(s) for s in scores)
    return out


def model_score(
    table: pa.Table,
    score_fn: Scorer,
    *,
    column: str = "text",
    out_column: str = "model_score",
    batch_size: int = 256,
) -> pa.Table:
    """Annotate each row with a model score (keeps all rows). ``score_fn`` maps a
    batch of texts to a batch of floats — a CPU classifier, remote endpoint, or
    any callable. The engine batches; the model scores."""
    scores = _batched_scores(table.column(column).to_pylist(), score_fn, batch_size)
    return table.append_column(out_column, pa.array(scores, type=pa.float64()))


def model_filter(
    table: pa.Table,
    score_fn: Scorer,
    *,
    column: str = "text",
    threshold: float = 0.5,
    keep: str = ">=",
    batch_size: int = 256,
    score_column: str | None = None,
) -> pa.Table:
    """Keep rows whose model score passes ``threshold`` (``keep`` in
    ``>= > <= <``) — the model-based quality filter (FineWeb-edu / DCLM pattern).
    If ``score_column`` is set, keep ALL rows and annotate the score instead of
    dropping. ``score_fn`` is the user's model (CPU / remote / mock)."""
    import operator

    op = {">=": operator.ge, ">": operator.gt, "<=": operator.le, "<": operator.lt}[keep]
    scores = _batched_scores(table.column(column).to_pylist(), score_fn, batch_size)
    if score_column is not None:
        return table.append_column(score_column, pa.array(scores, type=pa.float64()))
    idx = [i for i, s in enumerate(scores) if op(s, threshold)]
    return table.take(pa.array(idx, type=pa.int64()))


class ModelScorer:
    """A picklable batched-scoring stage body for RelationPipeline.map_batches.

    Wraps a scoring backend as a stage callable ``(pa.Table) -> pa.Table`` that
    adds/updates ``out_column``. Use with ``pipe.map_batches(ModelScorer(...),
    gpus=N)`` — cosmos/Ray schedules it with the requested resources (gpus=0 =
    CPU/remote). ``setup_fn`` (optional) loads the model once per worker (so a
    heavy model isn't re-loaded per batch)."""

    def __init__(self, score_fn: Scorer | None = None, *, column: str = "text",
                 out_column: str = "model_score", batch_size: int = 256,
                 setup_fn: Callable[[], Scorer] | None = None):
        self._score_fn = score_fn
        self._setup_fn = setup_fn
        self.column = column
        self.out_column = out_column
        self.batch_size = batch_size

    def _fn(self) -> Scorer:
        if self._score_fn is None:
            if self._setup_fn is None:
                raise ValueError("ModelScorer needs a score_fn or setup_fn")
            self._score_fn = self._setup_fn()  # lazy per-worker model load
        return self._score_fn

    def __call__(self, table: "pa.Table") -> "pa.Table":
        return model_score(table, self._fn(), column=self.column,
                           out_column=self.out_column, batch_size=self.batch_size)


# --- concrete model backends (optional deps; the model is user-supplied) -----


def fasttext_scorer(model_path: str, *, label: str | None = None) -> Scorer:
    """A batch Scorer backed by a fastText model (CPU) — e.g. lid.176.bin for
    language ID (L1.3) or a FineWeb-edu-style quality classifier. Returns the
    probability of ``label`` (the top label's prob if ``label`` is None). Pair
    with ``model_filter`` for a model-based quality/language gate. Needs the
    ``fasttext`` package (lazy-imported)."""
    try:
        import fasttext
    except ImportError as e:  # pragma: no cover - exercised via the missing-dep test
        raise ImportError("fasttext_scorer needs `fasttext` (pip install fasttext-wheel)") from e

    model = fasttext.load_model(model_path)

    def score(batch: list[str]) -> list[float]:
        out: list[float] = []
        for text in batch:
            labels, probs = model.predict((text or "").replace("\n", " "), k=-1)
            if label is None:
                out.append(float(probs[0]) if len(probs) else 0.0)
            else:
                want = f"__label__{label}"
                out.append(float(next((p for l, p in zip(labels, probs) if l == want), 0.0)))
        return out

    return score


def kenlm_perplexity_scorer(model_path: str) -> Scorer:
    """A batch Scorer returning each doc's **perplexity** under a KenLM n-gram
    model (L3.3) — lower = more fluent/in-domain. Use with ``model_filter(...,
    keep="<=", threshold=...)`` to drop high-perplexity junk. Needs the ``kenlm``
    package (lazy-imported)."""
    try:
        import kenlm
    except ImportError as e:  # pragma: no cover
        raise ImportError("kenlm_perplexity_scorer needs `kenlm` (pip install kenlm)") from e

    model = kenlm.Model(model_path)

    def score(batch: list[str]) -> list[float]:
        return [float(model.perplexity(t or "")) for t in batch]

    return score
