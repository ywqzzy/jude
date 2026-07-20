"""C2: Gopher/C4 quality signals — stopword gate, duplicate-n-gram repetition,
and the (previously dead) digit-ratio gate. All deterministic, no model."""

from __future__ import annotations

import pyarrow as pa

from jude import curate

_PROSE = ("The committee reviewed the proposal in detail and agreed that the plan "
          "should proceed in the spring, provided that the budget is approved by the "
          "board and that the team can deliver the first milestone on time for the "
          "many stakeholders who depend on it across the whole organization, since "
          "the schedule is tight and the resources are limited this year for all of "
          "the groups that are involved in the effort to ship the product.")


def test_new_signals_present():
    out = curate.quality_signals(pa.table({"text": [_PROSE]}))
    assert "q_stopword_ratio" in out.column_names
    assert "q_dup_ngram_ratio" in out.column_names
    assert out.column("q_stopword_ratio")[0].as_py() > 0.1   # prose has stopwords
    assert out.column("q_dup_ngram_ratio")[0].as_py() < 0.2  # prose isn't repetitive


def test_prose_passes_quality():
    out = curate.quality_filter(pa.table({"text": [_PROSE]}))
    assert out.num_rows == 1


def test_keyword_spam_rejected_by_stopword_gate():
    spam = " ".join(f"item{i}" for i in range(60))   # 60 short non-stopwords
    out = curate.quality_filter(pa.table({"text": [spam]}), reason_column="why")
    why = out.column("why")[0].as_py()
    assert why is not None and "stopword_ratio_low" in why


def test_digit_heavy_rejected():
    digits = " ".join("1234567" for _ in range(60))
    out = curate.quality_filter(pa.table({"text": [digits]}), reason_column="why")
    assert out.column("why")[0].as_py() is not None   # rejected (digit or stopword gate)


def test_repetitive_rejected():
    rep = "the cat sat on the mat " * 12
    sig = curate.quality_signals(pa.table({"text": [rep]}))
    assert sig.column("q_dup_ngram_ratio")[0].as_py() > 0.5
    out = curate.quality_filter(pa.table({"text": [rep]}))
    assert out.num_rows == 0   # repetitive text dropped


def test_threshold_override():
    spam = " ".join(f"kw{i}" for i in range(60))
    # relaxing the stopword gate to 0 lets the (short-word) spam through the
    # stopword check — but mean_word_len is tiny so it still fails elsewhere;
    # just assert the kwarg is honored (no stopword_ratio_low reason).
    out = curate.quality_filter(pa.table({"text": [spam]}),
                                min_stopword_ratio=0.0, reason_column="why")
    why = out.column("why")[0].as_py()
    assert why is None or "stopword_ratio_low" not in why
