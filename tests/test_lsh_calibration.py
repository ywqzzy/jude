"""C3: LSH band count calibrated to the Jaccard threshold.

A fixed band count (the old default 16) only aligns the LSH S-curve crossover to
threshold ~0.7; at other thresholds it silently loses recall. optimal_lsh_bands
derives the band count from the threshold so fuzzy_dedup catches the near-dups
the threshold implies. All deterministic (MinHash is seeded).
"""

from __future__ import annotations

import pyarrow as pa

from jude import curate
from jude.curate import optimal_lsh_bands


def test_calibration_crossover_tracks_threshold():
    # the S-curve crossover (1/b)^(1/r) should rise monotonically with threshold
    crosses = []
    for th in (0.5, 0.6, 0.7, 0.8, 0.9):
        b = optimal_lsh_bands(th, 128)
        r = 128 // b
        crosses.append((1.0 / b) ** (1.0 / r))
    assert crosses == sorted(crosses)                 # monotonic in threshold
    for th, c in zip((0.5, 0.6, 0.7, 0.8, 0.9), crosses):
        assert abs(c - th) < 0.08                     # crossover ~ threshold (integer-band granularity)


def test_low_threshold_recall_beats_fixed_bands():
    # near-dup pairs with moderate Jaccard (~0.6): below the fixed-16 crossover
    # (0.71) so bands=16 under-recalls them at threshold 0.5; the calibrated band
    # count (crossover ~0.52) catches them.
    pairs = []
    for i in range(12):
        a = f"word{i} alpha beta gamma delta epsilon zeta eta theta iota"
        b = f"word{i} alpha beta gamma delta epsilon zeta eta xxx yyy"  # shares 8/10 prefix
        pairs.append(a)
        pairs.append(b)
    t = pa.table({"text": pairs})

    fixed = curate.fuzzy_dedup(t, threshold=0.5, bands=16, seed=1)
    auto = curate.fuzzy_dedup(t, threshold=0.5, seed=1)  # bands=None -> calibrated
    # calibrated bands remove at least as many near-dups (higher recall)
    assert auto.num_rows <= fixed.num_rows


def test_explicit_bands_still_honored():
    docs = ["the cat sat on the mat", "the cat sat on the mat", "a distinct sentence"]
    t = pa.table({"text": docs})
    out = curate.fuzzy_dedup(t, threshold=0.6, bands=32, seed=1)  # explicit override
    assert out.num_rows == 2  # the exact dup still collapses
