import math

import pytest

from lkv.stats import bootstrap_ci, mcnemar


def test_mcnemar_rejects_length_mismatch():
    """A length mismatch means the two runs were not over the same slice,
    which would silently invalidate the paired comparison."""
    with pytest.raises(ValueError):
        mcnemar([True, False, True], [True, False])


def test_mcnemar_counts_discordant_pairs_only():
    a = [True, True, False, False, True]
    b = [True, False, True, False, False]
    r = mcnemar(a, b)
    assert r["n10"] == 2   # a right, b wrong (indices 1, 4)
    assert r["n01"] == 1   # b right, a wrong (index 2)


def test_mcnemar_matches_hand_computed_chi2():
    """With continuity correction: (|n01-n10|-1)^2 / (n01+n10).
    n01=5, n10=20 -> (15-1)^2/25 = 7.84"""
    a = [True] * 20 + [False] * 5 + [True] * 10
    b = [False] * 20 + [True] * 5 + [True] * 10
    r = mcnemar(a, b)
    assert r["n10"] == 20 and r["n01"] == 5
    assert math.isclose(r["chi2"], 7.84, abs_tol=1e-6)
    assert r["p_exact"] < 0.01


def test_mcnemar_no_discordance_is_p_one():
    r = mcnemar([True, False], [True, False])
    assert r["n01"] == 0 and r["n10"] == 0
    assert r["chi2"] == 0.0 and r["p_exact"] == 1.0


def test_bootstrap_ci_brackets_the_mean_and_is_seeded():
    scores = [0.0] * 50 + [1.0] * 50
    lo, hi = bootstrap_ci(scores, n_boot=2000, seed=7)
    assert lo < 0.5 < hi
    assert bootstrap_ci(scores, n_boot=2000, seed=7) == (lo, hi)
