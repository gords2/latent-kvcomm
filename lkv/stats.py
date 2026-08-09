"""Paired tests. Two runs on the same seeded slice are paired, so McNemar is
the correct instrument -- an unpaired z-test on the same data needs a much
larger difference to reach the same significance."""
from __future__ import annotations

import random

from scipy.stats import binomtest


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    if len(a) != len(b):
        raise ValueError(f"unpaired inputs: {len(a)} vs {len(b)}")
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if y and not x)
    total = n01 + n10
    if total == 0:
        return {"n01": 0, "n10": 0, "chi2": 0.0, "p_exact": 1.0}
    chi2 = (abs(n01 - n10) - 1) ** 2 / total
    p = binomtest(n10, total, 0.5).pvalue
    return {"n01": n01, "n10": n10, "chi2": float(chi2), "p_exact": float(p)}


def bootstrap_ci(scores: list[float], n_boot: int = 10000, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float]:
    if not scores:
        raise ValueError("empty scores")
    rng = random.Random(seed)
    n = len(scores)
    means = []
    for _ in range(n_boot):
        means.append(sum(scores[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    tail = (1.0 - level) / 2.0
    return means[int(tail * n_boot)], means[min(n_boot - 1, int((1 - tail) * n_boot))]
