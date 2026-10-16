"""Rank-at-accuracy of interval products of transfer maps (FINDINGS F4).

rank_at_eps(M, eps): smallest r with ||M - M_r||_F <= eps ||M||_F (M_r = best rank-r, Eckart-Young).
heuristic_rank(n, age, eps) = n (1 - eps^(2/age))  [hypothesis: Gaussian-product spectrum].
"""
from __future__ import annotations

import numpy as np


def rank_at_eps(M=None, eps=0.01, sv=None):
    if sv is None:
        sv = np.linalg.svd(M, compute_uv=False)
    tail = np.sqrt(np.cumsum(sv[::-1] ** 2)[::-1] / np.sum(sv ** 2))  # tail[r] = rel. error of rank r
    return int(np.searchsorted(-tail, -eps))


def heuristic_rank(n, age, eps):
    return int(n * (1 - eps ** (2 / age)))
