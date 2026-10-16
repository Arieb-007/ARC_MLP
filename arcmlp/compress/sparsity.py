"""Skewness sparsity (Stage 2.5): drop columns of Q diag(a) with negligible |a_m| before (2,1)
products, under an explicit error budget.

keep_columns(a, q_norms, budget): keep the smallest set of columns such that the dropped part's
Frobenius mass sum_dropped (a_m * ||Q[:, m]||^2)^2 <= budget^2 * total. Returns a boolean mask.
Status: tooling; whether the dropped mass maps to final MSE linearly is a hypothesis (Stage 2.5).
"""
from __future__ import annotations

import numpy as np


def keep_columns(a, q_norms, budget=0.01):
    w = (np.abs(a) * q_norms ** 2) ** 2
    order = np.argsort(w)                  # ascending: cheapest to drop first
    cum = np.cumsum(w[order])
    n_drop = int(np.searchsorted(cum, budget ** 2 * w.sum(), side="right"))
    mask = np.ones(a.shape[0], dtype=bool)
    mask[order[:n_drop]] = False
    return mask
