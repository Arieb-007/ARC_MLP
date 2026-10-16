"""K=2 Gaussian closure ("covariance propagation", the starter-kit baseline), xp-agnostic.

Per layer: m = W mu, C = W S W^T, s = sqrt(diag C), t = m/s;
mu' = psi(m, s); E[h^2] = (m^2 + s^2) Phi + m s phi;
S'_off = Phi Phi^T * C_off (Price, first order); diag S' = E[h^2] - mu'^2.
Cost: 2 dense units per layer (W S, (W S) W^T) at n x n. Status: reference implementation of a
known method; matches experiments/*.py.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr


def cov_prop(weights, mu0=None, S0=None, exact_c2: bool = False):
    """weights: list of (n, n) float arrays. Returns (mus (L, n), S_last).
    exact_c2: add the second-order Price term phi_i phi_j/(2 s_i s_j) C_ij^2 to the off-diagonal."""
    n = weights[0].shape[0]
    mu = np.zeros(n) if mu0 is None else mu0
    S = np.eye(n) if S0 is None else S0
    mus = []
    for W in weights:
        m = W @ mu
        C = W @ S @ W.T
        s = np.sqrt(np.diag(C))
        t = m / s
        P = ndtr(t)
        p = np.exp(-0.5 * t * t) / np.sqrt(2 * np.pi)
        mu = m * P + s * p
        e2 = (m * m + s * s) * P + m * s * p
        S = np.outer(P, P) * C
        if exact_c2:
            q = p / s
            S += 0.5 * np.outer(q, q) * C * C
        np.fill_diagonal(S, e2 - mu * mu)
        mus.append(mu)
    return np.array(mus), S
