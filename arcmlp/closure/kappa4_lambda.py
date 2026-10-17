"""Analytic (tree-level) kappa4 regeneration coefficient lambda_l (Stage 1; numpy reference).

Object (504aldo lean_k3_aug.py:1262, the r=1 harmonic projection): the off-diagonal kappa4 core
    G_ij = c_A * sum_k kappa4(h_i, h_j, h_k, h_k),   c_A = 6/(n+4),   i != j,
of the post-ReLU activations h = ReLU(z). V29 regenerates G_off = lambda_l * C_post,off with a fitted
lambda_l. Here lambda_l is computed from the pre-activation statistics (m, s, C_pre) alone.

Derivation [proven at leading order in the pre-activation correlations c_ij = O(n^{-1/2}),
z Gaussian]: joint cumulants of functions of weakly correlated Gaussians are sums over connected
trees, each vertex a contributing E[f_a^(deg a)(z_a)]. With f = ReLU, E f' = Phi, E f'' = phi/s,
E (f^2)' = 2 mu, E (f^2)'' = 2 Phi, and kappa(X,Y,Z,Z) = kappa(X,Y,Z^2) - 2 E[Z] kappa3(X,Y,Z)
- 2 kappa(X,Z) kappa(Y,Z) (Leonov-Shiryaev), for k not in {i, j}:
    kappa4_ijkk = c_ik c_jk Phi_i Phi_j gamma_k + c_ij c_ik (phi_i/s_i) Phi_j beta_k
                  + c_ij c_jk Phi_i (phi_j/s_j) beta_k + O(c^3),
    gamma_k = 2[Phi_k (1 - Phi_k) - mu_k phi_k / s_k],   beta_k = 2 mu_k (1 - Phi_k);
and for k in {i, j} (the (3,1) slice), to first order in c_ij,
    kappa4_iiij = c_ij Phi_j d_m kappa3(h_i),   d_m kappa3 = 3 (1 - Phi)(E[h^2] - 2 E[h]^2).
Summing over k gives three terms:
    A_ij = Phi_i Phi_j (C~ Gamma C~)_ij   (coherent k-sum)       C~ = C_pre with zero diagonal
    B_ij = c_ij [(phi_i/s_i) Phi_j (C~ beta)_i + Phi_i (phi_j/s_j) (C~ beta)_j]
    D_ij = c_ij [Phi_j k3'_i + Phi_i k3'_j]                     ((3,1) slice; dominant)
and lambda = c_A <A + B + D, X>_off / <X, X>_off with X = C_post,off (least squares, as V29's fit).
Neglected: O(c^3) loops (O(n^{-1}) vs O(n^{-1/2})), non-Gaussianity of z (kappa3/kappa4 of the
pre-activation: the transported memory), and the cI * trace term (diagonal only).

`lambda_tree(...)` computes A exactly (dense, O(n^3)); `lambda_fast(...)` replaces A by the
random-W mean field  A ~ 2 gamma_bar rho3 * Phi_i Phi_j c_ij,  rho3 = tr(C_prev^3)/tr(C_prev^2),
which is O(n^2) given rho3 (C_prev = post-ReLU covariance of the previous layer; I at layer 0).
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr


def relu_stats(m, s):
    t = m / s
    Phi = ndtr(t)
    phi = np.exp(-0.5 * t * t) / np.sqrt(2 * np.pi)
    E1 = m * Phi + s * phi
    E2 = (m * m + s * s) * Phi + m * s * phi
    k3p = 3.0 * (1.0 - Phi) * (E2 - 2.0 * E1 * E1)
    gamma = 2.0 * (Phi * (1.0 - Phi) - E1 * phi / s)
    beta = 2.0 * E1 * (1.0 - Phi)
    return dict(Phi=Phi, phi=phi, E1=E1, E2=E2, k3p=k3p, gamma=gamma, beta=beta)


def _terms(m, C_pre, X):
    n = m.size
    s = np.sqrt(np.diag(C_pre))
    st = relu_stats(m, s)
    Ct = C_pre - np.diag(np.diag(C_pre))
    Phi, phi = st["Phi"], st["phi"]
    bt = Ct @ st["beta"]
    D = Ct * (np.outer(st["k3p"], Phi) + np.outer(Phi, st["k3p"]))
    Bm = Ct * (np.outer(phi / s * bt, Phi) + np.outer(Phi, phi / s * bt))
    return st, Ct, D, Bm


def lambda_tree(m, C_pre, X, n_cA=None):
    """Exact tree-level lambda (dense A term). m, C_pre: pre-activation mean / covariance;
    X: post-ReLU covariance (only the off-diagonal is used). Returns dict of term lambdas."""
    n = m.size
    cA = 6.0 / ((n if n_cA is None else n_cA) + 4)
    st, Ct, D, Bm = _terms(m, C_pre, X)
    Phi = st["Phi"]
    A = np.outer(Phi, Phi) * (Ct @ (st["gamma"][:, None] * Ct))
    off = ~np.eye(n, dtype=bool)
    x = X[off]; xx = x @ x
    lam = {k: cA * float(M[off] @ x / xx) for k, M in (("A", A), ("B", Bm), ("D", D))}
    lam["total"] = lam["A"] + lam["B"] + lam["D"]
    return lam


def lambda_fast(m, C_pre, X, rho3, n_cA=None):
    """O(n^2) version: A term from the mean field 2 gamma_bar rho3 Phi_i Phi_j c_ij."""
    n = m.size
    cA = 6.0 / ((n if n_cA is None else n_cA) + 4)
    st, Ct, D, Bm = _terms(m, C_pre, X)
    Phi = st["Phi"]
    off = ~np.eye(n, dtype=bool)
    x = X[off]; xx = x @ x
    Aproxy = (np.outer(Phi, Phi) * Ct)[off] @ x / xx
    lam = dict(A=cA * 2.0 * float(np.mean(st["gamma"])) * rho3 * float(Aproxy),
               B=cA * float(Bm[off] @ x / xx), D=cA * float(D[off] @ x / xx))
    lam["total"] = lam["A"] + lam["B"] + lam["D"]
    return lam


def rho3(C_prev):
    C2 = C_prev @ C_prev
    return float(np.sum(C2 * C_prev) / np.sum(C_prev * C_prev))
