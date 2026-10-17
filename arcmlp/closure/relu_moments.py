"""Closed-form Gaussian ReLU moments, cumulants and Wick/Hermite coefficients (FINDINGS F1).

Written by an LLM agent (Claude Code); every formula is unit-tested against scipy quadrature
(tests/test_relu_moments.py) and, where cancellation matters, against mpmath at 60 digits.

X ~ N(m, s^2), t = m/s, Phi/phi = standard normal CDF/PDF, Z ~ N(0,1).

    E[ReLU(X)]      = psi(m, s) = m Phi(t) + s phi(t)
    E[ReLU(X) X^k]  (k <= 3) closed forms below
    E[ReLU(X)^p]    = s^p sum_j C(p,j) t^(p-j) I_j(t),  I_j(t) = int_{-t}^inf z^j phi(z) dz,
                      I_0 = Phi(t), I_1 = phi(t), I_j = (-t)^(j-1) phi(t) + (j-1) I_{j-2}
    Wick / Hermite coefficient (Stein: s^-k E[f(X) He_k((X-m)/s)] = E[f^(k)(X)]):
      w(k, p) := E[d^k ReLU(X)^p]
        k < p : (p)_k E[ReLU^(p-k)]                      (falling factorial; no boundary terms)
        k >= p: p! * E[d^(k-p+1) ReLU(X)],  with
                E[ReLU] = psi, E[ReLU'] = Phi(t), E[ReLU^(k)] = (-1)^k He_{k-2}(t) phi(t) / s^(k-1), k >= 2.
This matches the reference implementation's relu_wick_coef (ARC mlp_cumulant_propagation, MIT,
src/mlp_kprop/wick.py) case by case.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtr

SQRT2PI = math.sqrt(2.0 * math.pi)


def _phi(t):
    return np.exp(-0.5 * t * t) / SQRT2PI


def _Phi(t):
    return ndtr(t)


def hermite_e(k: int, t):
    """Probabilists' Hermite polynomial He_k(t) (recurrence He_{k+1} = t He_k - k He_{k-1})."""
    t = np.asarray(t, dtype=np.float64)
    h0, h1 = np.ones_like(t), t
    if k == 0:
        return h0
    for j in range(1, k):
        h0, h1 = h1, t * h1 - j * h0
    return h1


def psi(m, s):
    t = m / s
    return m * _Phi(t) + s * _phi(t)


def relu_x(m, s):
    """E[ReLU(X) X] (= E[ReLU(X)^2])."""
    t = m / s
    return (m * m + s * s) * _Phi(t) + m * s * _phi(t)


def relu_x2(m, s):
    """E[ReLU(X) X^2]."""
    t = m / s
    return (m ** 3 + 3 * m * s * s) * _Phi(t) + s * (m * m + 2 * s * s) * _phi(t)


def relu_x3(m, s):
    """E[ReLU(X) X^3] (= E[ReLU(X)^4])."""
    return relu_raw_moments(m, s, 4)[4]


def relu_second_moment(m, s):
    return relu_x(m, s)


def relu_raw_moments(m, s, pmax: int):
    """[E ReLU^0, ..., E ReLU^pmax] (list of arrays, float64)."""
    m = np.asarray(m, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    t = m / s
    P, p = _Phi(t), _phi(t)
    I = [P, p]
    for j in range(2, pmax + 1):
        I.append((-t) ** (j - 1) * p + (j - 1) * I[j - 2])
    out = [np.ones_like(t)]
    for q in range(1, pmax + 1):
        acc = 0.0
        for j in range(q + 1):
            acc = acc + math.comb(q, j) * t ** (q - j) * I[j]
        out.append(s ** q * acc)
    return out


def moments_to_cumulants(raw):
    """raw[0..K] -> cumulants kappa[1..K] (kappa[0] unused = 0), standard recursion
    kappa_n = mu_n - sum_{m=1}^{n-1} C(n-1, m-1) kappa_m mu_{n-m}."""
    K = len(raw) - 1
    kap = [0.0] * (K + 1)
    for nn in range(1, K + 1):
        acc = raw[nn]
        for mm in range(1, nn):
            acc = acc - math.comb(nn - 1, mm - 1) * kap[mm] * raw[nn - mm]
        kap[nn] = acc
    return kap


def relu_cumulants_all(m, s, kmax: int = 8):
    """kappa_1..kappa_kmax of ReLU(X) (list index = order). Cancellation-prone for t >> 1 at
    high order; use relu_cumulants_mp for t > ~4 when signs matter."""
    return moments_to_cumulants(relu_raw_moments(m, s, kmax))


def relu_central_moments(m, s, kmax: int = 8):
    raw = relu_raw_moments(m, s, kmax)
    mu = raw[1]
    out = [np.ones_like(mu), np.zeros_like(mu)]
    for q in range(2, kmax + 1):
        acc = 0.0
        for j in range(q + 1):
            acc = acc + math.comb(q, j) * raw[j] * (-mu) ** (q - j)
        out.append(acc)
    return out


def relu_cumulants(m, s, quad_n: int = 0):
    """kappa1..kappa4 (kept for backward compatibility with Stage-1 code)."""
    k = relu_cumulants_all(m, s, 4)
    return k[1], k[2], k[3], k[4]


def relu_raw_moments_std(k: int) -> float:
    if k == 0:
        return 1.0
    return 2 ** (k / 2 - 1) * math.gamma((k + 1) / 2) / math.sqrt(math.pi)


def relu_cumulants_std():
    k = relu_cumulants_all(0.0, 1.0, 4)
    return float(k[1]), float(k[2]), float(k[3]), float(k[4])


def wick(m, s, k: int, p: int = 1):
    """w(k, p) = E[d^k ReLU(X)^p], X ~ N(m, s^2) (= s^-k E[ReLU^p(X) He_k((X-m)/s)])."""
    m = np.asarray(m, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    if p == 0:
        return np.ones_like(m) if k == 0 else np.zeros_like(m)
    if k < p:
        ff = math.prod(range(p - k + 1, p + 1))
        return ff * relu_raw_moments(m, s, p - k)[p - k]
    if p > 1:
        return math.factorial(p) * wick(m, s, k - p + 1, 1)
    t = m / s
    if k == 0:
        return psi(m, s)
    if k == 1:
        return _Phi(t)
    return (-1) ** k * hermite_e(k - 2, t) * _phi(t) / s ** (k - 1)


def hermite_coef(m, s, k: int, p: int = 1):
    """hat-phi_k for ReLU^p: s^-k E[ReLU(X)^p He_k((X-m)/s)] (identical to wick(k, p))."""
    return wick(m, s, k, p)


def dpsi(m, s, k):
    """k-th derivative of psi in m (= wick(k, 1))."""
    return wick(m, s, k, 1)


# ---------------------------------------------------------------- high precision (tests)
def relu_cumulants_mp(t: float, kmax: int = 8, dps: int = 60):
    """Cumulants of ReLU(t + Z) (s = 1) with mpmath at `dps` digits (exact quadrature-free
    closed form through the I_j recurrence)."""
    import mpmath as mp
    mp.mp.dps = dps
    t = mp.mpf(t)
    P = mp.ncdf(t)
    p = mp.npdf(t)
    I = [P, p]
    for j in range(2, kmax + 1):
        I.append((-t) ** (j - 1) * p + (j - 1) * I[j - 2])
    raw = [mp.mpf(1)]
    for q in range(1, kmax + 1):
        raw.append(sum(mp.binomial(q, j) * t ** (q - j) * I[j] for j in range(q + 1)))
    kap = [mp.mpf(0)] * (kmax + 1)
    for nn in range(1, kmax + 1):
        acc = raw[nn]
        for mm in range(1, nn):
            acc -= mp.binomial(nn - 1, mm - 1) * kap[mm] * raw[nn - mm]
        kap[nn] = acc
    return kap
