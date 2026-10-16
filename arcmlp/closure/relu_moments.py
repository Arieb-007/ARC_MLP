"""Closed-form Gaussian ReLU moments and derivatives (FINDINGS F1, status: proven).

X ~ N(m, s^2), t = m/s, Phi/phi = standard normal CDF/PDF.

    E[ReLU(X)]      = psi(m, s) = m Phi(t) + s phi(t)
    E[ReLU(X) X]    = (m^2 + s^2) Phi + m s phi
    E[ReLU(X) X^2]  = (m^3 + 3 m s^2) Phi + s (m^2 + 2 s^2) phi
    d_m psi = Phi,  d_m^2 psi = phi/s,  d_m^3 psi = -t phi/s^2,  d_m^4 psi = (t^2 - 1) phi/s^3

Edgeworth (cumulant) expansion of E[ReLU(Z)] for Z with mean m, variance s^2, cumulants k3, k4:
    E[ReLU(Z)] ~= psi + k3/6 d_m^3 psi + k4/24 d_m^4 psi + k3^2/72 d_m^6 psi + ...
All functions are xp-agnostic: pass numpy or flopscope.numpy as `xp` (default numpy) and a
`norm` object with .cdf/.pdf (default scipy.stats.norm).
"""
from __future__ import annotations

import math

import numpy as np

SQRT2PI = math.sqrt(2.0 * math.pi)


def _phi(t, xp=np):
    return xp.exp(-0.5 * t * t) / SQRT2PI


def _Phi(t, xp=np):
    # scipy-free CDF via erf (numpy has no erf; use math.erf vectorised for numpy inputs)
    if xp is np:
        from scipy.special import ndtr
        return ndtr(t)
    return xp.stats.norm.cdf(t)  # flopscope path (flops.stats)


def psi(m, s, xp=np):
    t = m / s
    return m * _Phi(t, xp) + s * _phi(t, xp)


def relu_x(m, s, xp=np):
    """E[ReLU(X) X]."""
    t = m / s
    return (m * m + s * s) * _Phi(t, xp) + m * s * _phi(t, xp)


def relu_x2(m, s, xp=np):
    """E[ReLU(X) X^2]."""
    t = m / s
    return (m ** 3 + 3 * m * s * s) * _Phi(t, xp) + s * (m * m + 2 * s * s) * _phi(t, xp)


def relu_second_moment(m, s, xp=np):
    """E[ReLU(X)^2] = E[ReLU(X) X]."""
    return relu_x(m, s, xp)


def dpsi(m, s, k, xp=np):
    """k-th derivative of psi in m, k = 0..6 (Hermite form: d^k psi = (-1)^k He_{k-2}(t) phi / s^{k-1})."""
    t = m / s
    if k == 0:
        return psi(m, s, xp)
    if k == 1:
        return _Phi(t, xp)
    he = {0: 1.0 + 0 * t, 1: t, 2: t * t - 1, 3: t ** 3 - 3 * t, 4: t ** 4 - 6 * t * t + 3}
    return (-1) ** k * he[k - 2] * _phi(t, xp) / s ** (k - 1)


def relu_raw_moments_std(k: int) -> float:
    """E[ReLU(Z)^k] for Z ~ N(0,1): (1/2) E|Z|^k = 2^{k/2-1} Gamma((k+1)/2)/sqrt(pi)."""
    if k == 0:
        return 1.0
    return 2 ** (k / 2 - 1) * math.gamma((k + 1) / 2) / math.sqrt(math.pi)


def relu_cumulants_std():
    """Cumulants k1..k4 of ReLU(Z), Z ~ N(0,1) (t = 0). k4 = 0.27971 (FINDINGS F3)."""
    m1, m2, m3, m4 = (relu_raw_moments_std(k) for k in (1, 2, 3, 4))
    k1 = m1
    k2 = m2 - m1 ** 2
    k3 = m3 - 3 * m2 * m1 + 2 * m1 ** 3
    k4 = m4 - 4 * m3 * m1 - 3 * m2 ** 2 + 12 * m2 * m1 ** 2 - 6 * m1 ** 4
    return k1, k2, k3, k4


def relu_cumulants(m, s, quad_n: int = 200):
    """Cumulants k1..k4 of ReLU(X), X ~ N(m, s^2), elementwise; float64 numpy, closed form for
    k1, k2 and Gauss-Hermite-free exact formulas for the raw moments up to order 4."""
    m = np.asarray(m, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    t = m / s
    P, p = _Phi(t), _phi(t)
    # raw moments E[ReLU(X)^k] = s^k * E[(t + Z)^k 1{Z > -t}]
    # I_j(t) = int_{-t}^inf z^j phi(z) dz:  I0 = Phi(t), I1 = phi(t), I_j = (-t)^{j-1} phi(t) + (j-1) I_{j-2}
    I = [P, p]
    for j in range(2, 5):
        I.append((-t) ** (j - 1) * p + (j - 1) * I[j - 2])
    raw = []
    for k in range(1, 5):
        acc = 0.0
        for j in range(k + 1):
            acc = acc + math.comb(k, j) * t ** (k - j) * I[j]
        raw.append(s ** k * acc)
    m1, m2, m3, m4 = raw
    k1 = m1
    k2 = m2 - m1 ** 2
    k3 = m3 - 3 * m2 * m1 + 2 * m1 ** 3
    k4 = m4 - 4 * m3 * m1 - 3 * m2 ** 2 + 12 * m2 * m1 ** 2 - 6 * m1 ** 4
    return k1, k2, k3, k4
