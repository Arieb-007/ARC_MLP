"""Closed forms vs scipy quadrature (1e-12) and mpmath; positivity of kappa4, kappa6 of ReLU(X)."""
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from arcmlp.closure import relu_moments as rm

GRID = [(0.0, 1.0), (0.7, 2.0), (-1.3, 0.5), (2.5, 1.2), (-0.4, 0.8), (1.0, 0.3), (-2.0, 1.5)]


def _quad(f, m, s):
    g = lambda x: f(x) * norm.pdf(x, m, s)
    a, b = m - 40 * s, m + 40 * s
    pts = [0.0] if a < 0 < b else None
    v1 = quad(g, a, 0.0, epsabs=1e-15, epsrel=1e-14, limit=400)[0] if a < 0 else 0.0
    v2 = quad(g, max(a, 0.0), b, epsabs=1e-15, epsrel=1e-14, limit=400)[0]
    return v1 + v2


def _close(a, b, tol=1e-12):
    return abs(a - b) <= tol * max(1.0, abs(b))


@pytest.mark.parametrize("m,s", GRID)
def test_relu_times_powers(m, s):
    for k, f in enumerate([rm.psi, rm.relu_x, rm.relu_x2, rm.relu_x3]):
        q = _quad(lambda x: max(x, 0.0) * x ** k, m, s)
        assert _close(float(f(m, s)), q), (k, float(f(m, s)), q)


@pytest.mark.parametrize("m,s", GRID)
def test_raw_moments(m, s):
    raw = rm.relu_raw_moments(m, s, 8)
    for p in range(9):
        q = _quad(lambda x: max(x, 0.0) ** p, m, s) if p > 0 else 1.0
        assert _close(float(raw[p]), q), (p, float(raw[p]), q)


@pytest.mark.parametrize("m,s", GRID)
def test_wick_hermite(m, s):
    for p in range(1, 5):
        for k in range(0, 9):
            w = float(rm.wick(m, s, k, p))
            q = _quad(lambda x: max(x, 0.0) ** p * float(rm.hermite_e(k, (x - m) / s)), m, s) / s ** k
            assert _close(w, q, 1e-11), (k, p, w, q)


@pytest.mark.parametrize("m,s", GRID)
def test_cumulants_vs_mpmath(m, s):
    kap = rm.relu_cumulants_all(m, s, 8)
    kmp = rm.relu_cumulants_mp(m / s, 8)
    for q in range(1, 9):
        ref = float(kmp[q]) * s ** q
        assert abs(float(kap[q]) - ref) <= 1e-9 * max(1.0, abs(ref)), (q, float(kap[q]), ref)


def test_central_moments_consistent():
    m, s = 0.3, 1.1
    cm = rm.relu_central_moments(m, s, 4)
    kap = rm.relu_cumulants_all(m, s, 4)
    assert _close(float(cm[2]), float(kap[2]))
    assert _close(float(cm[3]), float(kap[3]))
    assert _close(float(cm[4]) - 3 * float(cm[2]) ** 2, float(kap[4]))


def test_kappa4_kappa6_sign_structure():
    """Measured 2026-10-17 (61 digits, confirmed by 2e7-sample MC): kappa4(ReLU(t+Z)) >= 0 iff
    t <= 0.7556; kappa6 changes sign at t = -0.316, 0.961, 2.329. The brief's positivity
    hypothesis is false; this test pins the measured sign structure on t in [-8, 8]."""
    ts = np.linspace(-8, 8, 161)
    k4 = np.array([float(rm.relu_cumulants_mp(t, 6)[4]) for t in ts])
    k6 = np.array([float(rm.relu_cumulants_mp(t, 6)[6]) for t in ts])
    assert np.all(k4[ts <= 0.75] >= 0) and np.all(k4[ts >= 0.76] < 0)
    sgn = np.sign(k6)
    flips = ts[1:][sgn[1:] != sgn[:-1]]
    assert len(flips) == 3 and abs(flips[0] + 0.3) < 0.11 and abs(flips[1] - 1.0) < 0.11 and abs(flips[2] - 2.4) < 0.11
