"""Dense cumulant reference: linear step, kit parity, Price/Mehler series exactness, linearity of
the term registry, and a fast ReLU-step check against MC full tensors at n=8."""
import itertools

import numpy as np
import pytest
from scipy import integrate
from scipy.stats import norm

from arcmlp.reference import dense_cumulants as dc
from arcmlp.reference.mc import mc_truth


def _sym(T):
    k = T.ndim
    return sum(np.transpose(T, p) for p in itertools.permutations(range(k))) / np.math.factorial(k) if hasattr(np, "math") else \
        sum(np.transpose(T, p) for p in itertools.permutations(range(k))) / len(list(itertools.permutations(range(k))))


def test_linear_step_bruteforce():
    rng = np.random.default_rng(0)
    n = 4
    W = rng.normal(size=(n, n))
    K = {1: rng.normal(size=n), 2: _sym(rng.normal(size=(n, n))), 3: _sym(rng.normal(size=(n,) * 3)),
         4: _sym(rng.normal(size=(n,) * 4))}
    Z = dc.linear_step(K, W, 4)
    ref3 = np.zeros((n,) * 3); ref4 = np.zeros((n,) * 4)
    for i, j, k in itertools.product(range(n), repeat=3):
        ref3[i, j, k] = sum(W[i, a] * W[j, b] * W[k, c] * K[3][a, b, c] for a, b, c in itertools.product(range(n), repeat=3))
    for i, j, k, l in itertools.product(range(n), repeat=4):
        ref4[i, j, k, l] = np.einsum("a,b,c,d,abcd->", W[i], W[j], W[k], W[l], K[4])
    assert np.allclose(Z[1], W @ K[1]) and np.allclose(Z[2], W @ K[2] @ W.T)
    assert np.allclose(Z[3], ref3) and np.allclose(Z[4], ref4)


def test_K2_equals_kit_closure():
    rng = np.random.default_rng(1); n, L = 24, 5
    Ws = [rng.normal(0, np.sqrt(2 / n), (n, n)) for _ in range(L)]
    mus, K = dc.propagate(Ws, K=2)
    mu = np.zeros(n); S = np.eye(n)
    for W in Ws:
        m = W @ mu; C = W @ S @ W.T; s = np.sqrt(np.diag(C)); P, p = norm.cdf(m / s), norm.pdf(m / s)
        mu = m * P + s * p; e2 = (m ** 2 + s ** 2) * P + m * s * p; S = np.outer(P, P) * C; np.fill_diagonal(S, e2 - mu ** 2)
    assert np.abs(mus[-1] - mu).max() < 1e-12 and np.abs(K[2] - S).max() < 1e-12


def _exact_relu_cov(m1, m2, s1, s2, c):
    rho = c / (s1 * s2)
    def f(y, x):
        return max(x, 0) * max(y, 0) * np.exp(-0.5 * (((x - m1) / s1) ** 2 - 2 * rho * (x - m1) * (y - m2) / (s1 * s2)
                                                         + ((y - m2) / s2) ** 2) / (1 - rho ** 2)) \
               / (2 * np.pi * s1 * s2 * np.sqrt(1 - rho ** 2))
    e12 = integrate.dblquad(f, 0, m1 + 12 * s1, 0, m2 + 12 * s2, epsabs=1e-13, epsrel=1e-12)[0]
    psi = lambda m, s: m * norm.cdf(m / s) + s * norm.pdf(m / s)
    return e12 - psi(m1, s1) * psi(m2, s2)


@pytest.mark.parametrize("c", [0.05, 0.15])
def test_price_mehler_series_converges_to_exact(c):
    """Kt=2 diagrams are covariance multi-edges c^k/k! w_k w_k: raising the cap must converge to the
    exact bivariate ReLU covariance with error O(c^(cap+1))."""
    m = np.array([0.3, -0.4]); s = np.array([1.1, 0.8])
    Z = {1: m, 2: np.array([[s[0] ** 2, c], [c, s[1] ** 2]])}
    exact = _exact_relu_cov(m[0], m[1], s[0], s[1], c)
    errs = []
    for cap in (1, 2, 3, 5, 7):
        prop = dc.DenseProp(Kt=2, cap=cap)
        pK = prop.power_cumulants(Z)
        errs.append(abs(pK[(1, 1)][0, 1] - exact))
    assert errs[-1] < 1e-9 * max(1, abs(exact)) or errs[-1] < 1e-11
    assert errs[0] > errs[1] > errs[2] > errs[3]


def test_registry_linearity():
    """pK is linear in the term set: sum over single-term runs equals the all-terms run."""
    rng = np.random.default_rng(2); n = 5
    prop = dc.DenseProp(Kt=4)
    W = rng.normal(0, np.sqrt(2 / n), (n, n))
    K = {1: np.abs(rng.normal(size=n)) * 0.3, 2: np.eye(n) * 0.5 + 0.02 * _sym(rng.normal(size=(n, n))),
         3: 0.01 * _sym(rng.normal(size=(n,) * 3)), 4: 0.01 * _sym(rng.normal(size=(n,) * 4))}
    Z = dc.linear_step(K, W, 4)
    full = prop.power_cumulants(Z)
    acc = {p: np.zeros_like(v) for p, v in full.items()}
    for t in prop.registry:
        one = prop.power_cumulants(Z, terms=[t])
        acc[t.p] += one[t.p]
    for p in full:
        assert np.allclose(acc[p], full[p], atol=1e-13), p


def test_relu_step_vs_mc_n8():
    """Fast version of the validation: layer-0 (Gaussian input) post-ReLU cumulants at n=8 match MC
    to within truncation + MC error, and Kt=4 beats Kt=3 on kappa3 and kappa4."""
    rng = np.random.default_rng(0); n = 8
    Ws = [rng.normal(0, np.sqrt(2 / n), (n, n))]
    tr = mc_truth(Ws, 2_000_000, seed=7, full_tensors_upto=4, keep_cov=False)
    st = {Kt: dc.propagate(Ws, K=Kt, return_all=True)[2][0] for Kt in (3, 4)}
    rel = lambda k, Kt: np.sqrt(np.mean((st[Kt].get(k, 0) - tr[f"K{k}"][0]) ** 2)) / np.sqrt(np.mean(tr[f"K{k}"][0] ** 2))
    assert rel(2, 4) < 0.02 and rel(3, 4) < 0.05 and rel(4, 4) < 0.2
    assert rel(4, 4) < rel(4, 3)


def _poly_gauss_cumulants(a, b, c):
    """Cumulants 1..4 of z = a + b g + c (g^2 - 1), g ~ N(0,1)."""
    return a, b * b + 2 * c * c, 6 * b * b * c + 8 * c ** 3, 48 * b * b * c * c + 48 * c ** 4


def test_single_vertex_coefficients_nongaussian():
    """1-neuron check of the kappa3/kappa4 block coefficients (1/6, 1/24, 1/72, 1/144):
    with cap 4 (kappa3, kappa4, kappa3^2, kappa3*kappa4 blocks) the error in E[ReLU z] must shrink
    like c^3 (next terms kappa5 ~ c^3, kappa3^3 ~ c^3)."""
    from scipy.stats import norm as N
    a, b = 0.2, 1.0
    errs = []
    for c in (0.08, 0.04, 0.02):
        m, v, k3, k4 = _poly_gauss_cumulants(a, b, c)
        exact = integrate.quad(lambda g: max(a + b * g + c * (g * g - 1), 0) * N.pdf(g), -15, 15, points=[-1, 0, 1],
                               epsabs=1e-15, epsrel=1e-13, limit=500)[0]
        Z = {1: np.array([m]), 2: np.array([[v]]), 3: np.array([[[k3]]]), 4: np.array([[[[k4]]]])}
        prop = dc.DenseProp(Kt=4, cap=4)
        mu = prop.power_cumulants(Z)[(1,)][0]
        errs.append(abs(mu - exact))
    slope = np.log(errs[0] / errs[2]) / np.log(4.0)
    assert 2.5 < slope < 3.6, (errs, slope)
