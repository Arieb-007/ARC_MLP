"""Round-5 oracle: no projection == K4c3-no1111 exactly; full rank == no projection; spectra sane.
Written by an LLM agent."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
from arcmlp.estimators.a211_oracle import A211Oracle, jkl_spectrum, parse_a211  # noqa: E402
from arcmlp.reference import dense_cumulants as dc  # noqa: E402
from w5_selection import FEATURES, gates  # noqa: E402


def _ws(n=12, L=4, seed=5):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, np.sqrt(2 / n), (n, n)) for _ in range(L)]


def _no1111(Ws):
    pr = dc.DenseProp(Kt=4, cap=3)
    feats = set(FEATURES["dense"]) - {"K4(1,1,1,1)"}
    pr.active = [t for t in pr.active if gates(t) <= feats]
    return pr.run(Ws)[0]


def test_spec_equals_no1111_and_full_rank_is_identity():
    Ws = _ws()
    ref = _no1111(Ws)
    spec = A211Oracle(**parse_a211("a211:spec")).run(Ws)[0]
    full = A211Oracle(**parse_a211("a211:12")).run(Ws)[0]
    assert np.max(np.abs(spec - ref)) < 1e-12
    assert np.max(np.abs(full - ref)) < 1e-9
    low = A211Oracle(**parse_a211("a211:1")).run(Ws)[0]
    assert np.max(np.abs(low - ref)) > 1e-9           # the projection does something


def test_bulk_kept_equals_K4c3():
    Ws = _ws()
    ref = dc.DenseProp(Kt=4, cap=3).run(Ws)[0]
    got = A211Oracle(**parse_a211("a211:spec:bulk")).run(Ws)[0]
    assert np.max(np.abs(got - ref)) < 1e-12


def test_spectrum_diag():
    Ws = _ws(L=3)
    o = A211Oracle(**parse_a211("a211:spec"))
    o.run(Ws)
    assert [d["layer"] for d in o.diag] == [0, 1, 2]
    for d in o.diag[1:]:
        assert 0 <= d["f_ones"] <= 1.0 + 1e-9 and 0 < d["f1"] <= 1.0 + 1e-9 and d["f1"] <= d["f4"] + 1e-12
    rng = np.random.default_rng(0)
    A = np.einsum("j,kl->jkl", np.ones(6), rng.normal(size=(6, 6)))
    w, V, tot = jkl_spectrum(A, 6)
    assert abs(w[0] / tot - 1.0) < 1e-10               # a rank-1 (j | kl) tensor has one eigenvalue
