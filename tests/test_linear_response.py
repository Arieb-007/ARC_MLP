"""Round 4 linear-response chain: the term partition is exact (two-way == K4c3, background == K3c3),
read-outs off == background, schedules and the masked spectrum behave. Written by an LLM agent."""
import itertools

import numpy as np

from arcmlp.estimators.linear_response import LinearResponseProp, masked_spectrum, parse_lr, split_terms
from arcmlp.reference import dense_cumulants as dc


def _ws(n=10, L=4, seed=3):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, np.sqrt(2 / n), (n, n)) for _ in range(L)]


def test_partition():
    bg, ro, trans, birth = split_terms(4, 3)
    reg4 = dc.build_registry(4, 3)
    assert len(bg) + len(ro) + len(trans) + len(birth) == len(reg4)
    key = lambda t: (t.p, t.diagram, t.count)
    assert {key(t) for t in bg} == {key(t) for t in dc.build_registry(3, 3)}
    assert ro and trans and birth


def test_twoway_equals_K4c3_and_bg_equals_K3c3():
    Ws = _ws()
    ref4 = dc.DenseProp(Kt=4, cap=3).run(Ws)[0]
    ref3 = dc.DenseProp(Kt=3, cap=3).run(Ws)[0]
    tw = LinearResponseProp(**parse_lr("lr:twoway")).run(Ws)[0]
    bg = LinearResponseProp(**parse_lr("lr:bg")).run(Ws)[0]
    assert np.max(np.abs(tw - ref4)) < 1e-12
    assert np.max(np.abs(bg - ref3)) < 1e-14
    assert np.max(np.abs(ref4 - ref3)) > 1e-6          # the kappa4 effect is not trivially zero here


def test_readouts_off_is_background_and_oneway_is_close():
    Ws = _ws(n=32, L=4)
    ref3 = dc.DenseProp(Kt=3, cap=3).run(Ws)[0]
    ref4 = dc.DenseProp(Kt=4, cap=3).run(Ws)[0]
    off = LinearResponseProp(**parse_lr("lr:oneway:eps=0")).run(Ws)[0]
    assert np.max(np.abs(off - ref3)) < 1e-14
    ow = LinearResponseProp(**parse_lr("lr:oneway")).run(Ws)[0]
    # one-way differs from K4c3 only at second order: much closer to K4c3 than K3c3 is
    assert np.linalg.norm(ow[-1] - ref4[-1]) < 0.2 * np.linalg.norm(ref3[-1] - ref4[-1])


def test_schedules_and_kinds_run():
    Ws = _ws(L=5)
    for cfg in ("lr:oneway:ro=1", "lr:oneway:b=1", "lr:oneway:bf=2", "lr:oneway:kind=coh", "lr:oneway:kind=rs",
                "lr:oneway:ro=1:b=2"):
        mus = LinearResponseProp(**parse_lr(cfg)).run(Ws)[0]
        assert np.all(np.isfinite(mus)), cfg
    # kind=coh + kind=rs read-outs add up to all read-outs at first order (perturbative width: n = 32, L = 4)
    Ws = _ws(n=32, L=4)
    ref3 = LinearResponseProp(**parse_lr("lr:bg")).run(Ws)[0][-1]
    a = LinearResponseProp(**parse_lr("lr:oneway")).run(Ws)[0][-1] - ref3
    c = LinearResponseProp(**parse_lr("lr:oneway:kind=coh")).run(Ws)[0][-1] - ref3
    r = LinearResponseProp(**parse_lr("lr:oneway:kind=rs")).run(Ws)[0][-1] - ref3
    assert np.linalg.norm(a - c - r) < 0.2 * np.linalg.norm(a)


def test_masked_spectrum_norm_and_rank():
    rng = np.random.default_rng(0)
    n = 6
    T = sum(np.einsum("a,b,c,d->abcd", u, u, u, u) for u in rng.normal(size=(2, n)))
    s, tot = masked_spectrum(T, r=36, power_iters=3)
    ref = sum(T[a, b, c, d] ** 2 for a, b, c, d in itertools.product(range(n), repeat=4) if len({a, b, c, d}) == 4)
    assert abs(tot - ref) < 1e-9 * ref
    assert abs(np.sum(s ** 2) - tot) < 1e-8 * tot      # r = n^2 captures everything


def test_rank_diagnostics_record():
    Ws = _ws(n=8, L=5)
    pr = LinearResponseProp(**{**parse_lr("lr:oneway:rank"), "rank_layers": (2, 3), "rank_r": 16})
    pr.run(Ws)
    whats = sorted({(d["layer"], d["what"]) for d in pr.extra["rank"]})
    assert whats == [(2, "birth"), (2, "probe"), (3, "birth"), (3, "probe")]
    assert len(pr.extra["pnorm"]) == 5


def test_peak_memory_of_order4_chains():
    """The n^4 pre-activation must be built with at most two n^4 buffers alive (regression of 2026-10-20:
    passing the state tensor itself to mode4 kept it alive in the caller -> 3.5 n^4 peak, OOM at n = 256)."""
    import tracemalloc
    n = 72                                              # large enough that the n^3 arrays are a small fraction
    Ws = _ws(n=n, L=2)
    unit = n ** 4 * 4                                   # float32 at n > 64
    for make in (lambda: dc.DenseProp(Kt=4, cap=3), lambda: LinearResponseProp(**parse_lr("lr:oneway"))):
        tracemalloc.start()
        make().run(Ws)
        peak = tracemalloc.get_traced_memory()[1] / unit
        tracemalloc.stop()
        assert peak < 2.75, peak
