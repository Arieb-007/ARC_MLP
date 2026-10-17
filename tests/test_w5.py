"""W5 feature switches reproduce the named truncations (LLM-written test)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from w5_selection import FEATURES, K3F, make_prop, weights  # noqa: E402
from arcmlp.estimators.coherent_even import CoherentEvenProp  # noqa: E402
from arcmlp.reference import dense_cumulants as dc  # noqa: E402

Ws = weights(12, 3, 5)


def mu(pr):
    return pr.run(Ws)[0][-1]


def test_dense_switches_match_named_truncations():
    F = FEATURES["dense"]
    np.testing.assert_allclose(mu(make_prop("dense", [])), dc.propagate(Ws, K=2)[0][-1], rtol=0, atol=1e-13)
    np.testing.assert_allclose(mu(make_prop("dense", F)), dc.propagate(Ws, K=4, cap=3)[0][-1], rtol=0, atol=1e-13)
    np.testing.assert_allclose(mu(make_prop("dense", set(F) - {"cap3"})), dc.propagate(Ws, K=4, cap=2)[0][-1], rtol=0, atol=1e-13)
    np.testing.assert_allclose(mu(make_prop("dense", ["Cpair", "cap3"] + K3F)), dc.propagate(Ws, K=3, cap=3)[0][-1], rtol=0, atol=1e-13)
    np.testing.assert_allclose(mu(make_prop("dense", ["Cpair"] + K3F)), dc.propagate(Ws, K=3, cap=2)[0][-1], rtol=0, atol=1e-13)


def test_light_switches_match_named_truncations():
    F = FEATURES["light"]
    np.testing.assert_allclose(mu(make_prop("light", F)), mu(CoherentEvenProp("coherent", cap=3)), rtol=0, atol=1e-13)
    np.testing.assert_allclose(mu(make_prop("light", set(F) - {"cap3"})), mu(CoherentEvenProp("coherent", cap=2)), rtol=0, atol=1e-13)
    np.testing.assert_allclose(mu(make_prop("light", [])), dc.propagate(Ws, K=2)[0][-1], rtol=0, atol=1e-13)
    np.testing.assert_allclose(mu(make_prop("light", ["Cpair"] + K3F)), dc.propagate(Ws, K=3, cap=2)[0][-1], rtol=0, atol=1e-12)
