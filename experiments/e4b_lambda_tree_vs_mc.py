"""E4b: tree-level analytic lambda_l vs Monte-Carlo-exact lambda_l (arrays saved by e4_lambda_mc.py).

Falsifier (stated before running): the newborn tree formula is not an adequate replacement for
V29's fitted lambda if it misses the MC-exact per-layer lambda by > 25% at more than a third of the
layers (then transported memory / pre-activation non-Gaussianity matter at leading order).

    python experiments/e4b_lambda_tree_vs_mc.py data/e4/n128_s0.npz data/e4/n128_s1.npz ...
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arcmlp.closure.kappa4_lambda import lambda_fast, lambda_tree, rho3  # noqa: E402

allrows = {}
for path in sys.argv[1:]:
    z = np.load(path)
    mu, C, PT, Ws = z["mu"], z["C"], z["PT"], z["Ws"]
    L, n = mu.shape
    cA = 6.0 / (n + 4)
    off = ~np.eye(n, dtype=bool)
    rows = []
    print(f"\n{path}: n={n} L={L}")
    print(" l | lam_MC    | tree(A,B,D) total     A        B        D   | fast total | tree/MC fast/MC")
    for l in range(L):
        if l == 0:
            m = np.zeros(n); Cp = Ws[0] @ Ws[0].T; Cprev = np.eye(n)
        else:
            m = Ws[l] @ mu[l - 1]; Cp = Ws[l] @ C[l - 1] @ Ws[l].T; Cprev = C[l - 1]
        X = C[l]
        x = X[off]
        lam_mc = cA * float(PT[l][off] @ x / (x @ x))
        tr = lambda_tree(m, Cp, X)
        fa = lambda_fast(m, Cp, X, rho3(Cprev))
        rows.append(dict(layer=l, lam_mc=lam_mc, tree=tr, fast=fa))
        print(f"{l:2d} | {lam_mc:.3e} | {tr['total']:.3e} {tr['A']:.2e} {tr['B']:+.2e} {tr['D']:.2e} | "
              f"{fa['total']:.3e} | {tr['total']/lam_mc:5.2f} {fa['total']/lam_mc:5.2f}")
    allrows[path] = rows
print(json.dumps(allrows))
