"""W7: do cheap coherent even-order cumulant terms move the final-layer error? (research runner)

Written by an LLM agent (Claude Code). Theory: docs/notes/w7_coherent_even_orders.md.
Falsifier (written before running): W7 is NOT a win if, at n = 128, '+k4 coherent' (O(n^2)) recovers
< 50% of the final-layer MSE gap between K=3 and '+all kappa4 slices' (dense Kt=4 at the same cap),
averaged over seeds with a bootstrap CI excluding 50%; kappa6/kappa8 effects are reported either way.

Variants (final-layer mean of h_L):
  K3c2   dense Kt=3, cap 2 (the standard K=3)            K3c3   dense Kt=3, cap 3
  coh2   K=3 + coherent kappa4 (rank-one (2,2) + W^4 b diagonal), cap 2       coh3  same, cap 3
  ex22   K=3 + exact transported (2,2) of the diagonal source (one matmul), cap 2
  K4c2   dense Kt=4, cap 2 (only kappa4 slices (4,), (2,2) enter; exact)       [n <= 128]
  K4c3   dense Kt=4, cap 3 (all kappa4 slices incl. random-sign)               [n <= 128]
  coh2_k6, coh2_k6_k8, coh2_k3sq : coh2 + order-n^-2 / n^-3 coherent mean corrections
Each task (n, L, seed) writes data/w7/n{n}_L{L}_s{seed}.json (checkpoint); MC truth cached in data/mc.

    python experiments/w7_coherent_even.py --n 64 --L 8 16 --seeds 0-9 --N 2e7
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arcmlp.estimators.coherent_even import CoherentEvenProp  # noqa: E402
from arcmlp.reference import dense_cumulants as dc  # noqa: E402
from arcmlp.reference.mc import cached_truth, debiased_mse  # noqa: E402


def weights(n, L, seed):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, np.sqrt(2 / n), (n, n)) for _ in range(L)]


def variants(n):
    v = ["K3c2", "K3c3", "coh2", "coh3", "ex22", "ex4c2", "ex4c3", "exallc2", "exallc3", "coh2_k6", "coh2_k6_k8", "coh2_k3sq", "ex4c2_k6", "ex4c2_k6_k8"]
    if n <= 128:
        v += ["K4c2", "K4c3"]
    return v


def run_variant(Ws, name):
    t0 = time.time()
    extra = {}
    if name.startswith("K3c") or name.startswith("K4c"):
        Kt, cap = int(name[1]), int(name[3])
        mus = dc.propagate(Ws, K=Kt, cap=cap)[0]
    else:
        base = name.split("_")[0]
        cap = int(base[-1])
        mode = {"coh": "coherent", "ex22": "exact22", "ex4c": "ex4", "exallc": "exall"}[base[:-1] if base != "ex22" else "ex22"]
        pr = CoherentEvenProp(mode=mode, cap=cap, k6="k6" in name, k8="k8" in name, k3sq="k3sq" in name)
        mus = pr.run(Ws)[0]
        extra["diag"] = pr.diag
        if pr.k6 or pr.k8 or pr.k3sq:
            extra["dmu_rms_final"] = float(np.sqrt(np.mean(pr.dmus[-1] ** 2)))
    return mus[-1], time.time() - t0, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--L", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--N", type=float, default=4e6)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--mc-only", action="store_true")
    ap.add_argument("--outdir", default="data/w7")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    os.makedirs(a.outdir, exist_ok=True)
    for L in a.L:
        for sd in range(lo, hi + 1):
            path = f"{a.outdir}/n{a.n}_L{L}_s{sd}.json"
            res = json.load(open(path)) if os.path.exists(path) else {}
            Ws = weights(a.n, L, sd)
            tr = cached_truth(Ws, a.n, L, sd, int(a.N), mc_seed=70_000 + 100 * L + sd, moments="final")
            if a.mc_only:
                print(f"MC n={a.n} L={L} s={sd} floor={float(tr['floor']):.2e}", flush=True)
                continue
            res.update(dict(n=a.n, L=L, seed=sd, N=int(a.N), floor=float(tr["floor"])))
            for name in (a.only or variants(a.n)):
                if name in res.get("v", {}):
                    continue
                est, sec, extra = run_variant(Ws, name)
                dm = debiased_mse(est, tr)
                res.setdefault("v", {})[name] = dict(mse=dm["debiased"], se=dm["se"], raw=dm["raw"], sec=sec,
                                                     est=est.tolist(), **extra)
                json.dump(res, open(path, "w"))
                print(f"n={a.n} L={L} s={sd} {name:12s} mse={dm['debiased']:.3e} (se {dm['se']:.1e}) {sec:.0f}s", flush=True)


if __name__ == "__main__":
    main()
