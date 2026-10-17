"""W5: measured, cost-aware selection of diagram terms (research runner). Written by an LLM agent.

Terms = the Kt=4, cap=3 registry of arcmlp/reference/dense_cumulants.py (95 terms). Groups (single
terms are mostly below the MDE, and 95 x 2 runs per network does not fit the CPU budget):
  backbone (always on): empty diagrams (Gaussian moments) and the single covariance edge (= K=2)
  use:<inputs>          terms writing orders <= 2 (mean, variance, covariance) from that input set
  prod<d>:<C|K3|K4>     terms writing order-d (3 or 4) state, by highest input cumulant family
Measurements per network (final-layer MSE vs cached MC truth, debiased):
  full, base (backbone only), leave-one-out (full - group), add-one-in (base + group).
Tiers: 'dense'  = DenseProp(Kt=4, cap=3) at n <= 64 (all groups);
       'light'  = CoherentEvenProp(mode='coherent', cap=3) (K=3 dense + coherent kappa4; groups that
                  read kappa4 (3,1)/(2,1,1)/(1,1,1,1) or write order-4 multi-neuron state are absent).
Falsifier (written before running): the selection does not transfer if the Spearman rank
correlation of leave-one-out deltas between the two widths is < 0.7.

    python experiments/w5_selection.py --tier dense --n 64 --L 16 --seeds 0-9
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


def group_of(t):
    if t.cost == 0:
        return None
    if t.p == (1, 1) and t.diagram == ((1, 1),):
        return None
    order = sum(t.p)
    ins = sorted(set(t.inputs))
    if order <= 2:
        return "use:" + "+".join(ins)
    kmax = max([int(x[1]) for x in ins if x.startswith("K")] or [2])
    return f"prod{order}:" + {2: "C", 3: "K3", 4: "K4"}[kmax]


def make_prop(tier, active_ids=None):
    if tier == "dense":
        pr = dc.DenseProp(Kt=4, cap=3)
    else:
        pr = CoherentEvenProp(mode="coherent", cap=3)
    if active_ids is not None:
        pr.active = [t for t in pr.active if t.tid in active_ids]
    return pr


def groups_for(tier):
    pr = make_prop(tier)
    gs = {}
    for t in pr.active:
        g = group_of(t)
        if g is not None:
            gs.setdefault(g, []).append(t.tid)
    return gs, [t.tid for t in pr.active if group_of(t) is None], [t.tid for t in pr.active]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["dense", "light"], required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--N", type=float, default=4e6)
    ap.add_argument("--outdir", default="data/w5")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    os.makedirs(a.outdir, exist_ok=True)
    gs, backbone, full = groups_for(a.tier)
    for sd in range(lo, hi + 1):
        path = f"{a.outdir}/{a.tier}_n{a.n}_L{a.L}_s{sd}.json"
        res = json.load(open(path)) if os.path.exists(path) else dict(tier=a.tier, n=a.n, L=a.L, seed=sd, runs={})
        rng = np.random.default_rng(sd)
        Ws = [rng.normal(0, np.sqrt(2 / a.n), (a.n, a.n)) for _ in range(a.L)]
        tr = cached_truth(Ws, a.n, a.L, sd, int(a.N), mc_seed=70_000 + 100 * a.L + sd, moments="final")
        res["floor"] = float(tr["floor"]); res["N"] = int(a.N)
        jobs = [("full", set(full)), ("base", set(backbone))]
        for g, ids in sorted(gs.items()):
            jobs.append(("loo:" + g, set(full) - set(ids)))
            jobs.append(("add:" + g, set(backbone) | set(ids)))
        for name, ids in jobs:
            if name in res["runs"]:
                continue
            t0 = time.time()
            mus = make_prop(a.tier, ids).run(Ws)[0]
            dm = debiased_mse(mus[-1], tr)
            res["runs"][name] = dict(mse=dm["debiased"], se=dm["se"], sec=time.time() - t0)
            json.dump(res, open(path, "w"))
            print(f"{a.tier} n={a.n} s={sd} {name:40s} mse={dm['debiased']:.3e} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
