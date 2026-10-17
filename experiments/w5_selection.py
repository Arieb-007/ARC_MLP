"""W5: measured, cost-aware selection of diagram terms (research runner). Written by an LLM agent.

Terms = the Kt=4, cap=3 registry of arcmlp/reference/dense_cumulants.py (95 terms). Single terms are
mostly below the minimum detectable effect, and a term that WRITES state is inert unless some term
READS it, so selection is done over FEATURES = the slices a term reads (a term is active iff every
feature gating it is on; terms with no gate = the K=2 backbone and the producers of state):
  Cpair        covariance-only diagrams with >= 2 edges writing the mean/variance/covariance
               (c_ab^2, c_ab^3 corrections; O(1/n) Gaussian-input terms)
  K3(3,) K3(2,1) K3(1,1,1)                 read the kappa3 slice of z_l
  K4(4,) K4(2,2) K4(3,1) K4(2,1,1) K4(1,1,1,1)   read the kappa4 slice of z_l
               (tier 'light': only (4,) and (2,2) exist and they are the COHERENT values of
                arcmlp/estimators/coherent_even.py: W^4 b + 12 Bbar and 4 bbar/n + 4 Bbar)
  cap3         diagrams of cost 3 (order n^-3/2): the paper's K = 4 cap
All features on = DenseProp(Kt=4, cap=3) ('dense') / CoherentEvenProp('coherent', cap=3) ('light');
all off = the K=2 closure (checked in tests/test_w5.py).
Modes per network (final-layer MSE vs cached MC truth, debiased; every run stored under its feature
set, so the report can derive any contrast):
  loo        full, base, full - f, base + f for every feature f
  exhaustive every subset of the features whose billed cost (arcmlp/compress/cost_model.py) fits
             the budget (default 100 units at n = 1024, L = 16)
  configs    named feature sets given with --configs "A,B;C" (verification on fresh seeds)
Falsifier (written before running): the selection does not transfer across widths if the Spearman
rank correlation of leave-one-out deltas between n = 128 and n = 256 (light tier, L = 16) is < 0.7.

    python experiments/w5_selection.py --tier light --n 128 --L 16 --seeds 0-19 --mode loo
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arcmlp.compress.cost_model import feature_cost  # noqa: E402
from arcmlp.estimators.coherent_even import CoherentEvenProp  # noqa: E402
from arcmlp.reference import dense_cumulants as dc  # noqa: E402
from arcmlp.reference.mc import cached_truth, debiased_mse  # noqa: E402

K3F = ["K3(3,)", "K3(2,1)", "K3(1,1,1)"]
K4F = ["K4(4,)", "K4(2,2)", "K4(3,1)", "K4(2,1,1)", "K4(1,1,1,1)"]
FEATURES = {"dense": ["Cpair"] + K3F + K4F + ["cap3"],
            "light": ["Cpair"] + K3F + ["K4(4,)", "K4(2,2)", "cap3"]}


def gates(t):
    """Features a registry term needs (empty set: always on)."""
    g = {s for s in t.inputs if s.startswith("K")}
    if t.cost >= 3:
        g.add("cap3")
    if sum(t.p) <= 2 and all(s == "C(1,1)" for s in t.inputs) and len(t.diagram) >= 2:
        g.add("Cpair")
    return g


def group_of(t):
    """v1 grouping (kept for results/2026-10-17_w5_term_registry.md); superseded by gates()."""
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


def key(feats):
    return "S:" + ",".join(sorted(feats))


def make_prop(tier, feats):
    feats = set(feats)
    if tier == "dense":
        Kt = 4 if feats & set(K4F) else 3
        pr = dc.DenseProp(Kt=Kt, cap=3)
    else:
        pr = CoherentEvenProp(mode="coherent", cap=3)
    pr.active = [t for t in pr.active if gates(t) <= feats]
    return pr


def weights(n, L, seed):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, np.sqrt(2 / n), (n, n)) for _ in range(L)]


def jobs_for(tier, mode, budget, configs, L):
    F = FEATURES[tier]
    if mode == "loo":
        js = [set(F), set()]
        for f in F:
            js += [set(F) - {f}, {f}]
    elif mode == "exhaustive":
        js = [set(s) for r in range(len(F) + 1) for s in itertools.combinations(F, r)
              if feature_cost(s, tier, L=16) <= budget]
    else:
        js = [set(x for x in c.split(",") if x) for c in configs.split(";")]
        js = [set(F) if s == {"ALL"} else s for s in js]
    out, seen = [], set()
    for s in js:
        if key(s) not in seen:
            seen.add(key(s)); out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["dense", "light"], required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--N", type=float, default=4e6)
    ap.add_argument("--mode", choices=["loo", "exhaustive", "configs"], default="loo")
    ap.add_argument("--budget", type=float, default=100.0)
    ap.add_argument("--configs", default="")
    ap.add_argument("--outdir", default="data/w5")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    os.makedirs(a.outdir, exist_ok=True)
    jobs = jobs_for(a.tier, a.mode, a.budget, a.configs, a.L)
    print(f"{len(jobs)} feature sets per network", flush=True)
    for sd in range(lo, hi + 1):
        path = f"{a.outdir}/{a.tier}_n{a.n}_L{a.L}_s{sd}.json"
        res = json.load(open(path)) if os.path.exists(path) else dict(tier=a.tier, n=a.n, L=a.L, seed=sd, runs={})
        Ws = weights(a.n, a.L, sd)
        tr = cached_truth(Ws, a.n, a.L, sd, int(a.N), mc_seed=70_000 + 100 * a.L + sd, moments="final")
        res["floor"] = float(tr["floor"]); res["N"] = int(a.N)
        for feats in jobs:
            k = key(feats)
            if k in res["runs"]:
                continue
            t0 = time.time()
            mus = make_prop(a.tier, feats).run(Ws)[0]
            dm = debiased_mse(mus[-1], tr)
            res["runs"][k] = dict(feats=sorted(feats), mse=dm["debiased"], se=dm["se"], sec=time.time() - t0)
            tmp = path + ".tmp"
            json.dump(res, open(tmp, "w")); os.replace(tmp, path)
            print(f"{a.tier} n={a.n} L={a.L} s={sd} {k:70s} mse={dm['debiased']:.3e} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
