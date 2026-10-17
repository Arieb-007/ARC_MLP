"""V1: validate the dense cumulant reference (arcmlp/reference/dense_cumulants.py) against Monte Carlo.

Part A (entrywise): n in {8,16,32}, L layers, weight seeds; MC full joint cumulant tensors of every
post-activation layer (orders 2..4) from `reps` independent MC runs (batch means -> MC standard
error per entry). Reports per layer and order: RMS and max |prop - MC| / RMS(MC), the MC SE / RMS,
and per-slice-type breakdown.
Part B (final-layer MSE): n=64, L=8, seeds 0-4, MC N=2e7: MSE of Kt = 2, 3, 4 (debiased).
Part C: Kt=2 at seed 3, n=256, L=8 vs experiments/conditional.py (3.563e-5).

Falsifier (written before running): the reference is wrong if (A) the Kt=4 deviation of any order
exceeds the Kt=3 deviation for orders <= 3, or does not shrink from n=16 to n=32, or (B) Kt=4 is
worse than Kt=3 in final-layer MSE on average.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arcmlp.reference import dense_cumulants as dc  # noqa: E402
from arcmlp.reference.mc import mc_truth, debiased_mse  # noqa: E402

SLICES = {2: [(2,), (1, 1)], 3: [(3,), (2, 1), (1, 1, 1)], 4: [(4,), (3, 1), (2, 2), (2, 1, 1), (1, 1, 1, 1)]}


def slice_mask(n, pat):
    """Boolean mask of dense-tensor entries whose index pattern has multiplicities `pat`."""
    k = sum(pat)
    idx = np.indices((n,) * k).reshape(k, -1).T
    out = np.zeros(len(idx), dtype=bool)
    for r, row in enumerate(idx):
        c = sorted(np.unique(row, return_counts=True)[1], reverse=True)
        out[r] = tuple(c) == pat
    return out.reshape((n,) * k)


def weights(n, L, seed):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, np.sqrt(2 / n), (n, n)) for _ in range(L)]


def part_a(n, L, seeds, N, reps):
    res = []
    masks = {k: {pat: slice_mask(n, pat) for pat in SLICES[k]} for k in (2, 3, 4)}
    for sd in seeds:
        Ws = weights(n, L, sd)
        t0 = time.time()
        subs = [mc_truth(Ws, N // reps, seed=1000 * sd + j, full_tensors_upto=4, keep_cov=False) for j in range(reps)]
        tmc = time.time() - t0
        props = {Kt: dc.propagate(Ws, K=Kt, dtype=np.float64, return_all=True)[2] for Kt in (2, 3, 4)}
        for l in range(L):
            row = dict(seed=sd, layer=l, n=n, N=N, reps=reps, mc_seconds=tmc)
            mc = {k: np.mean([s[f"K{k}"][l] for s in subs], axis=0) for k in (2, 3, 4)}
            se = {k: np.std([s[f"K{k}"][l] for s in subs], axis=0, ddof=1) / np.sqrt(reps) for k in (2, 3, 4)}
            mc[1] = np.mean([s["mu"][l] for s in subs], axis=0)
            for k in (2, 3, 4):
                rms = float(np.sqrt(np.mean(mc[k] ** 2)))
                row[f"rms_mc_k{k}"] = rms
                row[f"se_over_rms_k{k}"] = float(np.sqrt(np.mean(se[k] ** 2)) / rms)
                for pat in SLICES[k]:
                    msk = masks[k][pat]
                    row[f"rms_mc_k{k}{pat}"] = float(np.sqrt(np.mean(mc[k][msk] ** 2)))
                    row[f"se_over_rms_k{k}{pat}"] = float(np.sqrt(np.mean(se[k][msk] ** 2)) / max(row[f"rms_mc_k{k}{pat}"], 1e-300))
                for Kt in (2, 3, 4):
                    P = props[Kt][l]
                    est = P[k] if k in P else np.zeros_like(mc[k])
                    d = est - mc[k]
                    row[f"K{Kt}_rel_rms_k{k}"] = float(np.sqrt(np.mean(d ** 2)) / rms)
                    row[f"K{Kt}_rel_max_k{k}"] = float(np.abs(d).max() / rms)
                    for pat in SLICES[k]:
                        msk = masks[k][pat]
                        row[f"K{Kt}_rel_rms_k{k}{pat}"] = float(np.sqrt(np.mean(d[msk] ** 2)) / max(row[f"rms_mc_k{k}{pat}"], 1e-300))
            for Kt in (2, 3, 4):
                row[f"K{Kt}_rel_rms_mu"] = float(np.sqrt(np.mean((props[Kt][l][1] - mc[1]) ** 2)) / np.sqrt(np.mean(mc[1] ** 2)))
            res.append(row)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                              if k in ("seed", "layer") or k.startswith(("K2_rel_rms_k", "K3_rel_rms_k", "K4_rel_rms_k", "se_over_rms_k")) and "(" not in k}), flush=True)
    return res


def part_b(n, L, seeds, N):
    out = []
    for sd in seeds:
        Ws = weights(n, L, sd)
        t0 = time.time()
        tr = mc_truth(Ws, N, seed=5000 + sd)
        row = dict(seed=sd, n=n, L=L, N=N, floor=tr["floor"], mc_seconds=time.time() - t0)
        for Kt in (2, 3, 4):
            t1 = time.time()
            mus, _ = dc.propagate(Ws, K=Kt)
            dm = debiased_mse(mus[-1], tr)
            row[f"K{Kt}"] = dm["debiased"]; row[f"K{Kt}_se"] = dm["se"]; row[f"K{Kt}_sec"] = time.time() - t1
        out.append(row)
        print(json.dumps(row), flush=True)
    return out


def part_c():
    rng = np.random.default_rng(3); n, L = 256, 8
    Ws = [rng.normal(0, np.sqrt(2 / n), (n, n)) for _ in range(L)]
    mus, _ = dc.propagate(Ws, K=2, dtype=np.float64)
    # same MC protocol as experiments/conditional.py (rng continues after the weights)
    N, B = 1_600_000, 100_000
    sumL = np.zeros(n)
    for b in range(N // B):
        h = rng.normal(size=(B, n))
        for W in Ws:
            h = np.maximum(h @ W.T, 0)
        sumL += h.sum(0)
    mse = float(np.mean((mus[-1] - sumL / N) ** 2))
    print(json.dumps(dict(part="C", mse_K2=mse, reference=3.563e-5)), flush=True)
    return dict(mse_K2=mse, reference=3.563e-5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A", "B", "C"], required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--N", type=float, default=4e7)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.part == "A":
        r = part_a(a.n, a.L, a.seeds, int(a.N), a.reps)
    elif a.part == "B":
        r = part_b(a.n, a.L, a.seeds, int(a.N))
    else:
        r = part_c()
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1)
