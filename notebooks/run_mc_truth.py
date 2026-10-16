"""Notebook entry point: chunked MC ground truth for self-generated He-init MLPs.

    python notebooks/run_mc_truth.py --width 256 --depth 16 --seeds 0-9 --n-samples 1e8
Weights: W_l ~ N(0, 2/n) from np.random.default_rng(seed) in layer order (same convention as
experiments/*.py). Saves data/mc/<w>x<d>_s<seed>_N<N>.npz (+ .json provenance) with per-layer
means, final-layer variances and covariance; prints the MC floor mean Var/N.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arcmlp.reference.mc import mc_truth, save  # noqa: E402
from arcmlp.eval.local_eval import provenance  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--width", type=int, default=256)
ap.add_argument("--depth", type=int, default=16)
ap.add_argument("--seeds", default="0-9")
ap.add_argument("--n-samples", type=float, default=1e8)
ap.add_argument("--mc-seed", type=int, default=12345)
a = ap.parse_args()
lo, hi = (int(x) for x in a.seeds.split("-")) if "-" in a.seeds else (int(a.seeds), int(a.seeds))
os.makedirs("data/mc", exist_ok=True)
N = int(a.n_samples)
for seed in range(lo, hi + 1):
    rng = np.random.default_rng(seed)
    Ws = [rng.normal(0, np.sqrt(2 / a.width), (a.width, a.width)) for _ in range(a.depth)]
    out = mc_truth(Ws, N, seed=a.mc_seed + seed)
    path = f"data/mc/{a.width}x{a.depth}_s{seed}_N{N:.0e}.npz"
    save(path, out, meta=dict(width=a.width, depth=a.depth, weight_seed=seed, mc_seed=a.mc_seed + seed,
                              N=N, floor=float(out["varL"].mean() / N), seconds=out["seconds"],
                              provenance=provenance(".")))
    print(f"seed {seed}: N={N:.1e}, floor={out['varL'].mean()/N:.2e}, {out['seconds']:.0f}s -> {path}")
