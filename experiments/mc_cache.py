"""Precompute and cache MC truths (data/mc) for the W7/W5 networks: weights from
np.random.default_rng(seed) (W ~ N(0, 2/n), layer order), MC seed 70000 + 100 L + seed, all cores.
    python experiments/mc_cache.py --n 64 --L 8 16 --seeds 0-19 --N 2e7"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arcmlp.reference.mc import cached_truth
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, required=True); ap.add_argument("--L", type=int, nargs="+", default=[8, 16])
ap.add_argument("--seeds", default="0-9"); ap.add_argument("--N", type=float, default=2e7)
a = ap.parse_args(); lo, hi = (int(x) for x in a.seeds.split("-"))
for L in a.L:
    for sd in range(lo, hi + 1):
        rng = np.random.default_rng(sd); Ws = [rng.normal(0, np.sqrt(2 / a.n), (a.n, a.n)) for _ in range(L)]
        t0 = time.time(); tr = cached_truth(Ws, a.n, L, sd, int(a.N), mc_seed=70_000 + 100 * L + sd, moments="final")
        print(f"n={a.n} L={L} seed={sd} floor={float(tr['floor']):.2e} {time.time()-t0:.0f}s", flush=True)
