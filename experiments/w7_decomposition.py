"""W7 decomposition: which kappa4 slice does the coherent replacement get wrong? (research runner)

Written by an LLM agent (Claude Code). Inside the dense Kt=4 reference, replace the pre-activation
kappa4 slices read by the ReLU step with their coherent values, one at a time, via slice hooks:
  (4,)_a  -> (W^4 b)_a + 12 Bbar       (2,2)_ab -> 4 bbar/n + 4 Bbar
with b_m = kappa4(h_m), Bbar = n^-2 sum_{m != p} kappa(h_m, h_m, h_p, h_p) from the dense post-ReLU state,
and (cap 3 only) zero the random-sign slices (3,1), (2,1,1), (1,1,1,1).
Falsifier (written before running) for "the coherent (2,2) rank-one value is adequate and the (4,)
replacement is what fails": replacing only (2,2) moves the final-layer MSE by more than replacing only (4,).

    python experiments/w7_decomposition.py --n 64 --L 8 --seeds 0-4
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
from arcmlp.reference.mc import cached_truth, debiased_mse  # noqa: E402

RS = [(3, 1), (2, 1, 1), (1, 1, 1, 1)]


class DecompProp(dc.DenseProp):
    def __init__(self, Ws, cap, coh4=False, coh22=False, zero_rs=False):
        super().__init__(Kt=4, cap=cap, slice_hook=self._hook)
        self.Ws, self.coh4, self.coh22, self.zero_rs = Ws, coh4, coh22, zero_rs
        self._h = None

    def relu_step(self, Z, layer=0):
        K = super().relu_step(Z, layer)
        self._h = K
        return K

    def _hook(self, layer, Z, sl):
        n = Z[1].shape[0]
        if self._h is None or layer == 0:
            b, Bbar = np.zeros(n), 0.0
        else:
            K4 = self._h[4]
            b = np.einsum("aaaa->a", K4).astype(np.float64)
            B = np.einsum("aabb->ab", K4).astype(np.float64)
            Bbar = float((B.sum() - np.trace(B)) / n ** 2)
        W2 = (self.Ws[layer] * self.Ws[layer]).astype(np.float64)
        dt = sl[(4,)].dtype
        if self.coh4:
            sl[(4,)] = ((W2 * W2) @ b + 12.0 * Bbar).astype(dt)
        if self.coh22:
            c = np.full((n, n), 4.0 * b.mean() / n + 4.0 * Bbar, dtype=dt)
            np.fill_diagonal(c, 0.0)
            sl[(2, 2)] = c
        if self.zero_rs:
            for p in RS:
                if p in sl:
                    sl[p] = np.zeros_like(sl[p])


VARIANTS = {  # name: (cap, coh4, coh22, zero_rs)
    "c2_ex_ex": (2, False, False, False), "c2_ex4_coh22": (2, False, True, False),
    "c2_coh4_ex22": (2, True, False, False), "c2_coh_coh": (2, True, True, False),
    "c3_full": (3, False, False, False), "c3_rs0": (3, False, False, True),
    "c3_coh_coh": (3, True, True, False), "c3_ex4_coh22": (3, False, True, False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--N", type=float, default=4e6)
    ap.add_argument("--out", default="data/w7_decomp.json")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for sd in range(lo, hi + 1):
        rng = np.random.default_rng(sd)
        Ws = [rng.normal(0, np.sqrt(2 / a.n), (a.n, a.n)) for _ in range(a.L)]
        tr = cached_truth(Ws, a.n, a.L, sd, int(a.N), mc_seed=70_000 + 100 * a.L + sd, moments="final")
        k = f"n{a.n}_L{a.L}_s{sd}"
        res.setdefault(k, dict(n=a.n, L=a.L, seed=sd, floor=float(tr["floor"]), v={}))
        for name, (cap, c4, c22, z) in VARIANTS.items():
            if name in res[k]["v"]:
                continue
            t0 = time.time()
            mus = DecompProp(Ws, cap, c4, c22, z).run(Ws)[0]
            dm = debiased_mse(mus[-1], tr)
            res[k]["v"][name] = dict(mse=dm["debiased"], se=dm["se"], sec=time.time() - t0)
            json.dump(res, open(a.out + ".tmp", "w")); os.replace(a.out + ".tmp", a.out)
            print(f"{k} {name:14s} mse={dm['debiased']:.3e} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
