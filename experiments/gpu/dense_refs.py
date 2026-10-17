"""Dense / light cumulant references at the official shape on the GPU (or CPU for small shapes),
scored against the dataset's Monte-Carlo reference. One JSON + NPZ per (variant, MLP) under --out;
resumable; out-of-memory and other failures are recorded, not fatal. Written by an LLM agent.

Variants (see docs/notes/w7_coherent_even_orders.md, results/2026-10-17_w7_coherent_even.md):
  K2      dense Kt=2 (== starter kit; also checks the weight orientation against kit_closure)
  K3c2    dense Kt=3, cap 2 (standard K=3)        K3c3   dense Kt=3, cap 3
  coh2    K=3 + coherent kappa4 (rank-one (2,2), W^4 b diagonal), cap 2;  coh3 at cap 3
  ex4c2   coh2 with the exact paired (4,) slice (one matmul per layer); exallc2 exact (4,) and (2,2)
  K4c2 / K4c3  dense Kt=4 (n <= ~192 on a 48 GB GPU)
  <light>_p31 / _p211 / _pboth   + the conditional-coherent proxies of the random-sign slices (cc_proxy.py)

    python experiments/gpu/dense_refs.py --mlps data/gpu/mlps --idx 0-9 --variants K2,K3c2,coh2 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, kit_closure, load_mlp, mse, ours_orientation, parse_idx, write_json  # noqa: E402
sys.path.insert(0, ROOT)
from arcmlp.estimators.coherent_even import CoherentEvenProp  # noqa: E402
from arcmlp.reference import dense_cumulants as dc  # noqa: E402
from arcmlp.reference.device import device_info, free_pool, peak_mem_gb, resolve_device  # noqa: E402


def make(variant, device):
    if variant == "K2":
        return dc.DenseProp(Kt=2, device=device)
    if variant.startswith("K3c") or variant.startswith("K4c"):
        return dc.DenseProp(Kt=int(variant[1]), cap=int(variant[3]), device=device)
    base = variant.split("_")[0]
    cap = int(base[-1])
    mode = {"coh": "coherent", "ex4c": "ex4", "exallc": "exall", "ex22": "exact22"}[base[:-1] if base != "ex22" else "ex22"]
    flags = variant.split("_")[1:]
    pb = "pboth" in flags
    return CoherentEvenProp(mode=mode, cap=cap, k6="k6" in flags, k8="k8" in flags, k3sq="k3sq" in flags, device=device,
                            p31=pb or "p31" in flags, p211=pb or "p211" in flags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlps", default="data/gpu/mlps")
    ap.add_argument("--idx", default="0-9")
    ap.add_argument("--variants", default="K2,K3c2,coh2,ex4c2")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="data/gpu/dense")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    dev = resolve_device(a.device)
    os.makedirs(a.out, exist_ok=True)
    info = device_info(dev)
    print("device:", json.dumps(info, default=str), flush=True)
    for i in parse_idx(a.idx):
        m = None
        for v in a.variants.split(","):
            v = v.strip()
            if not v:
                continue
            jpath = os.path.join(a.out, f"{v}_mlp_{i:04d}.json")
            if os.path.exists(jpath) and not a.force:
                continue
            if m is None:
                m = load_mlp(a.mlps, i)
                Ws = ours_orientation(m["W"])
            L, n = m["W"].shape[0], m["W"].shape[1]
            row = dict(variant=v, idx=i, name=m["name"], n=n, L=L, device=dev)
            t0 = time.time()
            try:
                pr = make(v, dev)
                mus = pr.run(Ws)[0]
                row.update(mse_final=mse(mus[-1], m["final_means"]), mse_all=mse(mus, m["all_layer_means"]),
                           mse_per_layer=[mse(mus[l], m["all_layer_means"][l]) for l in range(L)],
                           seconds=time.time() - t0, peak_mem_gb=peak_mem_gb(), error=None)
                if v == "K2":
                    kit = kit_closure(m["W"])
                    row["k2_vs_kit_maxabs"] = float(np.max(np.abs(kit - mus)))
                    row["kit_mse_final"] = mse(kit[-1], m["final_means"])
                np.savez_compressed(os.path.join(a.out, f"{v}_mlp_{i:04d}.npz"), preds=mus.astype(np.float64))
                del pr, mus
            except Exception as e:
                row.update(error=f"{type(e).__name__}: {str(e)[:300]}", seconds=time.time() - t0,
                           traceback=traceback.format_exc()[-1500:])
            free_pool()
            write_json(jpath, row)
            print(json.dumps({k: row.get(k) for k in ("variant", "idx", "mse_final", "seconds", "peak_mem_gb", "error", "k2_vs_kit_maxabs")}), flush=True)


if __name__ == "__main__":
    main()
