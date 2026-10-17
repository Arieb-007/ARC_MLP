"""Sanity check of the dataset's Monte-Carlo reference with our own device MC: the mean squared
difference between the two must be ~ floor_ours + floor_ref (both are unbiased estimates of the
same means). Also records the per-neuron output variance (the 0.08/neuron figure of FINDINGS F6).
Written by an LLM agent (Claude Code).

    python experiments/gpu/mc_check.py --mlps data/gpu/mlps --idx 0,1 --N 1e8 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_mlp, mse, ours_orientation, parse_idx, read_json, write_json  # noqa: E402
sys.path.insert(0, ROOT)
from arcmlp.reference.device import device_info, resolve_device  # noqa: E402
from arcmlp.reference.mc import mc_truth_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlps", default="data/gpu/mlps")
    ap.add_argument("--idx", default="0")
    ap.add_argument("--N", type=float, default=1e8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="data/gpu/mc_check")
    ap.add_argument("--chunk", type=int, default=65536)
    a = ap.parse_args()
    dev = resolve_device(a.device)
    os.makedirs(a.out, exist_ok=True)
    meta = read_json(os.path.join(a.mlps, "meta.json"), {}) or {}
    n_ref = meta.get("n_samples")
    print("device:", json.dumps(device_info(dev), default=str), flush=True)
    for i in parse_idx(a.idx):
        path = os.path.join(a.out, f"mlp_{i:04d}.json")
        if os.path.exists(path):
            continue
        m = load_mlp(a.mlps, i)
        Ws = ours_orientation(m["W"])
        t0 = time.time()
        tr = mc_truth_device(Ws, int(a.N), seed=90_000 + i, device=dev, chunk=a.chunk, log_every=60)
        d = mse(tr["mu"][-1], m["final_means"])
        floor_ours = float(np.mean(tr["varL"]) / int(a.N))
        floor_ref = float(np.mean(tr["varL"]) / n_ref) if n_ref else None
        row = dict(idx=i, name=m["name"], N=int(a.N), n_samples_ref=n_ref, device=dev, seconds=time.time() - t0,
                   mse_ours_vs_ref=d, floor_ours=floor_ours, floor_ref=floor_ref,
                   expected=(floor_ours + floor_ref) if floor_ref else None,
                   ratio=(d / (floor_ours + floor_ref)) if floor_ref else None,
                   var_per_neuron_mean=float(np.mean(tr["varL"])),
                   mse_all_layers_ours_vs_ref=mse(tr["mu"], m["all_layer_means"]))
        np.savez_compressed(os.path.join(a.out, f"mlp_{i:04d}.npz"), mu=tr["mu"], varL=tr["varL"])
        write_json(path, row)
        print(json.dumps(row, default=str), flush=True)


if __name__ == "__main__":
    main()
