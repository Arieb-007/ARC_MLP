"""Materialise MLPs of a whestbench dataset (local bake or hf://owner/repo@rev) as per-MLP .npz
files: weights (L, n, n) float32 in the dataset's own orientation (z = W.T @ h, the starter-kit
convention), the organizers' Monte-Carlo reference (final_means, all_layer_means) and the dataset
metadata. Written by an LLM agent (Claude Code).

    python experiments/gpu/fetch_dataset.py --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 \
        --split mini --idx 0-99 --out data/gpu/mlps
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def parse_idx(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def open_dataset(spec, split):
    from whestbench.dataset import load_dataset
    if spec.startswith("hf://"):
        repo = spec[len("hf://"):]
        rev = None
        if "@" in repo:
            repo, rev = repo.split("@", 1)
        return load_dataset(repo, revision=rev, split=split)
    return load_dataset(spec, split=split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="mini")
    ap.add_argument("--idx", default="0-19")
    ap.add_argument("--out", default="data/gpu/mlps")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    idx = parse_idx(a.idx)
    t0 = time.time()
    ds = open_dataset(a.dataset, a.split)
    try:
        from whestbench.dataset import metadata
        md = metadata(ds)
    except Exception as e:  # pragma: no cover
        md = dict(error=str(e))
    n_total = len(ds)
    idx = [i for i in idx if i < n_total]
    ds = ds.with_format("numpy")
    meta = dict(dataset=a.dataset, split=a.split, n_total=n_total, idx=idx, metadata=md,
                n_samples=md.get("n_samples") if isinstance(md, dict) else None)
    for i in idx:
        path = os.path.join(a.out, f"mlp_{i:04d}.npz")
        if os.path.exists(path):
            continue
        row = ds[i]
        W = np.stack([np.asarray(w, dtype=np.float32) for w in row["weights"]])
        np.savez(path, weights=W, final_means=np.asarray(row["final_means"], dtype=np.float64),
                 all_layer_means=np.asarray(row["all_layer_means"], dtype=np.float64),
                 mlp_seed=int(row["mlp_seed"]), mlp_name=str(row["mlp_name"]), idx=i,
                 avg_variance=float(row.get("avg_variance", np.nan)))
        print(f"saved {path} ({W.shape}, {time.time() - t0:.0f}s)", flush=True)
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1, default=str)
    print(f"done: {len(idx)} MLPs from {a.dataset}:{a.split} (n_samples={meta['n_samples']}) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
