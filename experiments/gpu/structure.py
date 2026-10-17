"""Algorithm E diagnostics at the official shape: is the post-ReLU covariance (and the kappa3 (2,1)
matrix, the kappa4 memory B) low-rank + diagonal, and do k anchor neurons explain the rest?
Written by an LLM agent (Claude Code). One JSON per MLP under --out.

    python experiments/gpu/structure.py --mlps data/gpu/mlps --idx 0-2 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_mlp, ours_orientation, parse_idx, write_json  # noqa: E402
sys.path.insert(0, ROOT)
from arcmlp.estimators.spiked import StructureDiag  # noqa: E402
from arcmlp.reference.device import device_info, free_pool, resolve_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlps", default="data/gpu/mlps")
    ap.add_argument("--idx", default="0-2")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="data/gpu/structure")
    ap.add_argument("--ranks", default="16,64,256")
    ap.add_argument("--anchors", default="64,256")
    a = ap.parse_args()
    dev = resolve_device(a.device)
    os.makedirs(a.out, exist_ok=True)
    print("device:", json.dumps(device_info(dev), default=str), flush=True)
    ranks = tuple(int(x) for x in a.ranks.split(",")); anchors = tuple(int(x) for x in a.anchors.split(","))
    for i in parse_idx(a.idx):
        path = os.path.join(a.out, f"mlp_{i:04d}.json")
        if os.path.exists(path):
            continue
        m = load_mlp(a.mlps, i)
        t0 = time.time()
        pr = StructureDiag(device=dev, ranks=ranks, anchors=anchors)
        pr.run(ours_orientation(m["W"]))
        write_json(path, dict(idx=i, name=m["name"], device=dev, seconds=time.time() - t0, layers=pr.structure))
        last = pr.structure[-1]
        print(json.dumps(dict(idx=i, seconds=round(time.time() - t0), last_layer=last), default=str)[:400], flush=True)
        del pr
        free_pool()


if __name__ == "__main__":
    main()
