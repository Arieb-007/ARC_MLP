"""Notebook entry point: grader-faithful evaluation of an estimator on the public HF split.

Run in a Kaggle/Colab CPU notebook after scripts/kaggle_bootstrap.sh (the cloud sandbox cannot
reach HF). Writes results/YYYY-MM-DD_<name>.{json,md}; then scripts/push_results.sh.

    python notebooks/run_official_eval.py --estimator submission/estimator.py --split mini \
        --n-mlps 20 --name v29_leaf32_dev20
Uses `whest run --runner subprocess --max-threads 1` (RLIMIT_AS 8 GiB, 0.4 s residual, 120 s wall).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arcmlp.eval.local_eval import provenance, run_whest, summarize  # noqa: E402
from arcmlp.eval.report import write_result  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--estimator", default="submission/estimator.py")
ap.add_argument("--dataset", default="hf://aicrowd/arc-whestbench-public-2026@v2-phase2")
ap.add_argument("--split", default="mini")
ap.add_argument("--n-mlps", type=int, default=20)
ap.add_argument("--name", required=True)
ap.add_argument("--env", nargs="*", default=[], help="KEY=VALUE pairs passed to the worker")
a = ap.parse_args()

env = dict(kv.split("=", 1) for kv in a.env)
os.makedirs("data/runs", exist_ok=True)
raw_json = f"data/runs/{a.name}.json"
rep = run_whest(a.estimator, a.dataset, raw_json, n_mlps=a.n_mlps, split=a.split, extra_env=env)
s = summarize(rep)
write_result(a.name, dict(kind="official_eval", estimator=a.estimator, dataset=a.dataset, split=a.split,
                          n_mlps=a.n_mlps, env=env, summary=s, provenance=provenance("."),
                          whest_returncode=rep["_returncode"], seconds=rep["_seconds"]))
print(json.dumps(s["agg"], indent=1))
