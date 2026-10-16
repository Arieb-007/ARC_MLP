"""Single-process probe of an estimator: FLOPs, wall, residual, peak RSS, predictions.

Unlike `whest run --runner subprocess` this applies NO address-space limit (RLIMIT_AS), so it
measures the true resident peak (VmHWM) — the grader's 8 GB is a RAM limit on the solution
process, and its arrays live in a flopscope server, so neither local mode is exact (see
results/stage0_submission.md). Predictions are saved so variants can be compared bit-for-bit.

    python -m arcmlp.eval.probe --estimator third_party/v29/estimator_v29.py \
        --dataset data/bake_1024x16_s0-3_n2e4 --split dev --idx 0 1 --env V26_STRASSEN_MIN=16 \
        --out data/runs/probe_leaf16.npz
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import resource
import sys
import time


def vm_status():
    out = {}
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(("VmHWM", "VmPeak", "VmRSS")):
                k, v = line.split(":")
                out[k] = int(v.split()[0]) * 1024
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--idx", type=int, nargs="+", default=[0])
    ap.add_argument("--env", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for kv in a.env:  # must precede the estimator import (module-level constants read env)
        k, v = kv.split("=", 1)
        os.environ[k] = v

    from threadpoolctl import threadpool_limits
    import numpy as np
    import flopscope as flops
    from whestbench import SetupContext
    from whestbench.dataset import load_dataset, iter_mlps

    ds = load_dataset(a.dataset, split=a.split)
    want = set(a.idx)
    mlps = [m for i, m in enumerate(iter_mlps(ds)) if i in want]
    del ds

    spec = importlib.util.spec_from_file_location("est_under_test", a.estimator)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["est_under_test"] = mod
    spec.loader.exec_module(mod)
    est = mod.Estimator()
    B = 2 ** 41
    rows, preds = [], []
    with threadpool_limits(1):
        t0 = time.time()
        if hasattr(est, "setup"):
            est.setup(SetupContext(width=mlps[0].width, depth=mlps[0].depth, flop_budget=B,
                                   api_version="probe", submission_dir=os.path.dirname(os.path.abspath(a.estimator)), seed=0))
        t_setup = time.time() - t0
        for i, mlp in zip(sorted(want), mlps):
            ctx = flops.BudgetContext(flop_budget=B, wall_time_limit_s=None, quiet=True)
            t0 = time.time()
            with ctx:
                p = est.predict(mlp, B)
                p = np.asarray(p, dtype=np.float32)
            rows.append(dict(idx=i, cb=ctx.flops_used / B, wall=time.time() - t0, residual=ctx.residual_wall_time_s,
                             backend=ctx.flopscope_backend_time_s, overhead=ctx.flopscope_overhead_time_s, **vm_status()))
            preds.append(p)
            print(json.dumps(rows[-1]), flush=True)
    np.savez(a.out, preds=np.array(preds), idx=np.array(sorted(want)))
    with open(a.out.replace(".npz", ".json"), "w") as f:
        json.dump(dict(rows=rows, setup_s=t_setup, env=a.env, estimator=a.estimator, dataset=a.dataset,
                       maxrss_kB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), f, indent=1)


if __name__ == "__main__":
    main()
