"""Run a flopscope estimator (the V29-based submission bundle) on materialised MLPs with an
environment-variable configuration, metering FLOPs like the grader, and score it against the
dataset's Monte-Carlo reference. Optionally dumps V29's per-layer diagnostics (V17_DEBUG=1) as
corrector features. Resumable: one JSON + NPZ per MLP under --out. Written by an LLM agent.

    python experiments/gpu/run_v29.py --estimator submission/estimator.py --mlps data/gpu/mlps \
        --idx 0-19 --tag baseline --env V26_STRASSEN=5 --features
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import B_FLOPS, ROOT, load_mlp, mse, parse_idx, write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", default=os.path.join(ROOT, "submission", "estimator.py"))
    ap.add_argument("--mlps", default="data/gpu/mlps")
    ap.add_argument("--idx", default="0-19")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--env", nargs="*", default=[])
    ap.add_argument("--out", default="data/gpu/v29")
    ap.add_argument("--features", action="store_true", help="dump V17_DEBUG per-layer diagnostics")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for kv in a.env:  # must precede the estimator import (module-level constants read env)
        k, v = kv.split("=", 1)
        os.environ[k] = v
    if a.features:
        os.environ["V17_DEBUG"] = "1"
    # V29 caps Strassen at depth 4 on the first predict() of a process; the grader runs every MLP in
    # one process, so a parallel harness must lift the cap or every MLP is billed as a first predict
    os.environ.setdefault("V28_STRASSEN_FIRST", os.environ.get("V26_STRASSEN", "5"))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    outdir = os.path.join(a.out, a.tag)
    os.makedirs(outdir, exist_ok=True)
    todo = [i for i in parse_idx(a.idx) if a.force or not os.path.exists(os.path.join(outdir, f"mlp_{i:04d}.json"))]
    if not todo:
        print(f"[{a.tag}] nothing to do"); return

    import numpy as np
    from threadpoolctl import threadpool_limits
    import flopscope as flops
    import flopscope.numpy as fnp
    from whestbench import SetupContext
    from whestbench.domain import MLP

    spec = importlib.util.spec_from_file_location("est_under_test", a.estimator)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["est_under_test"] = mod
    spec.loader.exec_module(mod)
    est = mod.Estimator()
    debug = getattr(mod, "DEBUG", None)
    first = load_mlp(a.mlps, todo[0])
    L, n = first["W"].shape[0], first["W"].shape[1]
    with threadpool_limits(1):
        t0 = time.time()
        if hasattr(est, "setup"):
            est.setup(SetupContext(width=n, depth=L, flop_budget=B_FLOPS, api_version="gpu-pipeline",
                                   submission_dir=os.path.dirname(os.path.abspath(a.estimator)), seed=0))
        t_setup = time.time() - t0
        for i in todo:
            m = load_mlp(a.mlps, i)
            mlp = MLP(width=n, depth=L, weights=[fnp.asarray(w, dtype=fnp.float32) for w in m["W"]],
                      seed=m["seed"], name=m["name"])
            if debug is not None:
                debug.clear()
            ctx = flops.BudgetContext(flop_budget=B_FLOPS, wall_time_limit_s=None, quiet=True)
            t0 = time.time()
            err = None
            try:
                with ctx:
                    p = est.predict(mlp, B_FLOPS)
                    p = np.asarray(p, dtype=np.float32)
            except Exception as e:  # record, keep going
                err, p = f"{type(e).__name__}: {e}"[-500:], np.zeros((L, n), np.float32)
            wall = time.time() - t0
            row = dict(idx=i, name=m["name"], tag=a.tag, env=a.env, error=err,
                       mse_final=mse(p[-1], m["final_means"]), mse_all=mse(p, m["all_layer_means"]),
                       mse_per_layer=[mse(p[l], m["all_layer_means"][l]) for l in range(L)],
                       cb=ctx.flops_used / B_FLOPS, wall=wall, residual=ctx.residual_wall_time_s,
                       backend=ctx.flopscope_backend_time_s, overhead=ctx.flopscope_overhead_time_s,
                       setup_s=t_setup, maxrss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20)
            row["score"] = row["mse_final"] * max(0.1, row["cb"])
            feats = {}
            if debug:
                for rec in debug:
                    li = int(rec.get("layer", -1))
                    for k, v in rec.items():
                        if k == "layer" or v is None:
                            continue
                        arr = np.asarray(v, dtype=np.float64)
                        if arr.ndim == 2:        # D21: keep diagonal and row norms only
                            feats[f"{k}_diag_l{li}"] = np.diag(arr)
                            feats[f"{k}_rownorm_l{li}"] = np.sqrt((arr * arr).sum(1))
                        elif arr.ndim == 1:
                            feats[f"{k}_l{li}"] = arr
                        else:
                            feats[f"{k}_l{li}"] = np.full(n, float(arr))
                row["feature_layers"] = sorted({int(r.get("layer", -1)) for r in debug})
                debug.clear()
            np.savez_compressed(os.path.join(outdir, f"mlp_{i:04d}.npz"), preds=p, **feats)
            write_json(os.path.join(outdir, f"mlp_{i:04d}.json"), row)
            print(json.dumps({k: row[k] for k in ("idx", "tag", "mse_final", "cb", "score", "wall", "residual", "error")}), flush=True)


if __name__ == "__main__":
    main()
