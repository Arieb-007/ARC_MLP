"""E5: V29's fitted/adapted lambda_l vs the analytic lambda_l on 1024x16 networks (Stage 1).

Runs submission_analytic/estimator.py with ARC_LAMBDA=table (V29 behaviour) and ARC_LAMBDA_DEBUG=1,
which records per layer: lam_table (LAM*0.95), lam_adapted (V25 rule, the value V29 uses) and
lam_analytic (ours, computed on V29's own trajectory). Falsifier (stated before running): the
analytic replacement is not justified if |analytic/adapted - 1| > 0.2 at more than a third of the
layers.
    python experiments/e5_lambda_v29_vs_analytic.py data/bake_1024x16_s0-3_n2e4 dev 1 2 > out.json
"""
import importlib.util, json, os, sys
os.environ["ARC_LAMBDA"] = "table"; os.environ["ARC_LAMBDA_DEBUG"] = "1"
import numpy as np
import flopscope as flops
from threadpoolctl import threadpool_limits
from whestbench.dataset import load_dataset, iter_mlps
ds, split, idx = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]]
mlps = [m for i, m in enumerate(iter_mlps(load_dataset(ds, split=split))) if i in idx]
spec = importlib.util.spec_from_file_location("s1", "submission_analytic/estimator.py")
mod = importlib.util.module_from_spec(spec); sys.modules["s1"] = mod; spec.loader.exec_module(mod)
out = {}
with threadpool_limits(1):
    for i, mlp in zip(idx, mlps):
        mod.DEBUG.clear()
        est = mod.Estimator()
        with flops.BudgetContext(flop_budget=2**42, quiet=True):
            est.predict(mlp, 2**41)
        rows = {}
        for d in mod.DEBUG:
            li = d["layer"]
            for k, v in d.items():
                if k != "layer":
                    rows.setdefault(k, {})[li] = float(v)
        out[i] = rows
        an, ad = rows.get("lam_analytic", {}), rows.get("lam_adapted", {})
        print(f"MLP {i}: post-layer l | table | adapted(used, applied at l+1) | analytic | ratio", file=sys.stderr)
        for l in sorted(an):
            a = ad.get(l + 1); t = rows.get("lam_table", {}).get(l + 1)
            print(f"  {l:2d} | {t if t is None else f'{t:.3e}'} | {a if a is None else f'{a:.3e}'} | {an[l]:.3e} | "
                  f"{'-' if not a else f'{an[l]/a:.2f}'}", file=sys.stderr)
print(json.dumps(out))
