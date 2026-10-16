"""Off-suite shape smoke test (the grader's smoke test uses shapes other than 16 x 1024).

    python -m arcmlp.eval.smoke_shapes submission/estimator.py 256x32 64x16 1024x3 4x2
Checks: setup() ok, predict() returns (depth, width), all finite, FLOPs <= 2**41, wall time.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time

import numpy as np


def main(path, shapes):
    import flopscope as flops
    import flopscope.numpy as fnp
    from threadpoolctl import threadpool_limits
    from whestbench import SetupContext
    from whestbench.domain import MLP
    spec = importlib.util.spec_from_file_location("est_smoke", path)
    mod = importlib.util.module_from_spec(spec); sys.modules["est_smoke"] = mod; spec.loader.exec_module(mod)
    out = []
    with threadpool_limits(1):
        for sh in shapes:
            w, d = (int(x) for x in sh.split("x"))
            rng = np.random.default_rng(w * 1000 + d)
            Ws = [fnp.asarray(rng.normal(0, np.sqrt(2 / w), (w, w)).astype(np.float32)) for _ in range(d)]
            mlp = MLP(width=w, depth=d, weights=Ws)
            est = mod.Estimator()
            if hasattr(est, "setup"):
                est.setup(SetupContext(width=w, depth=d, flop_budget=2 ** 41, api_version="smoke"))
            ctx = flops.BudgetContext(flop_budget=2 ** 41, quiet=True)
            t0 = time.time()
            with ctx:
                p = np.asarray(est.predict(mlp, 2 ** 41), dtype=np.float32)
            r = dict(shape=sh, out_shape=list(p.shape), finite=bool(np.isfinite(p).all()),
                     ok_shape=p.shape == (d, w), cb=ctx.flops_used / 2 ** 41, wall=round(time.time() - t0, 2),
                     residual=round(ctx.residual_wall_time_s, 3))
            print(json.dumps(r), flush=True); out.append(r)
    return out


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
