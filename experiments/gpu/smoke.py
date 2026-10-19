"""Smoke test of the device port: the dense K<=4 chain, the light coherent chain and the MC tool
must give the same numbers on the requested device as on the CPU at tiny sizes. Exit code 0 on
success; prints a JSON line. Written by an LLM agent (Claude Code).

    python experiments/gpu/smoke.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from arcmlp.estimators.coherent_even import CoherentEvenProp  # noqa: E402
from arcmlp.estimators.linear_response import LinearResponseProp, parse_lr  # noqa: E402
from arcmlp.reference import dense_cumulants as dc  # noqa: E402
from arcmlp.reference.device import device_info, resolve_device  # noqa: E402
from arcmlp.reference.mc import mc_truth_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--L", type=int, default=3)
    a = ap.parse_args()
    dev = resolve_device(a.device)
    info = device_info(dev)
    rng = np.random.default_rng(0)
    n, L = a.n, a.L
    Ws = [rng.normal(0, np.sqrt(2 / n), (n, n)) for _ in range(L)]
    out = dict(device=dev, info=info, checks={})
    t0 = time.time()
    for name, make in [("K2", lambda d: dc.DenseProp(Kt=2, device=d)),
                       ("K3c2", lambda d: dc.DenseProp(Kt=3, cap=2, device=d)),
                       ("K4c3", lambda d: dc.DenseProp(Kt=4, cap=3, device=d)),
                       ("coh2", lambda d: CoherentEvenProp("coherent", cap=2, device=d)),
                       ("ex4c3_k6", lambda d: CoherentEvenProp("ex4", cap=3, k6=True, device=d)),
                       ("lr_oneway_b1", lambda d: LinearResponseProp(**parse_lr("lr:oneway:b=1"), device=d)),
                       ("lr_rank", lambda d: LinearResponseProp(**{**parse_lr("lr:oneway:rank"), "rank_layers": (1,), "rank_r": 32}, device=d))]:
        ref = make("cpu").run(Ws)[0]
        got = make(dev).run(Ws)[0]
        err = float(np.max(np.abs(ref - got)) / (np.max(np.abs(ref)) + 1e-30))
        out["checks"][name] = err
    # float32 path (n > 64 uses float32 in the chains): compare device float32 vs CPU float32
    n2 = 72
    Ws2 = [rng.normal(0, np.sqrt(2 / n2), (n2, n2)) for _ in range(2)]
    ref = dc.DenseProp(Kt=3, cap=2, device="cpu").run(Ws2)[0]
    got = dc.DenseProp(Kt=3, cap=2, device=dev).run(Ws2)[0]
    out["checks"]["K3c2_f32_n72"] = float(np.max(np.abs(ref - got)) / np.max(np.abs(ref)))
    mc = mc_truth_device(Ws, 200_000, seed=3, device=dev)
    mc_cpu = mc_truth_device(Ws, 200_000, seed=3, device="cpu")
    out["checks"]["mc_mu_rms_diff_over_se"] = float(np.sqrt(np.mean((mc["mu"] - mc_cpu["mu"]) ** 2))
                                                   / np.sqrt(np.mean(mc_cpu["varL"]) / 200_000))
    out["seconds"] = time.time() - t0
    tol = {"K2": 1e-9, "K3c2": 1e-9, "K4c3": 1e-9, "coh2": 1e-9, "ex4c3_k6": 1e-9, "lr_oneway_b1": 1e-9, "lr_rank": 1e-9,
           "K3c2_f32_n72": 1e-4}
    if dev == "cuda":
        tol = {k: (1e-5 if v < 1e-6 else v) for k, v in tol.items()}   # different summation order on the GPU
    ok = all(out["checks"][k] <= tol[k] for k in tol) and out["checks"]["mc_mu_rms_diff_over_se"] < 4.0
    out["ok"] = bool(ok)
    print(json.dumps(out, default=str))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
