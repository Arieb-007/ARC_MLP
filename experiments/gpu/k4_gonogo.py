"""Go/no-go for the factored-kappa4 program (F-k4): dense Kt=4 cap-3 at depth 16 for widths the
GPU can hold, with the kappa4 input slices switched one at a time, against device Monte-Carlo
truth. Weights W ~ N(0, 2/n) from default_rng(seed) (our convention z = W h). Written by an LLM agent.

Pre-registered (docs/notes/w7_prereg_n256_L8.md, the plan of 2026-10-17): the program is dead if
K4c3 / K3c2 at n = 192, L = 16 exceeds 0.25; prediction 0.10 [0.05, 0.20] at n = 256.
Configs are feature sets of experiments/w5_selection.py (a term is on iff every slice it reads is on).

    python experiments/gpu/k4_gonogo.py --n 128 160 192 --L 16 --seeds 0-4 --N 1e7 --device cuda
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
from common import ROOT, parse_idx, write_json  # noqa: E402
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from arcmlp.estimators.ablations import ALL_FAMILIES, ALL_FAMILIES_OUT, FamilyProp, GenProp  # noqa: E402
from arcmlp.estimators.cc_proxy import CC_CONFIGS, CCProp  # noqa: E402
from arcmlp.estimators.coherent_even import CoherentEvenProp  # noqa: E402
from arcmlp.estimators.linear_response import ROUND4, LinearResponseProp, parse_lr  # noqa: E402
from arcmlp.estimators.a211_oracle import ROUND5, A211Oracle, parse_a211  # noqa: E402
from arcmlp.reference import dense_cumulants as dc  # noqa: E402
from arcmlp.reference.device import device_info, free_pool, peak_mem_gb, resolve_device  # noqa: E402
from arcmlp.reference.mc import cached_truth, debiased_mse  # noqa: E402
from w5_selection import FEATURES, K3F, K4F, gates  # noqa: E402

import subprocess
try:
    COMMIT = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
except Exception:
    COMMIT = ""
ALL = set(FEATURES["dense"])
RS = {"K4(3,1)", "K4(2,1,1)", "K4(1,1,1,1)"}
CONFIGS = {
    "K3c2": {"Cpair", *K3F},
    "K3c3": {"Cpair", *K3F, "cap3"},
    "K4c2": {"Cpair", *K3F, *K4F},
    "K4c3": ALL,
    "K4c3-no31": ALL - {"K4(3,1)"},
    "K4c3-no211": ALL - {"K4(2,1,1)"},
    "K4c3-no1111": ALL - {"K4(1,1,1,1)"},
    "paired_c3": ALL - RS,                     # (4,) + (2,2) exact, no random-sign slices
    "paired_c3+31": (ALL - RS) | {"K4(3,1)"},  # A1: the (3,1) family alone
    "paired_c3+211": (ALL - RS) | {"K4(2,1,1)"},
    "coh2": "light:coherent:2",
    "ex4c2": "light:ex4:2",
    "coh2_pboth": "light:coherent:2:pboth",
    "ex4c2_pboth": "light:ex4:2:pboth",
}
CONFIGS.update({f"cc:{k}": ("cc", k) for k in CC_CONFIGS if k != "K4c3"})
ROUND2 = "K3c2,K3c3,K4c2,K4c3,paired_c3,coh2,ex4c2,coh2_pboth,ex4c2_pboth," + ",".join(f"cc:{k}" for k in CC_CONFIGS if k != "K4c3")
# round 3: kappa4 generations (gen:A = births only at layers <= A) and cost-3 diagram families
# (fam:-x = all but family x; fam:+x+y = cost<=2 terms plus the named families)
ROUND3 = ("K3c2,K4c2,K4c3," + ",".join(f"fam:-{f}" for f in ALL_FAMILIES_OUT)
          + ",fam:+rs+k3e+p4e,fam:+rs_4+rs_3+k3e_3+p4e_3,fam:+rs_4+rs_3+k3e_3+p4e_3+c3_3,fam:+rs_4+rs_3+k3e_4+k3e_3+p4e_3,fam:+rs_4+rs_3+k3e_3+p4e_4+p4e_3")
# gen:A (kappa4 births truncated after layer A) is NOT in the list: dropping births while transmitting
# the old kappa4 is unstable at every width tried (n = 48 .. 192, 2026-10-18) — the births regulate the
# transmitted kappa4; generations cannot be ablated by term removal.


# round 4: linear response of the kappa4 memory (docs/notes/r4_prereg_linear_response.md); the rank
# diagnostics need one extra n^4 copy and the births spectrum three, so they run at n <= 192 only
ROUND4R = ROUND4 + ",lr:oneway:rank"


def make_prop(cfg, device, Ws=None):
    if cfg.startswith("a211:"):
        return A211Oracle(**parse_a211(cfg), device=device)
    if cfg.startswith("lr:"):
        return LinearResponseProp(**parse_lr(cfg), device=device)
    if cfg.startswith("gen:"):
        return GenProp(int(cfg[4:]), device=device)
    if cfg.startswith("fam:-"):
        drop = cfg[5:]
        return FamilyProp([f for f in (ALL_FAMILIES_OUT if "_" in drop else ALL_FAMILIES) if f != drop], device=device)
    if cfg.startswith("fam:+"):
        return FamilyProp([f for f in cfg[5:].split("+") if f], device=device)
    feats = CONFIGS[cfg]
    if isinstance(feats, tuple) and feats[0] == "cc":
        m211, m31, m1111, terms = CC_CONFIGS[feats[1]]
        return CCProp(Ws, m211, m31, m1111, terms, device=device)
    if isinstance(feats, str):
        parts = feats.split(":")
        _, mode, cap = parts[:3]
        pb = len(parts) > 3 and parts[3] == "pboth"
        return CoherentEvenProp(mode=mode, cap=int(cap), device=device, p31=pb, p211=pb)
    Kt = 4 if feats & set(K4F) else 3
    pr = dc.DenseProp(Kt=Kt, cap=3, device=device)
    pr.active = [t for t in pr.active if gates(t) <= feats]
    return pr


def weights(n, L, seed):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, np.sqrt(2 / n), (n, n)) for _ in range(L)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[128, 160, 192])
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--N", type=float, default=1e7)
    ap.add_argument("--device", default=None)
    ap.add_argument("--configs", default=",".join(CONFIGS), help="comma list; 'round2' = the CC proxy set")
    ap.add_argument("--out", default="data/gpu/k4")
    ap.add_argument("--mc-cache", default="data/gpu/mc")
    ap.add_argument("--chunk", type=int, default=65536)
    ap.add_argument("--chunk-rows", type=int, default=0, help="rows per chunk of the n^4 operations (memory)")
    a = ap.parse_args()
    dev = resolve_device(a.device)
    os.makedirs(a.out, exist_ok=True)
    print("device:", json.dumps(device_info(dev), default=str), flush=True)
    presets = {"round2": ROUND2, "round3": ROUND3, "round4": ROUND4, "round4+rank": ROUND4R, "round5": ROUND5}
    cfgs = [c.strip() for part in a.configs.split(",") for c in presets.get(part.strip(), part).split(",") if c.strip()]
    if a.chunk_rows:
        dc.K4_CHUNK_ROWS = a.chunk_rows
    for n in a.n:
        for sd in parse_idx(a.seeds):
            path = os.path.join(a.out, f"n{n}_L{a.L}_s{sd}.json")
            res = json.load(open(path)) if os.path.exists(path) else dict(n=n, L=a.L, seed=sd, N=int(a.N), device=dev, v={})
            Ws = weights(n, a.L, sd)
            t0 = time.time()
            tr = cached_truth(Ws, n, a.L, sd, int(a.N), mc_seed=70_000 + 100 * a.L + sd, cache_dir=a.mc_cache,
                              device=dev, chunk=a.chunk)
            res["floor"] = float(tr["floor"]); res["mc_seconds"] = float(tr.get("seconds", time.time() - t0))
            res["requested"] = sorted(set(res.get("requested", [])) | set(cfgs))   # rows of configs never requested since an error are stale
            for cfg in cfgs:
                if cfg in res["v"] and res["v"][cfg].get("error") is None:
                    continue
                t0 = time.time()
                try:
                    pr = make_prop(cfg, dev, Ws)
                    mus = pr.run(Ws)[0]
                    dm = debiased_mse(mus[-1], tr)
                    res["v"][cfg] = dict(mse=dm["debiased"], raw=dm["raw"], se=dm["se"], seconds=time.time() - t0,
                                         peak_mem_gb=peak_mem_gb(), error=None)
                    res["v"][cfg]["mu"] = [float(x) for x in mus[-1]]   # final-layer means (truth-free residuals, linearity)
                    if cfg.startswith("a211:"):
                        res["v"][cfg]["diag"] = pr.diag
                    if getattr(pr, "extra", None):
                        res["v"][cfg].update(pr.extra)
                    diag = [d for d in getattr(pr, "diag", []) if any(k.startswith("r2_") for k in d)]
                    if diag:
                        for key in ("r2_211", "r2_31", "corr211", "corr31"):
                            v = [d[key] for d in diag if d.get(key) is not None and np.isfinite(d[key])]
                            if v:
                                res["v"][cfg][key + "_mean"] = float(np.mean(v))
                        res["v"][cfg]["diag"] = diag
                    del pr
                except Exception as e:
                    res["v"][cfg] = dict(mse=None, error=f"{type(e).__name__}: {str(e)[:200]}", seconds=time.time() - t0,
                                         traceback=traceback.format_exc()[-800:])
                res["v"][cfg]["commit"] = COMMIT
                free_pool()
                write_json(path, res)
                r = res["v"][cfg]
                print(f"n={n} L={a.L} s={sd} {cfg:16s} mse={r['mse'] if r['mse'] is None else f'{r['mse']:.3e}'} "
                      f"{r['seconds']:.0f}s mem={r.get('peak_mem_gb')} {r['error'] or ''}", flush=True)


if __name__ == "__main__":
    main()
