"""Round-5 section: the (2,1,1) slice's (j | kl) rank oracle (docs/notes/r5_cp_memory_theorem.md).
Written by an LLM agent (Claude Code) for team Arieb-007.
Standalone:  python experiments/gpu/report_r5.py data/r5_cpu/k4
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import paired_log_ratio  # noqa: E402

ORDER = ["K4c3-no1111", "a211:spec", "a211:1", "a211:4", "a211:16", "a211:64", "a211:256"]
MEANING = {"K4c3-no1111": "bulk dropped, A exact", "a211:spec": "bulk dropped, A exact (+ spectra; = K4c3-no1111)",
           "a211:1": "A → rank 1 (the λ ⊗ N ceiling)", "a211:4": "A → rank 4", "a211:16": "A → rank 16",
           "a211:64": "A → rank 64", "a211:256": "A → rank 256"}


def _f(x, p=2):
    return "—" if x is None or not np.isfinite(x) else (f"{x:.{p}f}" if abs(x) < 1e3 else f"{x:.1e}")


def section(rows_all):
    rows_all = [r for r in rows_all if any(c.startswith("a211:") for c in r.get("v", {}))]
    if not rows_all:
        return [], {}
    byn = defaultdict(list)
    for r in rows_all:
        byn[(r["n"], r["L"])].append(r)
    out = {}
    L = ["", "### Round 5 — rank of the (2,1,1) κ4 slice in its (j | kl) unfolding (rules: docs/notes/r5_cp_memory_theorem.md)", "",
         "Dense K4c3 with the (1,1,1,1) bulk dropped and the (2,1,1) input slice A[j,k,l] = κ4(z)[j,j,k,l] projected after every linear step "
         "onto its top-r (j | kl) directions. Ratios: geometric means over seeds, paired bootstrap CI. 'resid' = ‖μ_cfg − μ_no1111‖² / "
         "‖μ_K3c2 − μ_no1111‖² (truth-free). Theorem A: the chain a CP memory of rank r implements is exactly a211:r.", "",
         "| n | L | config | meaning | / K4c3 [95% CI] | / K4c3-no1111 | resid | s/run | errors |", "|---|---|---|---|---|---|---|---|---|"]
    for (n, Lr), rows in sorted(byn.items()):
        def col(cfg):
            return [r["v"].get(cfg, {}).get("mse") for r in rows]

        def ratio(a, b):
            A, B = col(a), col(b)
            ok = [(x, y) for x, y in zip(A, B) if x is not None and y is not None and x > 0 and y > 0]
            return paired_log_ratio([x for x, _ in ok], [y for _, y in ok]) if ok else (np.nan,) * 4

        def mu(r, cfg):
            v = r["v"].get(cfg, {})
            return np.asarray(v["mu"]) if v.get("mu") is not None and v.get("error") is None else None

        def resid(cfg):
            vals = []
            for r in rows:
                a, ref, bg = mu(r, cfg), mu(r, "K4c3-no1111"), mu(r, "K3c2")
                if a is None or ref is None or bg is None:
                    continue
                d = float(np.sum((bg - ref) ** 2))
                if d > 0:
                    vals.append(float(np.sum((a - ref) ** 2)) / d)
            return float(np.median(vals)) if vals else np.nan

        g = out.setdefault(f"n{n}", {})
        for cfg in [c for c in ORDER if any(c in r["v"] for r in rows)]:
            errs = [r["v"][cfg].get("error") for r in rows if cfg in r["v"] and r["v"][cfg].get("error")]
            vals = [x for x in col(cfg) if x is not None]
            sec = float(np.mean([r["v"][cfg].get("seconds") or 0 for r in rows if cfg in r["v"]]))
            if errs and not vals:
                L.append(f"| {n} | {Lr} | {cfg} | {MEANING.get(cfg, '')} | error: {errs[0][:70]} | — | — | {sec:.0f} | {len(errs)} |")
                continue
            r4, rn = ratio(cfg, "K4c3"), ratio(cfg, "K4c3-no1111")
            rs = resid(cfg)
            fr = lambda t: "—" if not np.isfinite(t[0]) else f"{_f(t[0])} [{_f(t[1])}, {_f(t[2])}]"
            L.append(f"| {n} | {Lr} | {cfg} | {MEANING.get(cfg, '')} | {fr(r4)} | {_f(rn[0])} | {_f(rs, 3)} | {sec:.0f} | {len(errs)} |")
            g[cfg] = dict(ratio_K4c3=r4[:3], ratio_no1111=rn[:3], resid=rs)
    # spectra
    sp = defaultdict(list)
    for r in rows_all:
        for d in r["v"].get("a211:spec", {}).get("diag", []) or []:
            sp[(r["n"], d["layer"])].append(d)
    if sp:
        L += ["", "(j | kl) spectrum of A = κ4(z)(2,1,1) per layer (means over seeds): f1 = energy fraction of the top direction, "
              "f_ones = fraction along the all-ones j-direction (the partial-trace / λ channel of Proposition B), k50/k90 = directions for 50 / 90 %.", "",
              "| n | layer | f1 | f4 | f16 | f64 | f_ones | k50 | k90 |", "|---|---|---|---|---|---|---|---|---|"]
        out["spectrum"] = {}
        for (n, layer), ds in sorted(sp.items()):
            if layer % 2:
                continue
            m = lambda k: float(np.mean([d[k] for d in ds]))
            fk = lambda k: "—" if any(d.get(k) is None for d in ds) else f"{np.mean([d[k] for d in ds]):.0f}"
            L.append(f"| {n} | {layer} | {_f(m('f1'))} | {_f(m('f4'))} | {_f(m('f16'))} | {_f(m('f64'))} | {_f(m('f_ones'))} | {fk('k50')} | {fk('k90')} |")
            out["spectrum"][f"n{n}_l{layer}"] = dict(f1=m("f1"), f4=m("f4"), f16=m("f16"), f64=m("f64"), f_ones=m("f_ones"))
    L += ["", "Kill rule: the rank r giving ≤ 1.3 × K4c3-no1111 must not grow proportionally with n (192 → 256); "
          "Proposition B predicts f_ones and f1 ≈ 0.15–0.35, flat in n.", ""]
    return L, out


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "data/gpu/k4"
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "n*_L*_s*.json")))]
    print("\n".join(section(rows)[0]))
