"""Round-4 section of the GPU report: linear response of the kappa4 memory.

Written by an LLM agent (Claude Code) for team Arieb-007. Rules: docs/notes/r4_prereg_linear_response.md.
Standalone use (any directory of k4_gonogo JSON rows):  python experiments/gpu/report_r4.py data/r4_cpu/k4
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

G1_MAX, NONLIN_MAX, G2_MIN, BF_DIVERGE = 1.3, 0.1, 0.5, 10.0
ORDER = ["lr:bg", "lr:twoway", "lr:oneway", "lr:oneway:eps=0.5", "lr:oneway:ro=1", "lr:oneway:ro=2", "lr:oneway:ro=4",
         "lr:oneway:b=1", "lr:oneway:b=2", "lr:oneway:b=4", "lr:oneway:b=8", "lr:oneway:ro=1:b=4",
         "lr:oneway:bf=4", "lr:oneway:bf=8", "lr:oneway:kind=coh", "lr:oneway:kind=rs", "lr:oneway:rank"]
MEANING = {
    "lr:bg": "background only (= K3c3)", "lr:twoway": "two-way (= K4c3; check)", "lr:oneway": "one-way, all on",
    "lr:oneway:eps=0.5": "read-outs × 0.5", "lr:oneway:ro=1": "read-outs at the last step only (direct path)",
    "lr:oneway:ro=2": "read-outs in the last 2 steps", "lr:oneway:ro=4": "read-outs in the last 4 steps",
    "lr:oneway:b=1": "1 generation of births", "lr:oneway:b=2": "2 generations", "lr:oneway:b=4": "4 generations",
    "lr:oneway:b=8": "8 generations", "lr:oneway:ro=1:b=4": "direct path, 4 generations (cheapest add-on)",
    "lr:oneway:bf=4": "births only in the first 4 steps", "lr:oneway:bf=8": "births only in the first 8 steps",
    "lr:oneway:kind=coh": "read-outs through paired z-slices only", "lr:oneway:kind=rs": "read-outs through random-sign z-slices only",
    "lr:oneway:rank": "one-way + spectra (same result as lr:oneway)",
}


def _f(x, p=2):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{p}f}" if abs(x) < 1e3 else f"{x:.1e}"


def _ci(t):
    return "—" if not np.isfinite(t[0]) else f"{_f(t[0])} [{_f(t[1])}, {_f(t[2])}]"


def section(rows_all):
    """rows_all: k4_gonogo JSON rows (any n, L). Returns (markdown lines, json dict); empty if no lr rows."""
    rows_all = [r for r in rows_all if any(c.startswith("lr:") for c in r.get("v", {}))]
    if not rows_all:
        return [], {}
    byn = defaultdict(list)
    for r in rows_all:
        byn[(r["n"], r["L"])].append(r)
    out = {}
    L = ["", "### Round 4 — linear response of the κ4 memory (rules: docs/notes/r4_prereg_linear_response.md)", "",
         "Three chains in lockstep: background S̄ (= K3c3), κ4 probe driven by S̄ (one-way) or by S (two-way = K4c3), perturbed chain S "
         "receiving the probe's read-outs. Ratios: geometric means over seeds, paired bootstrap CI. 'recovery' = 1 − log(cfg/K4c3)/log(K3c3/K4c3) "
         "against the MC truth; 'resid' = ‖μ_cfg − μ_K4c3‖² / ‖μ_K3c3 − μ_K4c3‖² (truth-free; 0 = K4c3, 1 = K3c3).", "",
         "| n | L | config | meaning | / K4c3 [95% CI] | recovery | resid | s/run | errors |", "|---|---|---|---|---|---|---|---|---|"]
    gates = {}
    for (n, Lr), rows in sorted(byn.items()):
        def col(cfg):
            return [r["v"].get(cfg, {}).get("mse") for r in rows]

        def ratio(a, b):
            A, B = col(a), col(b)
            ok = [(x, y) for x, y in zip(A, B) if x is not None and y is not None and x > 0 and y > 0]
            return paired_log_ratio([x for x, _ in ok], [y for _, y in ok]) if ok else (np.nan,) * 4

        k3name = "K3c3" if any(r["v"].get("K3c3", {}).get("mse") for r in rows) else "lr:bg"
        gap = ratio(k3name, "K4c3")

        def mu(r, cfg):
            v = r["v"].get(cfg, {})
            return np.asarray(v["mu"]) if v.get("mu") is not None and v.get("error") is None else None

        def resid(cfg):
            vals = []
            for r in rows:
                a, ref, bg = mu(r, cfg), mu(r, "lr:twoway"), mu(r, "lr:bg")
                if a is None or ref is None or bg is None:
                    continue
                d = float(np.sum((bg - ref) ** 2))
                if d > 0:
                    vals.append(float(np.sum((a - ref) ** 2)) / d)
            return float(np.median(vals)) if vals else np.nan

        cfgs = [c for c in ORDER if any(c in r["v"] for r in rows)] + sorted(
            {c for r in rows for c in r["v"] if c.startswith("lr:")} - set(ORDER))
        g = gates.setdefault(n, {})
        for cfg in cfgs:
            errs = [r["v"][cfg].get("error") for r in rows if cfg in r["v"] and r["v"][cfg].get("error")]
            vals = [x for x in col(cfg) if x is not None]
            sec = float(np.mean([r["v"][cfg].get("seconds") or 0 for r in rows if cfg in r["v"]]))
            if errs and not vals:
                L.append(f"| {n} | {Lr} | {cfg} | {MEANING.get(cfg, '')} | error: {errs[0][:70]} | — | — | {sec:.0f} | {len(errs)} |")
                continue
            r4 = ratio(cfg, "K4c3")
            rec = 1.0 - np.log(r4[0]) / np.log(gap[0]) if np.isfinite(r4[0]) and np.isfinite(gap[0]) and gap[0] > 1 else np.nan
            rs = resid(cfg)
            refs = [x for x in col(k3name) + col("K4c3") if x is not None and np.isfinite(x) and x > 0]
            med = float(np.median(vals)) if vals else np.nan
            diverged = bool(vals) and bool(refs) and (not np.isfinite(med) or med > BF_DIVERGE * max(refs))
            tag = " (diverged: > 10 × max(K3c3, K4c3))" if diverged else ""
            L.append(f"| {n} | {Lr} | {cfg} | {MEANING.get(cfg, '')}{tag} | {_ci(r4)} | {_f(rec)} | {_f(rs, 3)} | {sec:.0f} | {len(errs)} |")
            g[cfg] = dict(ratio=r4[:3], recovery=rec, resid=rs, diverged=diverged)
        # linearity: ||mu(1) - 2 mu(1/2) + mu_bar||^2 / ||mu(1) - mu_bar||^2
        nl = []
        for r in rows:
            m1, mh, mb = mu(r, "lr:oneway"), mu(r, "lr:oneway:eps=0.5"), mu(r, "lr:bg")
            if m1 is not None and mh is not None and mb is not None:
                d = float(np.sum((m1 - mb) ** 2))
                if d > 0:
                    nl.append(float(np.sum((m1 - 2 * mh + mb) ** 2)) / d)
        g["nonlinearity"] = float(np.median(nl)) if nl else np.nan
        g["gap_K3c3_K4c3"] = gap[:3]
        # stability: probe norm growth per layer after the births stop
        for A in (4, 8):
            cfg = f"lr:oneway:bf={A}"
            gr = []
            for r in rows:
                pn = r["v"].get(cfg, {}).get("pnorm")
                if pn and len(pn) == Lr and pn[A - 1] > 0 and np.isfinite(pn[-1]):
                    gr.append((pn[-1] / pn[A - 1]) ** (1.0 / (Lr - A)))
            g[f"growth_bf{A}"] = float(np.median(gr)) if gr else np.nan
    out["rows"] = {f"n{n}": g for n, g in gates.items()}
    # gates summary
    L += ["", "Gates (pre-registered):", ""]
    for n in sorted(gates):
        g = gates[n]
        tw, ow, cheap = g.get("lr:twoway", {}), g.get("lr:oneway", {}), g.get("lr:oneway:ro=1:b=4", {})
        bf4, bf8 = g.get("lr:oneway:bf=4", {}), g.get("lr:oneway:bf=8", {})
        p0 = tw.get("ratio", (np.nan,))[0]
        g1r = ow.get("ratio", (np.nan,))[0]
        nlv = g.get("nonlinearity", np.nan)
        g1 = "—" if not np.isfinite(g1r) else ("**pass**" if g1r <= G1_MAX and (not np.isfinite(nlv) or nlv <= NONLIN_MAX) else "**fail**")
        g2v = cheap.get("recovery", np.nan)
        g2 = "—" if not np.isfinite(g2v) else ("**viable**" if g2v >= G2_MIN else "**not viable**")
        g4 = "—" if not (bf4 or bf8) else ("**finite** (round 3's gen:A divergence was the inconsistent filter)"
                                          if not (bf4.get("diverged") or bf8.get("diverged")) else "**diverged** (expanding transport)")
        L += [f"- n = {n}: P0 two-way/K4c3 = {_f(p0, 3)} (must be 1.00 ± 0.02); G1 one-way/K4c3 = {_f(g1r)} (≤ {G1_MAX}), "
              f"nonlinearity = {_f(nlv, 3)} (≤ {NONLIN_MAX}) → {g1}; G2 direct path + 4 generations recovery = {_f(g2v)} (≥ {G2_MIN}) → {g2}; "
              f"G4 bf=4/bf=8: {g4}, probe-norm growth per layer after the births stop = {_f(g.get('growth_bf4', np.nan), 3)} / {_f(g.get('growth_bf8', np.nan), 3)}."]
    # rank spectra
    rk = defaultdict(list)
    for r in rows_all:
        for d in r["v"].get("lr:oneway:rank", {}).get("rank", []) or []:
            if not d.get("error"):
                rk[(r["n"], d["what"], d["layer"])].append(d)
    if rk:
        L += ["", "Rank of the index-distinct bulk ((ab)(cd) unfolding; randomized SVD, top 256; means over seeds). "
              "'bulk' = share of the tensor's squared norm on index-distinct entries; 'top-k' = share of the bulk's energy in k directions; "
              "k90 = directions for 90 % (— = more than 256). G3 rule: k90 must grow < 1.3× from n = 128 to 192.", "",
              "| n | tensor | layer | bulk | top-16 | top-64 | top-256 | k50 | k90 |", "|---|---|---|---|---|---|---|---|---|"]
        out["rank"] = {}
        for (n, what, layer), ds in sorted(rk.items()):
            m = lambda k: float(np.mean([d[k] for d in ds if d.get(k) is not None])) if any(d.get(k) is not None for d in ds) else np.nan
            k90 = [d["k90"] for d in ds]
            k50 = [d["k50"] for d in ds]
            fk = lambda v: "—" if any(x is None for x in v) else f"{np.mean(v):.0f}"
            L.append(f"| {n} | {what} | {layer} | {_f(m('bulk_frac'))} | {_f(m('frac16'))} | {_f(m('frac64'))} | {_f(m('frac256'))} | {fk(k50)} | {fk(k90)} |")
            out["rank"][f"n{n}_{what}_l{layer}"] = dict(bulk=m("bulk_frac"), frac16=m("frac16"), frac64=m("frac64"), frac256=m("frac256"),
                                                         k50=k50, k90=k90)
    L.append("")
    return L, out


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "data/gpu/k4"
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "n*_L*_s*.json")))]
    lines, js = section(rows)
    print("\n".join(lines))
