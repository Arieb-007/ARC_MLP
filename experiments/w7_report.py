"""Aggregate data/w7/*.json -> results/<date>_w7_coherent_even.{md,json}. Written by an LLM agent."""
import glob
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arcmlp.eval.local_eval import provenance  # noqa: E402

rng = np.random.default_rng(0)
B = 10000


def boot_mean(x):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), (B, len(x)))
    s = x[idx].mean(1)
    return float(x.mean()), float(np.quantile(s, .025)), float(np.quantile(s, .975))


def boot_ratio_of_means(num, den):
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = rng.integers(0, len(num), (B, len(num)))
    s = num[idx].mean(1) / den[idx].mean(1)
    return float(num.mean() / den.mean()), float(np.quantile(s, .025)), float(np.quantile(s, .975))


data = defaultdict(list)
for p in glob.glob("data/w7/n*_L*_s*.json"):
    r = json.load(open(p))
    if "v" in r:
        data[(r["n"], r["L"])].append(r)
day = time.strftime("%Y-%m-%d", time.gmtime())
VAR = ["K3c2", "coh2", "ex22", "ex4c2", "exallc2", "K4c2", "K3c3", "coh3", "ex4c3", "exallc3", "K4c3", "coh2_k6", "coh2_k6_k8", "coh2_k3sq", "ex4c2_k6", "ex4c2_k6_k8"]
out = dict(cells={}, provenance=provenance("."))
L = [f"# W7 — coherent even-order cumulant terms ({day})", "",
     "> LLM-generated record (Claude Code). Runner `experiments/w7_coherent_even.py`, aggregation",
     "> `experiments/w7_report.py`, theory `docs/notes/w7_coherent_even_orders.md`. Weights: W ~ N(0, 2/n) from",
     "> `default_rng(seed)`; MC truth N = 4e6 per network, MC seed 70000 + 100 L + seed (cached in data/mc).", "",
     "**Falsifier (written before running).** W7 is not a win if at n = 128 '+κ4 coherent' recovers < 50% of the",
     "final-layer MSE gap between K=3 and dense '+all κ4 slices' at the same diagram cap (bootstrap CI over",
     "networks excluding 50%); κ6/κ8 are reported either way.", "",
     "MSE = debiased final-layer MSE (MC floor subtracted); mean over networks with 95% bootstrap CI.", ""]
for (n, Lr), rows in sorted(data.items()):
    rows = [r for r in rows if all(v in r["v"] for v in VAR if (v not in ("K4c2", "K4c3") or n <= 128))]
    if not rows:
        continue
    seeds = sorted(r["seed"] for r in rows)
    floor = np.mean([r["floor"] for r in rows])
    cell = dict(n=n, L=Lr, seeds=seeds, floor=floor, variants={})
    L += [f"## n = {n}, L = {Lr} ({len(rows)} networks, seeds {seeds[0]}–{seeds[-1]}; mean MC floor {floor:.1e})", "",
          "| variant | mean MSE [95% CI] | ratio to K3c2 [95% CI] | mean per-net SE |", "|---|---|---|---|"]
    base = [r["v"]["K3c2"]["mse"] for r in rows]
    for v in VAR:
        if v not in rows[0]["v"]:
            continue
        x = [r["v"][v]["mse"] for r in rows]
        m, lo, hi = boot_mean(x)
        rt, rlo, rhi = boot_ratio_of_means(x, base)
        se = float(np.mean([r["v"][v]["se"] for r in rows]))
        cell["variants"][v] = dict(mean=m, ci=[lo, hi], ratio=rt, ratio_ci=[rlo, rhi], se=se, per_seed=x)
        L.append(f"| {v} | {m:.3e} [{lo:.2e}, {hi:.2e}] | {rt:.3f} [{rlo:.3f}, {rhi:.3f}] | {se:.1e} |")
    for cap in (2, 3):
        k3, k4 = f"K3c{cap}", f"K4c{cap}"
        if k4 in rows[0]["v"]:
            gap = np.array([r["v"][k3]["mse"] - r["v"][k4]["mse"] for r in rows])
            for cand in (f"coh{cap}", f"ex4c{cap}", f"exallc{cap}"):
                rec = np.array([r["v"][k3]["mse"] - r["v"][cand]["mse"] for r in rows])
                f, flo, fhi = boot_ratio_of_means(rec, gap)
                cell[f"recovery_{cand}"] = dict(frac=f, ci=[flo, fhi], gap_mean=float(gap.mean()))
                L.append(f"\nGap recovery at cap {cap}: (K3 − {cand})/(K3 − K4 dense) = **{f:.2f}** [{flo:.2f}, {fhi:.2f}] "
                         f"(mean gap {gap.mean():.2e}).")
    for v in ("coh2_k6", "coh2_k6_k8", "coh2_k3sq", "ex4c2_k6", "ex4c2_k6_k8"):
        if v in rows[0]["v"] and "dmu_rms_final" in rows[0]["v"][v]:
            d = float(np.mean([r["v"][v]["dmu_rms_final"] for r in rows]))
            cell["variants"][v]["dmu_rms_final"] = d
            L.append(f"\n{v}: RMS of the added final-layer mean correction = {d:.2e} (its square {d*d:.1e} vs MC floor {floor:.1e}).")
    beta = np.array([[d["beta"] for d in r["v"]["coh2"]["diag"]] for r in rows])
    bbar = np.array([[d["bbar"] for d in r["v"]["coh2"]["diag"]] for r in rows])
    cell["beta_per_layer"] = beta.mean(0).tolist(); cell["bbar_per_layer"] = bbar.mean(0).tolist()
    L += ["", "Per-layer coherent κ4 constant β_l = 4 b̄/n + 4 B̄ (z-side, mean over networks) and b̄_l:", "",
          "| layer | " + " | ".join(str(i) for i in range(beta.shape[1])) + " |",
          "|---|" + "---|" * beta.shape[1],
          "| β | " + " | ".join(f"{x:.2e}" for x in beta.mean(0)) + " |",
          "| b̄ | " + " | ".join(f"{x:.3f}" for x in bbar.mean(0)) + " |", ""]
    out["cells"][f"n{n}_L{Lr}"] = cell
# scaling fits
L += ["## Scaling MSE(n) ∝ n^{-α} (per variant and depth; fit of log mean MSE on log n; bootstrap over networks)", "",
      "| L | variant | widths | α [95% CI] |", "|---|---|---|---|"]
for Lr in (8, 16):
    for v in VAR:
        ns = sorted(n for (n, l) in data if l == Lr and f"n{n}_L{Lr}" in out["cells"] and v in out["cells"][f"n{n}_L{Lr}"]["variants"])
        if len(ns) < 2:
            continue
        per = [np.array(out["cells"][f"n{n}_L{Lr}"]["variants"][v]["per_seed"]) for n in ns]
        def fit(samples):
            y = np.log([max(s.mean(), 1e-300) for s in samples]); x = np.log(ns)
            return -np.polyfit(x, y, 1)[0]
        a0 = fit(per)
        boots = []
        for _ in range(2000):
            boots.append(fit([s[rng.integers(0, len(s), len(s))] for s in per]))
        lo, hi = np.quantile(boots, [.025, .975])
        out.setdefault("alpha", {})[f"L{Lr}_{v}"] = dict(alpha=a0, ci=[float(lo), float(hi)], widths=ns)
        L.append(f"| {Lr} | {v} | {ns} | {a0:.2f} [{lo:.2f}, {hi:.2f}] |")
json.dump(out, open(f"results/{day}_w7_coherent_even.json", "w"), indent=1, default=float)
open(f"results/{day}_w7_coherent_even.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
