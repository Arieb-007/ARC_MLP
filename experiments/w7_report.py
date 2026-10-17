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
    if "K4c3" in rows[0]["v"]:
        gap = np.array([r["v"]["K3c2"]["mse"] - r["v"]["K4c3"]["mse"] for r in rows])
        L.append(f"\nRecovery of the gap between the standard K=3 (cap 2) and the paper's K=4 (dense, cap 3), mean gap {gap.mean():.2e}:")
        for cand in ("coh2", "ex22", "ex4c2", "exallc2", "K4c2", "K3c3", "coh3", "ex4c3", "exallc3"):
            rec = np.array([r["v"]["K3c2"]["mse"] - r["v"][cand]["mse"] for r in rows])
            f, flo, fhi = boot_ratio_of_means(rec, gap)
            cell[f"xrecovery_{cand}"] = dict(frac=f, ci=[flo, fhi])
            L.append(f"(K3c2 − {cand})/(K3c2 − K4c3) = {f:.2f} [{flo:.2f}, {fhi:.2f}]; ")
        nd = sum(r["v"]["K4c3"]["mse"] > 1e-2 for r in rows)
        nw = sum(r["v"]["K4c3"]["mse"] > r["v"]["K3c2"]["mse"] for r in rows)
        L.append(f"\nK4c3 worse than K3c2 on {nw}/{len(rows)} networks; MSE > 1e-2 (diverged) on {nd}/{len(rows)}.")
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
L += ["## Scaling MSE(n) ∝ n^{-α} (per variant and depth; fit of log mean MSE on log n over n ≥ 64 — n = 32 is",
      "excluded because the K=4 truncations are unstable there; bootstrap over networks)", "",
      "| L | variant | widths | α [95% CI] |", "|---|---|---|---|"]
for Lr in (8, 16):
    for v in VAR:
        ns = sorted(n for (n, l) in data if l == Lr and n >= 64 and f"n{n}_L{Lr}" in out["cells"] and v in out["cells"][f"n{n}_L{Lr}"]["variants"])
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
# decomposition inside the dense reference (experiments/w7_decomposition.py)
if os.path.exists("data/w7_decomp.json"):
    dec = json.load(open("data/w7_decomp.json"))
    out["decomposition"] = dec
    byc = defaultdict(list)
    for k, r in dec.items():
        byc[(r["n"], r["L"])].append(r)
    L += ["## Decomposition: coherent replacement of one κ4 slice inside the dense reference", "",
          "Slice hooks replace the pre-activation (4,) slice by W⁴b + 12B̄ and/or the (2,2) slice by 4b̄/n + 4B̄",
          "(b, B from the dense post-ReLU κ4), or zero the random-sign slices (3,1), (2,1,1), (1,1,1,1) (cap 3).",
          "Falsifier (written before running) for '(2,2) rank-one is adequate, (4,) replacement is what fails':",
          "replacing only (2,2) moves the MSE more than replacing only (4,).", ""]
    names = ["c2_ex_ex", "c2_ex4_coh22", "c2_coh4_ex22", "c2_coh_coh", "c3_full", "c3_ex4_coh22", "c3_coh_coh", "c3_rs0"]
    for (n, Lr), rs in sorted(byc.items()):
        rs = [r for r in rs if all(v in r["v"] for v in names)]
        if not rs:
            continue
        L += [f"n = {n}, L = {Lr}, seeds {sorted(r['seed'] for r in rs)} — mean MSE (ratio of means to exact at the same cap):", "",
              "| variant | mean MSE | ratio to exact |", "|---|---|---|"]
        for v in names:
            x = [r["v"][v]["mse"] for r in rs]
            ref = [r["v"]["c2_ex_ex" if v.startswith("c2") else "c3_full"]["mse"] for r in rs]
            rt = boot_ratio_of_means(x, ref)
            L.append(f"| {v} | {np.mean(x):.3e} | {rt[0]:.2f} [{rt[1]:.2f}, {rt[2]:.2f}] |")
        L.append("")
# n = 1024 newborn coherent constants vs V29 lambda (experiments/w7_v29_compare.py)
if os.path.exists("data/w7_v29_compare.json"):
    cmp_ = json.load(open("data/w7_v29_compare.json"))
    rows_ = cmp_["rows"]; nl = len(rows_[0]["per_layer"])
    mean = lambda k: [float(np.mean([r["per_layer"][l][k] for r in rows_])) for l in range(nl)]
    out["v29_compare"] = dict(bbar=mean("bbar"), Bbar=mean("Bbar"), beta=mean("beta"), lam=mean("lam_v29"),
                              ratio=mean("ratio_lam_over_beta"), frac_b_neg=mean("frac_b_neg"), provenance=cmp_["provenance"])
    L += ["## n = 1024: newborn coherent κ4 constant β_l vs V29's fitted λ_l (4 self-baked MLPs, seeds 0–3)", "",
          "β_l = 4 b̄/n + 4 B̄ from the Gaussian-input (cap 2) newborn κ4 of each post-ReLU layer along the K=2 closure;",
          "λ_l = V29's table × 0.95. Falsifier for 'β is V29's λ': λ/β varies by > 2× across layers.", "",
          "| layer | " + " | ".join(str(l) for l in range(nl)) + " |", "|---|" + "---|" * nl,
          "| β | " + " | ".join(f"{x:.1e}" for x in mean("beta")) + " |",
          "| λ (V29) | " + " | ".join(f"{x:.1e}" for x in mean("lam_v29")) + " |",
          "| λ/β | " + " | ".join(f"{x:.3g}" for x in mean("ratio_lam_over_beta")) + " |",
          "| frac b<0 | " + " | ".join(f"{x:.2f}" for x in mean("frac_b_neg")) + " |", "",
          f"Outcome: falsified — λ/β grows from {mean('ratio_lam_over_beta')[0]:.2f} to {mean('ratio_lam_over_beta')[-1]:.0f}: the newborn",
          "β decays ~10³× over 16 layers (a growing fraction of neurons have t > 0.76, where κ4 < 0, so b̄ cancels)",
          "while λ is flat; V29's λ multiplies C_off in a partial-trace core and is not the coherent κ4 of z.", ""]
json.dump(out, open(f"results/{day}_w7_coherent_even.json", "w"), indent=1, default=float)
open(f"results/{day}_w7_coherent_even.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
