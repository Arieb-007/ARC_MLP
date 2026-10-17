"""Aggregate data/w5/*.json -> results/<date>_w5_selection.{md,json} + Pareto plot. Written by an LLM agent.

Feature sets are runs of experiments/w5_selection.py (features = input slices read, see its docstring).
Per feature f and cell (tier, n, L), over networks:
  LOO  = log MSE(F - f) - log MSE(F)       (> 0: f helps on top of everything else)
  ADD  = log MSE(base) - log MSE(base + f) (> 0: f helps on top of the K=2 closure)
Effects are per-network log ratios (MSE spans orders of magnitude across networks); means with 95%
bootstrap CIs (10^4 resamples of networks); runs with MSE > 1e-2 are counted as DIVERGED; the minimum
detectable effect is MDE = 2.8 sd / sqrt(#networks) (80% power, two-sided 5%).
Rank stability: Spearman rho of mean LOO between two cells over their common features, with a
bootstrap CI (networks resampled independently in each cell).
Budgeted selection: every feature subset whose billed cost (arcmlp/compress/cost_model.feature_cost)
is <= 100 units is measured directly (mode 'exhaustive'), so the 'knapsack' is solved exactly on the
measured networks; an additive surrogate (sum of ADD effects) is reported for comparison.
"""
import glob
import itertools
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arcmlp.compress.cost_model import feature_cost  # noqa: E402
from arcmlp.eval.local_eval import provenance  # noqa: E402
from w5_selection import FEATURES, K3F, key  # noqa: E402

DIVERGE = 1e-2
BUDGET = 100.0
rng = np.random.default_rng(0)
day = time.strftime("%Y-%m-%d", time.gmtime())
# public points at n = 1024, L = 16 [external: leaderboard / 504aldo ledger, see results/STATUS.md]
KIT = 3.9e-6
PUBLIC = [("starter-kit K=2 (billed at the 0.1 floor)", 100, 1.0), ("V29 (504aldo) raw", 260, 2.13e-8 / KIT),
          ("leaders, raw (range)", 100, 1.2e-8 / KIT), ("leaders, raw (range)", 130, 1.4e-8 / KIT)]


def boot(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float(np.mean(x)) if len(x) else np.nan, np.nan, np.nan)
    s = x[rng.integers(0, len(x), (10000, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(s, .025)), float(np.quantile(s, .975))


def rank(a):
    return np.argsort(np.argsort(a)).astype(float)


def spearman(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def lmse(r, feats):
    v = r["runs"].get(key(feats))
    return None if v is None else np.log(max(v["mse"], 1e-14))


cells = defaultdict(list)
for p in sorted(glob.glob("data/w5/*.json")):
    r = json.load(open(p))
    cells[(r["tier"], r["n"], r["L"])].append(r)

out = dict(cells={}, spearman={}, provenance=provenance("."))
L = [f"# W5 — measured, cost-aware diagram selection ({day})", "",
     "> LLM-generated record (Claude Code) for team Arieb-007. Runner `experiments/w5_selection.py`, aggregation",
     "> `experiments/w5_report.py`, registry `results/2026-10-17_w5_term_registry.md`, cost model",
     "> `docs/notes/w5_cost_model.md` (**a hypothesis** calibrated on one external point, V29).",
     "> Weights W ~ N(0, 2/n) from `default_rng(seed)`; MC truth N = 4e6 per network, MC seed 70000 + 100 L + seed",
     "> (cached in data/mc); MSE = debiased final-layer MSE (MC floor subtracted).", "",
     "**Falsifier (written before running).** The selection does not transfer across widths if the Spearman rank",
     "correlation of leave-one-out effects between n = 128 and n = 256 (light tier, L = 16) is < 0.7; then a",
     "selection made at small width is not evidence for n = 1024.", "",
     "**Design change (recorded before the runs below).** The first design (groups of terms by output order and",
     "input family, dense K=4 cap-3 tier at n = 64, L = 16) was abandoned after two networks: (i) a group that",
     "only WRITES state is inert when added to the K=2 base (ADD effect identically 0), and (ii) the full dense",
     "K=4 cap-3 set is itself unstable at n = 64, L = 16 (MSE 8.1e-3 on seed 0 vs 1.6e-3 for K=2; removing",
     "groups gave MSE up to 3e4). Raw data kept in data/w5_groups_v1/. Features are now the input slices a",
     "term READS (a term is on iff all slices it reads are on); the dense tier is run at L = 8, where the",
     "full set is stable; the transfer test uses the light tier (K ≤ 3 dense + coherent κ4), which is what",
     "an n = 1024 implementation could afford.", "",
     "Features: `Cpair` = covariance-only diagrams with ≥ 2 edges on mean/covariance (c², c³ corrections);",
     "`K3(s)` / `K4(s)` = terms reading that cumulant slice of the pre-activation (light tier: K4 slices are the",
     "coherent O(n²) values); `cap3` = all cost-3 (order n^-3/2) diagrams. All on = DenseProp(4, cap 3) or",
     "CoherentEvenProp('coherent', cap 3); all off = K=2 (tests/test_w5.py checks these identities exactly).", ""]

summ = {}
for (tier, n, Lr), rows in sorted(cells.items()):
    F = FEATURES[tier]
    rows = [r for r in rows if key(F) in r["runs"] and key([]) in r["runs"]]
    if len(rows) < 2:
        continue
    seeds = sorted(r["seed"] for r in rows)
    full = np.array([r["runs"][key(F)]["mse"] for r in rows])
    base = np.array([r["runs"][key([])]["mse"] for r in rows])
    cell = dict(tier=tier, n=n, L=Lr, seeds=seeds, full=full.tolist(), base=base.tolist(), features={})
    L += [f"## Tier {tier}, n = {n}, L = {Lr}: {len(rows)} networks (seeds {seeds[0]}–{seeds[-1]})", "",
          f"All features: MSE {boot(full)[0]:.3e} [{boot(full)[1]:.2e}, {boot(full)[2]:.2e}] ({int((full > DIVERGE).sum())} diverged); "
          f"K=2 base: {boot(base)[0]:.3e} [{boot(base)[1]:.2e}, {boot(base)[2]:.2e}]; mean log(full/base) = "
          f"{boot(np.log(full / base))[0]:+.2f}; mean MC floor {np.mean([r['floor'] for r in rows]):.1e}.", "",
          "| feature | LOO mean [95% CI] | MDE | LOO div. | ADD mean [95% CI] | ADD div. | billed units of base+f |",
          "|---|---|---|---|---|---|---|"]
    for f in F:
        loo, add, dl, da = [], [], 0, 0
        for r in rows:
            a, b = lmse(r, set(F) - {f}), lmse(r, {f})
            if a is not None:
                loo.append(a - np.log(r["runs"][key(F)]["mse"])); dl += r["runs"][key(set(F) - {f})]["mse"] > DIVERGE
            if b is not None:
                add.append(np.log(r["runs"][key([])]["mse"]) - b); da += r["runs"][key({f})]["mse"] > DIVERGE
        bl, ba = boot(loo), boot(add)
        mde = 2.8 * float(np.std(loo, ddof=1)) / np.sqrt(len(loo)) if len(loo) > 1 else np.nan
        cost = feature_cost({f}, tier)
        cell["features"][f] = dict(loo=bl, add=ba, loo_div=int(dl), add_div=int(da), mde=mde, cost_alone=cost,
                                   loo_per_seed=list(map(float, loo)), add_per_seed=list(map(float, add)))
        L.append(f"| {f} | {bl[0]:+.3f} [{bl[1]:+.3f}, {bl[2]:+.3f}] | {mde:.3f} | {dl} | {ba[0]:+.3f} [{ba[1]:+.3f}, {ba[2]:+.3f}] | {da} | {cost:.0f} |")
    L += ["", "(Effects are natural-log ratios of MSE: +0.69 = the feature halves the error.)", ""]
    inert = [f for f in F if cell["features"][f]["add_per_seed"] and np.allclose(cell["features"][f]["add_per_seed"], 0, atol=1e-12)]
    if inert:
        L += [f"ADD ≡ 0 exactly for {', '.join(inert)}: these slices are read only by terms that write κ3/κ4 state, which",
              "nothing in the K=2 base reads — they are inert alone (structural zero), not useless.", ""]
    # factorial main effects (full 2^|F| lattice measured on some networks)
    allsets = [set(c) for r_ in range(len(F) + 1) for c in itertools.combinations(F, r_)]
    lat = [r for r in rows if all(key(S) in r["runs"] for S in allsets)]
    if len(lat) >= 2:
        me = {}
        for f in F:
            per = []
            for r in lat:
                d = [np.log(max(r["runs"][key(S)]["mse"], 1e-14)) - np.log(max(r["runs"][key(S | {f})]["mse"], 1e-14))
                     for S in allsets if f not in S]
                per.append(float(np.mean(d)))
            me[f] = dict(main=boot(per), per_seed=per)
        inter = []
        for f, g in itertools.combinations(F, 2):
            per = []
            for r in lat:
                lm = lambda S: np.log(max(r["runs"][key(S)]["mse"], 1e-14))
                per.append(float(np.mean([lm(S) - lm(S | {f}) - lm(S | {g}) + lm(S | {f, g}) for S in allsets if f not in S and g not in S])))
            inter.append((boot(per), f, g))
        inter.sort(key=lambda x: -abs(x[0][0]))
        cell["main_effects"] = me
        cell["interactions"] = [dict(f=f, g=g, ci=b) for b, f, g in inter]
        L += [f"### Full factorial (2^{len(F)} = {len(allsets)} feature sets) on {len(lat)} networks (seeds {sorted(r['seed'] for r in lat)})", "",
              "Main effect = mean over all contexts S ∌ f of log MSE(S) − log MSE(S ∪ f) (> 0: f helps on average).", "",
              "| feature | main effect [95% CI] |", "|---|---|"]
        for f in sorted(F, key=lambda f: -me[f]["main"][0]):
            b = me[f]["main"]
            L.append(f"| {f} | {b[0]:+.3f} [{b[1]:+.3f}, {b[2]:+.3f}] |")
        L += ["", "Largest two-feature interactions (mean over contexts of lm(S) − lm(S+f) − lm(S+g) + lm(S+f+g); < 0: the pair",
              "helps more than the sum of its parts):", ""]
        for b, f, g in inter[:5]:
            L.append(f"- {f} × {g}: {b[0]:+.3f} [{b[1]:+.3f}, {b[2]:+.3f}]")
        L.append("")
    # exhaustive within budget
    inb = [S for S in allsets if feature_cost(S, tier) <= BUDGET]
    exr = [r for r in rows if all(key(S) in r["runs"] for S in inb)]
    if len(exr) >= 2 and len(inb) > 2 * len(F) + 2:
        lb = {r["seed"]: np.log(r["runs"][key([])]["mse"]) for r in exr}
        add_ex = {f: float(np.mean([lb[r["seed"]] - np.log(max(r["runs"][key({f})]["mse"], 1e-14)) for r in exr])) for f in F}
        tab = []
        for S in inb:
            gain = [lb[r["seed"]] - np.log(max(r["runs"][key(S)]["mse"], 1e-14)) for r in exr]
            tab.append((boot(gain), S, feature_cost(S, tier), float(np.mean([r["runs"][key(S)]["mse"] for r in exr])),
                        sum(add_ex[f] for f in S)))
        tab.sort(key=lambda x: -x[0][0])
        best_sur = max(tab, key=lambda x: (x[4], -len(x[1])))
        cell["exhaustive"] = [dict(feats=sorted(t[1]), gain=t[0], cost=t[2], mean_mse=t[3], surrogate=t[4]) for t in tab]
        cell["exhaustive_seeds"] = sorted(r["seed"] for r in exr)
        L += [f"### Budgeted selection (billed cost ≤ {BUDGET:.0f} units at n = 1024, L = 16): all {len(tab)} feature sets that fit, measured on {len(exr)} networks (seeds {cell['exhaustive_seeds'][0]}–{cell['exhaustive_seeds'][-1]})", "",
              "| rank | features on | billed units | mean MSE | gain vs K=2: mean log ratio [95% CI] | additive surrogate (sum of ADD) |", "|---|---|---|---|---|---|"]
        for i, t in enumerate(tab[:10]):
            L.append(f"| {i+1} | {', '.join(sorted(t[1])) or '(K=2)'} | {t[2]:.0f} | {t[3]:.3e} | {t[0][0]:+.3f} [{t[0][1]:+.3f}, {t[0][2]:+.3f}] | {t[4]:+.3f} |")
        ref = {}
        for nm, S in [("K=3 (cap 2) = Cpair+K3 all", {"Cpair"} | set(K3F)), ("O(1/n)-complete (all but cap3)", set(F) - {"cap3"}), ("all features", set(F))]:
            if all(key(S) in r["runs"] for r in exr):
                g_ = boot([lb[r["seed"]] - np.log(max(r["runs"][key(S)]["mse"], 1e-14)) for r in exr])
                ref[nm] = (g_, feature_cost(S, tier))
                L.append(f"| ref | {nm} | {feature_cost(S, tier):.0f} | {np.mean([r['runs'][key(S)]['mse'] for r in exr]):.3e} | {g_[0]:+.3f} [{g_[1]:+.3f}, {g_[2]:+.3f}] | |")
        L += ["", f"Exact optimum within budget on these networks: **{', '.join(sorted(tab[0][1]))}** ({tab[0][2]:.0f} units). "
              f"Additive-surrogate optimum (ADD effects assumed additive): {', '.join(sorted(best_sur[1])) or '(K=2)'} "
              f"(measured gain {best_sur[0][0]:+.3f}).",
              "O(1/n)-completeness (every cost ≤ 2 diagram, i.e. all features except `cap3`) needs `K3(2,1)` and hence the",
              f"k3_21 family: billed {feature_cost(set(F) - {'cap3'}, tier):.0f} units > {BUDGET:.0f}, so the constrained knapsack is",
              "**infeasible** under this cost model.", ""]
    summ[(tier, n, Lr)] = cell
    out["cells"][f"{tier}_n{n}_L{Lr}"] = cell

# rank stability
L += ["## Rank stability (Spearman ρ of mean LOO effects over common features; bootstrap 95% CI)", ""]
keys = sorted(summ)
for a_, b_ in itertools.combinations(keys, 2):
    common = [f for f in summ[a_]["features"] if f in summ[b_]["features"]]
    if len(common) < 4:
        continue
    va = np.array([summ[a_]["features"][f]["loo"][0] for f in common])
    vb = np.array([summ[b_]["features"][f]["loo"][0] for f in common])
    rho = spearman(va, vb)
    pa = [np.array(summ[a_]["features"][f]["loo_per_seed"]) for f in common]
    pb = [np.array(summ[b_]["features"][f]["loo_per_seed"]) for f in common]
    na, nb = min(len(x) for x in pa), min(len(x) for x in pb)
    bs = []
    for _ in range(2000):
        ia, ib = rng.integers(0, na, na), rng.integers(0, nb, nb)
        bs.append(spearman([x[:na][ia].mean() for x in pa], [x[:nb][ib].mean() for x in pb]))
    lo, hi = np.nanquantile(bs, [.025, .975])
    out["spearman"][f"{a_}|{b_}"] = dict(rho=rho, ci=[float(lo), float(hi)], features=common)
    L.append(f"- {a_[0]} n={a_[1]} L={a_[2]} vs {b_[0]} n={b_[1]} L={b_[2]}: ρ = **{rho:.2f}** [{lo:.2f}, {hi:.2f}] over {len(common)} features")
L.append("")

# fresh-seed verification (data/w5_verify; pre-registered in docs/notes/w5_prereg_verification.md)
ver = defaultdict(list)
for p_ in sorted(glob.glob("data/w5_verify/*.json")):
    r = json.load(open(p_))
    ver[(r["tier"], r["n"], r["L"])].append(r)
for (tier, n, Lr), rows in sorted(ver.items()):
    rows = [r for r in rows if key([]) in r["runs"]]
    if len(rows) < 2:
        continue
    names = {key([]): "K=2", key({"Cpair", "K3(3,)", "K4(4,)", "K4(2,2)"}): "selected (≤100 units)",
             key({"Cpair", "K3(3,)"}): "Cpair+K3(3,)", key({"Cpair"} | set(K3F)): "K=3 cap 2",
             key({"Cpair", "K4(4,)", "K4(2,2)"} | set(K3F)): "O(1/n)-complete light"}
    common = [k for k in names if all(k in r["runs"] for r in rows)]
    L += [f"## Verification on fresh networks: tier {tier}, n = {n}, L = {Lr}, seeds {sorted(r['seed'] for r in rows)}", "",
          "Pre-registered in `docs/notes/w5_prereg_verification.md` before these runs.", "",
          "| config | billed units | mean MSE | gain vs K=2: mean log ratio [95% CI] | wins vs K=2 |", "|---|---|---|---|---|"]
    vres = {}
    for k in common:
        g_ = [np.log(r["runs"][key([])]["mse"]) - np.log(max(r["runs"][k]["mse"], 1e-14)) for r in rows]
        feats = set(rows[0]["runs"][k]["feats"])
        vres[names[k]] = dict(gain=boot(g_), per_seed=g_, cost=feature_cost(feats, tier), mean_mse=float(np.mean([r["runs"][k]["mse"] for r in rows])))
        L.append(f"| {names[k]} | {feature_cost(feats, tier):.0f} | {vres[names[k]]['mean_mse']:.3e} | {vres[names[k]]['gain'][0]:+.3f} "
                 f"[{vres[names[k]]['gain'][1]:+.3f}, {vres[names[k]]['gain'][2]:+.3f}] | {sum(x > 0 for x in g_)}/{len(g_)} |")
    if "selected (≤100 units)" in vres and "O(1/n)-complete light" in vres:
        sel, comp = vres["selected (≤100 units)"], vres["O(1/n)-complete light"]
        diff = boot(np.array(comp["per_seed"]) - np.array(sel["per_seed"]))
        p1 = sel["gain"][1] > 0
        p2 = diff[0] > 1.0
        p3 = sel["gain"][0] < 0.5
        L += ["", f"Prediction 1 (selected beats K=2, CI excludes 0): **{'confirmed' if p1 else 'falsified'}** (gain {sel['gain'][0]:+.3f} [{sel['gain'][1]:+.3f}, {sel['gain'][2]:+.3f}]).",
              f"Prediction 2 (O(1/n)-complete beats selected by > 1.0): **{'confirmed' if p2 else 'falsified'}** (difference {diff[0]:+.3f} [{diff[1]:+.3f}, {diff[2]:+.3f}]).",
              f"Prediction 3 (selected gain < 0.5): **{'confirmed' if p3 else 'falsified'}**.", ""]
        vres["predictions"] = dict(p1=p1, p2=p2, p3=p3, diff=diff)
    out.setdefault("verification", {})[f"{tier}_n{n}_L{Lr}"] = vres

# Pareto plot
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    mk = {64: "s", 128: "o", 256: "^"}
    for (tier, n, Lr), cell in sorted(summ.items()):
        if tier != "light" or Lr != 16:
            continue
        pts = []
        rows = cells[(tier, n, Lr)]
        base = {r["seed"]: r["runs"][key([])]["mse"] for r in rows if key([]) in r["runs"]}
        allk = defaultdict(list)
        for r in rows:
            for k, v in r["runs"].items():
                if r["seed"] in base:
                    allk[k].append((v["mse"] / base[r["seed"]], tuple(v["feats"])))
        nmax = max(len(v) for v in allk.values())
        for k, v in allk.items():
            if len(v) == nmax:
                pts.append((feature_cost(set(v[0][1]), tier), float(np.exp(np.mean(np.log([x[0] for x in v]))))))
        pts = np.array(pts)
        ax.scatter(pts[:, 0], pts[:, 1], marker=mk.get(n, "x"), s=18, alpha=.6, label=f"measured, light tier, n={n}, L=16")
    for lab, c, y in PUBLIC:
        ax.scatter([c], [y], marker="*", s=90, color="k")
        ax.annotate(lab, (c, y), fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax.axvline(BUDGET, ls="--", color="gray", lw=.8)
    ax.set_yscale("log"); ax.set_xlabel("billed units at n = 1024, L = 16 (cost model, hypothesis)")
    ax.set_ylabel("MSE / MSE(K=2) (geo-mean over networks)")
    ax.set_title("W5 cost vs error (small-width points are NOT n = 1024 errors)", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    os.makedirs("docs/writeup/figures", exist_ok=True)
    fig.tight_layout(); fig.savefig(f"docs/writeup/figures/w5_pareto.png", dpi=150)
    L += ["![Pareto](../docs/writeup/figures/w5_pareto.png)", "",
          "Pareto plot: measured feature sets (light tier, L = 16) at their predicted n = 1024 billed cost, y = MSE relative",
          "to K=2 at the same width; stars = public n = 1024 points relative to the starter kit's raw MSE 3.9e-6. The",
          "two y-scales are not comparable (the ratio to K=2 of a higher-order method shrinks with n); the plot shows",
          "the ordering of costs, not an n = 1024 prediction.", ""]
except Exception as e:  # pragma: no cover
    L += [f"(Pareto plot failed: {e})", ""]

json.dump(out, open(f"results/{day}_w5_selection.json", "w"), indent=1, default=float)
open(f"results/{day}_w5_selection.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
