"""Aggregate data/w5/*.json -> results/<date>_w5_selection.{md,json} + Pareto plot. Written by an LLM agent.

Per group g (see experiments/w5_selection.py):
  LOO delta  = MSE(full - g) - MSE(full)   (> 0: the group helps)
  ADD delta  = MSE(base) - MSE(base + g)   (> 0: the group helps on top of K=2)
Runs whose MSE exceeds 1e-2 are classed as DIVERGED (structurally required group); deltas are
summarised by medians and bootstrap CIs over networks; the minimum detectable effect (MDE) of a
paired mean difference is 2.8 sd/sqrt(n_networks). Rank stability between tiers/widths: Spearman
over groups common to both, using mean LOO deltas (diverged = +inf rank top).
Knapsack: choose groups to maximise the sum of ADD deltas subject to billed cost <= 100 units
(cost_model.selection_cost over the union of the chosen groups' terms), exhaustive over subsets of
the groups with positive ADD delta (<= 2^16), with and without the O(1/n)-completeness constraint.
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
from arcmlp.compress.cost_model import selection_cost  # noqa: E402
from arcmlp.eval.local_eval import provenance  # noqa: E402
from arcmlp.reference import dense_cumulants as dc  # noqa: E402
from w5_selection import group_of  # noqa: E402

DIVERGE = 1e-2
rng = np.random.default_rng(0)
day = time.strftime("%Y-%m-%d", time.gmtime())


def boot(x):
    x = np.asarray(x, float)
    s = x[rng.integers(0, len(x), (10000, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(s, .025)), float(np.quantile(s, .975))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


tiers = defaultdict(list)
for p in glob.glob("data/w5/*.json"):
    r = json.load(open(p))
    tiers[(r["tier"], r["n"], r["L"])].append(r)
reg = dc.build_registry(4, 3)
gterms = defaultdict(list)
for t in reg:
    g = group_of(t)
    if g:
        gterms[g].append(t)
backbone = [t for t in reg if group_of(t) is None]
out = dict(tiers={}, provenance=provenance("."))
L = [f"# W5 — measured, cost-aware diagram selection ({day})", "",
     "> LLM-generated record (Claude Code). Runner `experiments/w5_selection.py`, aggregation `experiments/w5_report.py`,",
     "> registry `results/2026-10-17_w5_term_registry.md`, cost model `docs/notes/w5_cost_model.md` (a hypothesis).", "",
     "**Falsifier (written before running).** The selection does not transfer across widths if the Spearman rank",
     "correlation of leave-one-out deltas between the two measured widths is < 0.7; then the knapsack at n = 1024 is",
     "not worth running.", ""]
summ = {}
for key, rows in sorted(tiers.items()):
    tier, n, Lr = key
    rows = [r for r in rows if "full" in r["runs"] and "base" in r["runs"]]
    if not rows:
        continue
    groups = sorted({k.split(":", 1)[1] for r in rows for k in r["runs"] if k.startswith("loo:")})
    full = np.array([r["runs"]["full"]["mse"] for r in rows]); base = np.array([r["runs"]["base"]["mse"] for r in rows])
    L += [f"## Tier {tier}, n = {n}, L = {Lr}: {len(rows)} networks (seeds {sorted(r['seed'] for r in rows)})", "",
          f"full set MSE {boot(full)[0]:.3e} [{boot(full)[1]:.2e}, {boot(full)[2]:.2e}]; backbone (K=2) MSE {boot(base)[0]:.3e}; "
          f"mean MC floor {np.mean([r['floor'] for r in rows]):.1e}.", "",
          "| group | terms | LOO Δ mean [95% CI] | LOO diverged | ADD Δ mean [95% CI] | ADD diverged | MDE (LOO) | billed units alone |",
          "|---|---|---|---|---|---|---|---|"]
    tinfo = {}
    for g in groups:
        loo = np.array([r["runs"].get("loo:" + g, {}).get("mse", np.nan) for r in rows])
        add = np.array([r["runs"].get("add:" + g, {}).get("mse", np.nan) for r in rows])
        dl, da = loo > DIVERGE, add > DIVERGE
        okl, oka = ~dl & ~np.isnan(loo), ~da & ~np.isnan(add)
        dloo = loo[okl] - full[okl]; dadd = base[oka] - add[oka]
        bl = boot(dloo) if okl.sum() > 1 else (np.nan,) * 3
        ba = boot(dadd) if oka.sum() > 1 else (np.nan,) * 3
        mde = 2.8 * float(np.std(dloo, ddof=1)) / np.sqrt(max(okl.sum(), 1)) if okl.sum() > 1 else np.nan
        cost = selection_cost(backbone + gterms.get(g, []))[0]
        tinfo[g] = dict(loo=bl, loo_div=int(dl.sum()), add=ba, add_div=int(da.sum()), mde=mde, cost=cost,
                        nterms=len(gterms.get(g, [])), loo_per_seed=dloo.tolist(), add_per_seed=dadd.tolist())
        L.append(f"| {g} | {len(gterms.get(g, []))} | {bl[0]:+.2e} [{bl[1]:+.1e}, {bl[2]:+.1e}] | {int(dl.sum())}/{len(rows)} | "
                 f"{ba[0]:+.2e} [{ba[1]:+.1e}, {ba[2]:+.1e}] | {int(da.sum())}/{len(rows)} | {mde:.1e} | {cost:.0f} |")
    summ[key] = tinfo
    out["tiers"][f"{tier}_n{n}_L{Lr}"] = dict(groups=tinfo, full=full.tolist(), base=base.tolist())
    L.append("")
# rank stability
keys = sorted(summ)
L += ["## Rank stability across widths (Spearman of mean LOO Δ over common groups; diverged ranked first)", ""]
for a_, b_ in itertools.combinations(keys, 2):
    common = sorted(set(summ[a_]) & set(summ[b_]))
    if len(common) < 4:
        continue
    va = [np.inf if summ[a_][g]["loo_div"] else summ[a_][g]["loo"][0] for g in common]
    vb = [np.inf if summ[b_][g]["loo_div"] else summ[b_][g]["loo"][0] for g in common]
    va = np.nan_to_num(np.array(va, float), posinf=1e9); vb = np.nan_to_num(np.array(vb, float), posinf=1e9)
    rho = spearman(va, vb)
    out.setdefault("spearman", {})[f"{a_}|{b_}"] = dict(rho=rho, groups=common)
    L.append(f"- {a_} vs {b_}: ρ = **{rho:.2f}** over {len(common)} groups")
L.append("")
json.dump(out, open(f"results/{day}_w5_selection.json", "w"), indent=1, default=float)
open(f"results/{day}_w5_selection.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
