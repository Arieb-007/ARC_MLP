"""Aggregate every stage of scripts/gpu_run.sh into results/<date>_gpu_run.{md,json}.
Written by an LLM agent (Claude Code). All numbers are measured in this run; predictions written
before the run are quoted with their source and marked confirmed / falsified.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, ROOT, boot_mean, paired_log_ratio, read_json  # noqa: E402

D = DATA
PRED = {  # pre-registered before the run (plan of 2026-10-17; alpha fits of results/2026-10-17_w7_coherent_even.md)
    "K3c2": dict(point=2.8e-7, lo=1.9e-7, hi=4.2e-7, text="dense K=3 cap 2 at 1024x16 = 2.8e-7 [1.9e-7, 4.2e-7] (n^-2.29 from n = 256, L = 16)"),
    "coh2": dict(point=1.5e-8, lo=0.9e-8, hi=2.6e-8, text="K=3 + coherent kappa4 at 1024x16 = 1.5e-8 [0.9e-8, 2.6e-8] (n^-3.71), i.e. below V29's 2.13e-8"),
}
K4_KILL = 0.25


def fmt(x, p=3):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{p}e}"


def rows_of(pattern):
    return [r for r in (read_json(p) for p in sorted(glob.glob(pattern))) if r]


def main():
    day = time.strftime("%Y-%m-%d", time.gmtime())
    out = dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), env=read_json(os.path.join(D, "env.json")),
               status=read_json(os.path.join(D, "status.json"), {}), meta=read_json(os.path.join(D, "mlps", "meta.json")))
    L = [f"# GPU run — V29 baseline, dense references at 1024×16, κ4 go/no-go, knob sweep, corrector ({day})", "",
         "> LLM-generated record (Claude Code) for team Arieb-007. Produced end-to-end by `scripts/gpu_run.sh`",
         "> (stages in `experiments/gpu/`). Truth = the dataset's Monte-Carlo reference (organizers' bake) unless",
         "> stated; MSE = final-layer MSE against it; score = MSE × max(0.1, C/B) per MLP, averaged.", ""]
    env = out["env"] or {}
    md = (out["meta"] or {})
    # reference floor = mean per-neuron output variance / n_samples of the organizers' MC
    avg_var = []
    for p in sorted(glob.glob(os.path.join(D, "mlps", "mlp_*.npz")))[:200]:
        try:
            avg_var.append(float(np.load(p)["avg_variance"]))
        except Exception:
            pass
    n_ref = md.get("n_samples")
    floor_ref = (float(np.mean(avg_var)) / float(n_ref)) if (avg_var and n_ref) else None
    out["floor_ref"] = floor_ref
    L += ["## Environment", "",
          f"- GPU: {env.get('nvidia_smi') or 'none'}; {env.get('cuda_version') or ''}; device used: {env.get('device')}; "
          f"cupy: {(env.get('device_info') or {}).get('cupy')}",
          f"- CPU: {env.get('nproc')} cores, {fmt(env.get('mem_total_gb'), 1)} GB; python {env.get('python')}; profile `{env.get('profile')}`",
          f"- commit {((env.get('provenance') or {}).get('commit') or '')[:12]} (dirty={((env.get('provenance') or {}).get('dirty'))}); "
          f"versions {json.dumps((env.get('provenance') or {}).get('versions'))}",
          f"- dataset: {md.get('dataset')} split {md.get('split')} ({md.get('n_total')} MLPs; reference n_samples = {md.get('n_samples')}; "
          f"reference MC floor ≈ {fmt(floor_ref, 1)} per neuron — every MSE below includes it)", "",
          "| stage | state | seconds | note |", "|---|---|---|---|"]
    for k, v in out["status"].items():
        L.append(f"| {k} | {v['state']} | {v['seconds']:.0f} | {v.get('note', '')} |")
    L.append("")

    # ---------------------------------------------------------------- V29 baseline and sweep
    tags = {}
    for d in sorted(glob.glob(os.path.join(D, "v29", "*"))):
        rows = rows_of(os.path.join(d, "mlp_*.json"))
        if rows:
            tags[os.path.basename(d)] = {r["idx"]: r for r in rows}
    out["v29"] = {}
    base = tags.get("baseline", {})
    if base:
        ok = [r for r in base.values() if not r.get("error")]
        m = boot_mean([r["mse_final"] for r in ok]); sc = boot_mean([r["score"] for r in ok])
        out["v29"]["baseline"] = dict(n=len(ok), n_err=len(base) - len(ok), mse=m, score=sc,
                                      cb=float(np.mean([r["cb"] for r in ok])), residual_max=float(np.max([r["residual"] for r in ok])),
                                      wall_mean=float(np.mean([r["wall"] for r in ok])), idx=sorted(base))
        L += ["## V29 (+ leaf 8) baseline on the dataset reference", "",
              f"{len(ok)} MLPs (idx {min(base)}–{max(base)}; {len(base) - len(ok)} failed): raw MSE **{fmt(m[0])}** [{fmt(m[1], 2)}, {fmt(m[2], 2)}], "
              f"C/B {out['v29']['baseline']['cb']:.4f}, score {fmt(sc[0])} [{fmt(sc[1], 2)}, {fmt(sc[2], 2)}]; residual max "
              f"{out['v29']['baseline']['residual_max']:.3f} s (parallel CPU runs — not the grader's timing), wall {out['v29']['baseline']['wall_mean']:.0f} s/MLP.",
              "Grader record for comparison: AIcrowd 336554 raw 2.13e-8 at C/B 0.245 → 5.23e-9.", ""]
        if len(ok) >= 2:
            per = np.array([r["mse_per_layer"] for r in ok]).mean(0)
            L += ["Mean all-layer MSE per layer (V29 vs reference): " + ", ".join(f"{x:.1e}" for x in per), ""]
    # sweep
    sweep = {t: v for t, v in tags.items() if t not in ("baseline",)}
    if sweep and base:
        L += ["## Knob sweep (Algorithm D): V29 variants on the same MLPs, paired against the baseline", "",
              "| variant | env | n | raw MSE | C/B | score | MSE ratio vs base [95% CI] | score ratio vs base [95% CI] |", "|---|---|---|---|---|---|---|---|"]
        tab = []
        for t, rows in sorted(sweep.items()):
            common = [i for i in rows if i in base and not rows[i].get("error") and not base[i].get("error")]
            if not common:
                continue
            a_m = [rows[i]["mse_final"] for i in common]; b_m = [base[i]["mse_final"] for i in common]
            a_s = [rows[i]["score"] for i in common]; b_s = [base[i]["score"] for i in common]
            rm, rs = paired_log_ratio(a_m, b_m), paired_log_ratio(a_s, b_s)
            cb = float(np.mean([rows[i]["cb"] for i in common]))
            rec = dict(tag=t, env=rows[common[0]].get("env"), n=len(common), mse=float(np.mean(a_m)), cb=cb, score=float(np.mean(a_s)),
                       mse_ratio=rm[:3], score_ratio=rs[:3], n_err=sum(1 for r in rows.values() if r.get("error")))
            tab.append(rec)
            L.append(f"| {t} | `{' '.join(rec['env'] or [])}` | {rec['n']} | {fmt(rec['mse'])} | {cb:.4f} | {fmt(rec['score'])} | "
                     f"{rm[0]:.3f} [{rm[1]:.3f}, {rm[2]:.3f}] | **{rs[0]:.3f}** [{rs[1]:.3f}, {rs[2]:.3f}] |")
        out["v29"]["sweep"] = tab
        if tab:
            best = min(tab, key=lambda r: r["score_ratio"][0])
            L += ["", f"Best score ratio: **{best['tag']}** ({best['score_ratio'][0]:.3f} [{best['score_ratio'][1]:.3f}, {best['score_ratio'][2]:.3f}]); "
                  "a ratio whose CI excludes 1 is a real change on these networks; the chosen knob must be confirmed on held-out MLPs "
                  "(the `confirm_*` rows, if present, are that confirmation).", ""]

    # ---------------------------------------------------------------- dense references
    dense = defaultdict(dict)
    for r in rows_of(os.path.join(D, "dense", "*_mlp_*.json")):
        dense[r["variant"]][r["idx"]] = r
    out["dense"] = {}
    if dense:
        L += ["## Dense / light references at the official shape (GPU), against the dataset reference", "",
              "| variant | n MLPs | raw MSE [95% CI] | ratio to V29 [95% CI] (paired) | ratio to K3c2 (paired) | s/MLP | peak GB | errors |",
              "|---|---|---|---|---|---|---|---|"]
        k3 = dense.get("K3c2", {})
        for v, rows in dense.items():
            ok = {i: r for i, r in rows.items() if not r.get("error")}
            if not ok:
                L.append(f"| {v} | 0 | — | — | — | — | — | {len(rows)}: {list(rows.values())[0].get('error')} |")
                continue
            m = boot_mean([r["mse_final"] for r in ok.values()])
            cv = [i for i in ok if i in base and not base[i].get("error")]
            rv = paired_log_ratio([ok[i]["mse_final"] for i in cv], [base[i]["mse_final"] for i in cv]) if cv else (np.nan,) * 4
            ck = [i for i in ok if i in k3 and not k3[i].get("error")]
            rk = paired_log_ratio([ok[i]["mse_final"] for i in ck], [k3[i]["mse_final"] for i in ck]) if ck and v != "K3c2" else (np.nan,) * 4
            rec = dict(n=len(ok), mse=m, ratio_v29=rv[:3], n_v29=rv[3], ratio_k3=rk[:3], seconds=float(np.mean([r["seconds"] for r in ok.values()])),
                       peak_gb=max([r.get("peak_mem_gb") or 0 for r in ok.values()]), n_err=len(rows) - len(ok), idx=sorted(ok))
            if v == "K2":
                rec["k2_vs_kit_maxabs"] = max(r.get("k2_vs_kit_maxabs", 0) for r in ok.values())
                rec["kit_mse_final"] = float(np.mean([r.get("kit_mse_final", np.nan) for r in ok.values()]))
            out["dense"][v] = rec
            L.append(f"| {v} | {len(ok)} | {fmt(m[0])} [{fmt(m[1], 2)}, {fmt(m[2], 2)}] | {rv[0]:.3f} [{rv[1]:.3f}, {rv[2]:.3f}] (n={rv[3]}) | "
                     f"{'—' if v == 'K3c2' else f'{rk[0]:.3f} [{rk[1]:.3f}, {rk[2]:.3f}]'} | {rec['seconds']:.0f} | {rec['peak_gb']:.1f} | {rec['n_err']} |")
        L.append("")
        if "K2" in out["dense"]:
            L.append(f"Orientation check: DenseProp(K=2) vs the starter-kit covariance closure on the dataset's weights, max |Δ| = "
                     f"{out['dense']['K2'].get('k2_vs_kit_maxabs'):.1e} (must be ~1e-6 or less in float32).")
        L += ["", "**Pre-registered predictions (plan of 2026-10-17, from the n ≤ 256 scaling fits):**", ""]
        for v, pr in PRED.items():
            if v in out["dense"]:
                m = out["dense"][v]["mse"]
                inside = pr["lo"] <= m[0] <= pr["hi"]
                L.append(f"- {pr['text']}: measured **{fmt(m[0])}** [{fmt(m[1], 2)}, {fmt(m[2], 2)}] → **{'confirmed' if inside else 'falsified'}**"
                         + ("" if inside else " (outside the pre-registered interval)"))
                out["dense"][v]["prediction"] = dict(pr, measured=m[0], confirmed=bool(inside))
        # error structure: is the dense reference's error correlated with V29's? does a 50/50 blend help?
        if base:
            blend = []
            for v, rows in dense.items():
                cs, rr, bl = [], [], []
                for i, r in rows.items():
                    if r.get("error") or i not in base or base[i].get("error"):
                        continue
                    try:
                        pv = np.load(os.path.join(D, "dense", f"{v}_mlp_{i:04d}.npz"))["preds"][-1]
                        pb = np.load(os.path.join(D, "v29", "baseline", f"mlp_{i:04d}.npz"))["preds"][-1].astype(np.float64)
                        tr = np.load(os.path.join(D, "mlps", f"mlp_{i:04d}.npz"))["final_means"]
                    except Exception:
                        continue
                    ev, eb = pv - tr, pb - tr
                    cs.append(float(np.corrcoef(ev, eb)[0, 1])); bl.append(float(np.mean(((pv + pb) / 2 - tr) ** 2))); rr.append(float(np.mean(eb ** 2)))
                if cs:
                    rb = paired_log_ratio(bl, rr)
                    out["dense"][v]["err_corr_with_v29"] = float(np.mean(cs)); out["dense"][v]["blend_ratio_vs_v29"] = rb[:3]
                    blend.append(f"{v}: corr(e_{v}, e_V29) = {np.mean(cs):+.2f}, MSE of the 50/50 blend / V29 = {rb[0]:.3f} [{rb[1]:.3f}, {rb[2]:.3f}] (n={rb[3]})")
            if blend:
                L += ["Error structure vs V29 (same final-layer neurons): " + "; ".join(blend) + ".", ""]
        if "coh2" in out["dense"] and out["dense"]["coh2"]["n_v29"]:
            r = out["dense"]["coh2"]["ratio_v29"]
            L.append(f"- coh2 / V29 paired ratio = {r[0]:.3f} [{r[1]:.3f}, {r[2]:.3f}]: a clean K=3 + coherent κ4 "
                     f"{'beats' if r[2] < 1 else ('does not beat' if r[1] > 1 else 'is not distinguishable from')} V29 on these networks.")
        L.append("")

    # ---------------------------------------------------------------- MC check
    mc = rows_of(os.path.join(D, "mc_check", "mlp_*.json"))
    if mc:
        out["mc_check"] = mc
        L += ["## Monte-Carlo cross-check of the dataset reference (our device MC vs the organizers' means)", "",
              "| idx | N ours | MSE(ours, ref) | floor ours | floor ref (n_samples) | expected = sum | ratio | var/neuron | s |", "|---|---|---|---|---|---|---|---|---|"]
        for r in mc:
            L.append(f"| {r['idx']} | {r['N']:.0e} | {fmt(r['mse_ours_vs_ref'])} | {fmt(r['floor_ours'])} | {fmt(r['floor_ref'])} | {fmt(r['expected'])} | "
                     f"{'—' if r['ratio'] is None else f'{r['ratio']:.2f}'} | {r['var_per_neuron_mean']:.3f} | {r['seconds']:.0f} |")
        L += ["", "A ratio near 1 means both references are unbiased estimates of the same means at their stated sample counts; "
              "a ratio ≫ 1 would mean the dataset reference (or our MC) is biased or its n_samples is not what the metadata says.", ""]

    # ---------------------------------------------------------------- K4 go/no-go
    k4 = rows_of(os.path.join(D, "k4", "n*_L*_s*.json"))
    if k4:
        byn = defaultdict(list)
        for r in k4:
            byn[(r["n"], r["L"])].append(r)
        out["k4"] = {}
        L += ["## Factored-κ4 go/no-go (Algorithm A): dense K=4 cap 3 at depth 16 vs device MC truth", "",
              f"Kill rule (pre-registered): K4c3 / K3c2 > {K4_KILL} at the largest width run → the program is dead. "
              "Ratios are geometric means over seeds with bootstrap CIs; per-slice rows are ratios to K4c3 (> 1 = removing the slice hurts).", "",
              "| n | L | seeds | floor | K3c2 | K4c3 | K4c3/K3c2 [95% CI] | K4c2/K3c2 | coh2/K3c2 | no(3,1)/K4c3 | no(2,1,1)/K4c3 | no(1,1,1,1)/K4c3 | paired/K4c3 | +(3,1)/K4c3 | +(2,1,1)/K4c3 | s/run (K4c3) | errors |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for (n, Lr), rows in sorted(byn.items()):
            def col(cfg):
                return [r["v"].get(cfg, {}).get("mse") for r in rows]
            def ratio(a, b):
                A, Bv = col(a), col(b)
                ok = [(x, y) for x, y in zip(A, Bv) if x is not None and y is not None and x > 0 and y > 0]
                if not ok:
                    return (np.nan, np.nan, np.nan, 0)
                return paired_log_ratio([x for x, _ in ok], [y for _, y in ok])
            def mean(cfg):
                v = [x for x in col(cfg) if x is not None]
                return float(np.mean(v)) if v else None
            rec = dict(n=n, L=Lr, seeds=sorted(r["seed"] for r in rows), floor=float(np.mean([r["floor"] for r in rows])),
                       K3c2=mean("K3c2"), K4c3=mean("K4c3"), ratio_K4c3_K3c2=ratio("K4c3", "K3c2")[:3],
                       ratio_K4c2_K3c2=ratio("K4c2", "K3c2")[:3], ratio_coh2_K3c2=ratio("coh2", "K3c2")[:3],
                       no31=ratio("K4c3-no31", "K4c3")[:3], no211=ratio("K4c3-no211", "K4c3")[:3], no1111=ratio("K4c3-no1111", "K4c3")[:3],
                       paired=ratio("paired_c3", "K4c3")[:3], plus31=ratio("paired_c3+31", "K4c3")[:3], plus211=ratio("paired_c3+211", "K4c3")[:3],
                       sec=float(np.mean([r["v"].get("K4c3", {}).get("seconds") or 0 for r in rows])),
                       errors=sum(1 for r in rows for v in r["v"].values() if v.get("error")))
            out["k4"][f"n{n}_L{Lr}"] = rec
            f3 = lambda t: f"{t[0]:.2f}" if np.isfinite(t[0]) else "—"
            r0 = rec["ratio_K4c3_K3c2"]
            L.append(f"| {n} | {Lr} | {len(rows)} | {fmt(rec['floor'], 1)} | {fmt(rec['K3c2'])} | {fmt(rec['K4c3'])} | "
                     f"**{f3(r0)}** [{r0[1]:.2f}, {r0[2]:.2f}] | {f3(rec['ratio_K4c2_K3c2'])} | {f3(rec['ratio_coh2_K3c2'])} | {f3(rec['no31'])} | {f3(rec['no211'])} | "
                     f"{f3(rec['no1111'])} | {f3(rec['paired'])} | {f3(rec['plus31'])} | {f3(rec['plus211'])} | {rec['sec']:.0f} | {rec['errors']} |")
        big = max(out["k4"].values(), key=lambda r: r["n"])
        r0 = big["ratio_K4c3_K3c2"]
        verdict = "no data" if not np.isfinite(r0[0]) else ("**GO** (ratio below the kill threshold)" if r0[0] <= K4_KILL else "**NO-GO** (ratio above the kill threshold)")
        L += ["", f"Verdict at the largest width run (n = {big['n']}): K4c3/K3c2 = {f3(r0)} vs kill threshold {K4_KILL} → {verdict}. "
              "The width trend of the ratio (≈ 4× per doubling at L = 8 in results/2026-10-17_w7_coherent_even.md) is what extrapolates to n = 1024; "
              "a ratio that does not fall with n at depth 16 falsifies the extrapolation.", ""]
        out["k4"]["verdict"] = verdict

    # ---------------------------------------------------------------- corrector
    cor = read_json(os.path.join(D, "corrector", "summary.json"))
    if cor:
        out["corrector"] = {k: v for k, v in cor.items() if k not in ("ridge_coef_std",)}
        L += ["## Residual corrector (Algorithm C) on V29, fitted on networks disjoint from the test networks", ""]
        if cor.get("error"):
            L.append(f"Not run: {cor['error']}")
        else:
            L += [f"Train: {len(cor['train_idx'])} networks ({cor['n_train_neurons']} neurons); test: {len(cor['test_idx'])} networks; "
                  f"features: {len(cor['features'])}; ridge λ = {cor.get('ridge_lambda')}.", "",
                  "| model | split | n | MSE before | MSE after | after/before [95% CI] | wins |", "|---|---|---|---|---|---|---|"]
            for split in ("cv", "test"):
                for mdl in ("ridge", "mlp"):
                    s = (cor.get(split) or {}).get(mdl)
                    if not s:
                        continue
                    r = s["ratio_after_over_before"]
                    L.append(f"| {mdl} | {split} | {s['n']} | {fmt(s['mse_before'][0])} | {fmt(s['mse_after'][0])} | **{r[0]:.3f}** [{r[1]:.3f}, {r[2]:.3f}] | {s['wins']}/{s['n']} |")
            co = cor.get("ridge_coef_std", {})
            top = sorted(co.items(), key=lambda kv: -abs(kv[1]))[:8]
            L += ["", "Largest standardized ridge coefficients: " + ", ".join(f"{k} {v:+.2e}" for k, v in top), "",
                  "Kill rule: < 10% MSE reduction on held-out networks → drop the corrector. A ratio of 0.73 would match Hisernberg's public GRU corrector.", ""]

    # ---------------------------------------------------------------- write
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    jpath = os.path.join(ROOT, "results", f"{day}_gpu_run.json")
    mpath = os.path.join(ROOT, "results", f"{day}_gpu_run.md")
    json.dump(out, open(jpath, "w"), indent=1, default=float)
    open(mpath, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {mpath} and {jpath}")


if __name__ == "__main__":
    main()
