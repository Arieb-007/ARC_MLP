"""Aggregate data/v1/*.json into results/<date>_dense_reference_validation.{md,json}."""
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arcmlp.eval.local_eval import provenance  # noqa: E402

day = time.strftime("%Y-%m-%d", time.gmtime())
out = {"A": {}, "B": None, "C": None, "provenance": provenance(".")}
lines = [f"# Dense cumulant reference — validation ({day})", "",
         "> LLM-generated record (Claude Code). Code: `arcmlp/reference/dense_cumulants.py` (transcribed from ARC's",
         "> reference implementation, MIT, commit 93d091a; the paper PDF was unreachable). Scripts:",
         "> `experiments/v1_dense_validation.py`, `experiments/v1_report.py`; unit tests `tests/test_dense.py`,",
         "> `tests/test_relu_moments.py`. Raw outputs: `data/v1/` (git-ignored) mirrored in the .json.", "",
         "**Falsifier (written before running).** The reference is wrong if (A) at n = 16 or 32 the Kt=4 entrywise",
         "deviation of an order exceeds the Kt=3 deviation, or the Kt=4 deviation does not shrink from n = 16 to",
         "n = 32; or (B) Kt=4 has a larger final-layer MSE than Kt=3 on average at n = 64, L = 8.", "",
         "**Independent oracle.** Running ARC's own code needs torch, whose CPU index is blocked here; the",
         "transcription is validated by exact identities instead: Kt=2 equals the starter-kit closure (1e-15);",
         "covariance-only diagrams converge to the exact bivariate ReLU covariance (2-D quadrature) with error",
         "1.9e-4 -> 3.5e-18 as the cap grows 1 -> 9 at c = 0.05; single-neuron kappa3/kappa4 coefficients give",
         "the predicted c^3 error law on z = a + b g + c (g^2 - 1) (slope 3.11 with cap 4, 1.99 without kappa3^2).", ""]
for path in sorted(glob.glob("data/v1/A_n*.json")):
    if "quick" in path:
        continue
    rows = json.load(open(path))
    n = rows[0]["n"]
    out["A"][n] = rows
    L = max(r["layer"] for r in rows) + 1
    seeds = sorted({r["seed"] for r in rows})
    lines += [f"## A. Entrywise vs MC full cumulant tensors, n = {n}, L = {L}, seeds {seeds}, N = {rows[0]['N']:.0e} "
              f"({rows[0]['reps']} independent MC runs → per-entry SE)", "",
              "Relative RMS deviation ‖K_prop − K_MC‖/‖K_MC‖ (mean over seeds); MC SE/RMS in the last column.", "",
              "| layer | order | Kt=2 | Kt=3 | Kt=4 | MC SE/RMS |", "|---|---|---|---|---|---|"]
    for l in range(L):
        rl = [r for r in rows if r["layer"] == l]
        for k in (2, 3, 4):
            f = lambda key: np.mean([r[key] for r in rl])
            lines.append(f"| {l} | κ{k} | {f(f'K2_rel_rms_k{k}'):.3f} | {f(f'K3_rel_rms_k{k}'):.3f} | {f(f'K4_rel_rms_k{k}'):.3f} | {f(f'se_over_rms_k{k}'):.4f} |")
    rl = [r for r in rows if r["layer"] == L - 1]
    lines += ["", f"Per-slice Kt=4 deviation at the last layer (layer {L-1}), relative to that slice's RMS:", "",
              "| slice | rel. RMS dev (Kt=3) | rel. RMS dev (Kt=4) | MC SE/RMS |", "|---|---|---|---|"]
    for k, pats in ((2, ["(2,)", "(1, 1)"]), (3, ["(3,)", "(2, 1)", "(1, 1, 1)"]),
                    (4, ["(4,)", "(3, 1)", "(2, 2)", "(2, 1, 1)", "(1, 1, 1, 1)"])):
        for pat in pats:
            f = lambda key: np.mean([r[key] for r in rl])
            lines.append(f"| κ{k} {pat} | {f(f'K3_rel_rms_k{k}{pat}'):.3f} | {f(f'K4_rel_rms_k{k}{pat}'):.3f} | {f(f'se_over_rms_k{k}{pat}'):.4f} |")
    lines.append("")
if os.path.exists("data/v1/B_n64.json"):
    rows = json.load(open("data/v1/B_n64.json"))
    out["B"] = rows
    lines += ["## B. Final-layer MSE vs MC, n = 64, L = 8, N = 2e7 (debiased; ± MC SE)", "",
              "| seed | MC floor | Kt=2 | Kt=3 | Kt=4 | Kt=3/Kt=4 |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['seed']} | {r['floor']:.1e} | {r['K2']:.3e} ± {r['K2_se']:.0e} | {r['K3']:.3e} ± {r['K3_se']:.0e} | "
                     f"{r['K4']:.3e} ± {r['K4_se']:.0e} | {r['K3']/r['K4']:.2f} |")
    g = lambda k: float(np.exp(np.mean(np.log([r[k] for r in rows]))))
    lines += [f"| geo-mean | | {g('K2'):.3e} | {g('K3'):.3e} | {g('K4'):.3e} | {g('K3')/g('K4'):.2f} |", ""]
if os.path.exists("data/v1/C.json"):
    c = json.load(open("data/v1/C.json"))
    out["C"] = c
    lines += ["## C. Cross-check against experiments/conditional.py (seed 3, n = 256, L = 8, same MC protocol)", "",
              f"Kt=2 final-layer MSE = {c['mse_K2']:.4e} (conditional.py: 3.563e-05).", ""]
json.dump(out, open(f"results/{day}_dense_reference_validation.json", "w"), indent=1, default=str)
open(f"results/{day}_dense_reference_validation.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
