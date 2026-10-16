# Stage 0 — V29 Strassen leaf size (STRASSEN_MIN 32 / 16 / 8): cost, memory, timing, predictions

> LLM-generated record (Claude Code), numbers copied from `2026-10-16_stage0_leaf_probe.json`
> (raw logs in git-ignored `data/runs/`). Status: **measured** unless marked otherwise.

**Question.** Is `STRASSEN_MIN = 8` a pure cost change for V29, and is it safe under the grader caps?
**Falsifier (stated before the run).** Not a pure cost change if final-layer predictions move by
more than ~1% of the raw MSE (≈ 2e-10 MSE-equivalent); unsafe if peak RSS / residual / wall grow
enough to breach the caps after scaling by the measured local/grader ratio.

**Setup.** 4 self-baked MLPs at 1024×16 (`whest dataset bake`, MLP seeds 0–3, 2e4-sample truth —
too noisy for MSE, used for cost/timing only). Estimator `third_party/v29/estimator_v29.py`
(byte-identical to upstream), leaf set via env `V26_STRASSEN_MIN`. Harness `arcmlp/eval/probe.py`:
one process, `threadpool_limits(1)`, no address-space limit; MLP idx 0 = first `predict()` of the
process (V29 caps Strassen at level 4 there), idx 1–2 steady state. Host: 4-core Xeon @ 2.1 GHz,
15 GB, Python 3.13.16, numpy 2.4.6, flopscope 0.12.3, whestbench 0.16.1.

## Results

| leaf | idx | C/B | wall s | residual s | peak RSS GB | peak VM GB |
|---|---|---|---|---|---|---|
| 32 (V29) | 0 (first) | 0.2671 | 71.2 | 0.474 | 4.20 | 6.21 |
| 32 | 1 | 0.2540 | 84.6 | 0.480 | 6.00 | 7.57 |
| 32 | 2 | 0.2526 | 79.0 | 0.435 | 6.00 | 7.57 |
| 16 | 0 (first) | 0.2624 | 78.8 | 0.525 | 4.49 | 6.51 |
| 16 | 1 | 0.2483 | 92.1 | 0.512 | 6.34 | 7.91 |
| 16 | 2 | 0.2470 | 84.1 | 0.482 | 6.34 | 7.91 |
| 8 | 0 (first) | 0.2592 | 87.9 | 0.529 | 5.25 | 7.26 |
| 8 | 1 | 0.2442 | 100.6 | 0.556 | 6.48 | 8.06 |
| 8 | 2 | 0.2432 | 93.1 | 0.505 | 6.48 | 8.06 |

Final-layer prediction change vs leaf 32 (MSE between the two prediction vectors), MLPs 0/1/2:
leaf 16: 4.1e-13 / 5.5e-13 / 9.1e-12; leaf 8: 3.8e-13 / 5.1e-13 / 1.1e-11 (max |Δ| 1.3e-5).
Relative to V29's raw MSE 2.13e-8 this is ≤ 0.05%; by Cauchy–Schwarz the raw MSE can move by at
most 2·√(2.13e-8 · 1.1e-11) ≈ 1e-9 (4.5%) in the worst (perfectly aligned) case [proven bound];
the expected change is ≈ 0 because the float32 rounding differences are not aligned with the error
[hypothesis, consistent with 504aldo's "MSE unchanged to 4 digits"].

**FLOP accounting reproduces the grader** [measured vs external]: leaf-32 first-call 0.2671 and
steady 0.2526 equal the grader's max/min multipliers for V29 verbatim (#333467: min 0.2526,
max 0.2671).

## Local `whest run --runner subprocess` is stricter than the grader (negative result)

Under `whest run --runner subprocess --max-threads 1` (worker `RLIMIT_AS` = 8 GiB, arrays inside
the worker) **both** leaf 8 and the unmodified V29 fail locally:

- leaf 8: `_ArrayMemoryError` allocating a 300 MiB Strassen pool on MLP 1, then the worker died
  on MLP 2; residual 0.55 s on the MLPs that ran (cap 0.4).
- leaf 32 (graded 100/100 on the grader): worker died on MLP 2 (unpinned), `_ArrayMemoryError`
  375 MiB on MLP 1 when pinned to one core (peak VM 8.28 GB); residual 0.45–0.49 s.

The grader runs a flopscope *client* in the participant process with arrays and kernels in a
flopscope server (report fields `flopscope_server_kernel_wall_s`, `participant_shim_version`), and
documents the limit as "8 GB of RAM" for the solution process. So the local RLIMIT_AS test does not
predict grader memory failures, and local residual is ≈ 2.4–2.6× the grader's (V29: 0.435–0.48 s
here vs 0.183 s mean / 0.195 s max on the grader). Only **ratios** between variants measured on
the same host are used below.

## Python 3.10 / numpy 2.2.6 check (the grader's interpreter and numpy)

| bundle | idx | C/B | wall s | residual s | peak RSS GB |
|---|---|---|---|---|---|
| `submission_fallback_v29` (leaf 32) | 0 / 1 | 0.2671 / 0.2540 | 73.7 / 91.1 | 0.585 / 0.602 | 4.04 / 5.89 |
| `submission` (leaf 8) | 0 / 1 | 0.2592 / 0.2442 | 95.7 / 105.6 | 0.661 / 0.674 | 5.21 / 6.37 |

Same FLOPs as under Python 3.13; leaf-8 predictions within 7e-13 MSE of the 3.13 run; residual
ratio leaf 8 / leaf 32 = 1.12–1.13 (3.13: 1.12–1.16); peak-RSS delta +0.48 GB (3.13: +0.48 GB).

## Projection to the grader for leaf 8 [hypothesis — projection, not a measurement]

Scaling the grader's V29 record (#333467, external) by the local leaf-8/leaf-32 ratios:

| quantity | V29 on grader | ratio (local) | projected leaf 8 | cap |
|---|---|---|---|---|
| mean multiplier | 0.2542 | 0.963 (steady), 0.970 (first) | ≈ 0.245 | — |
| adjusted score | 5.418e-9 | same | ≈ 5.22e-9 (raw unchanged) | — |
| residual s (mean / max) | 0.183 / 0.195 | 1.12–1.16 | ≈ 0.21 / 0.23 | 0.4 |
| predict wall s (mean / max) | 68.9 / 75.4 | 1.18 | ≈ 81 / 89 | 120 |
| peak RSS | not reported | +0.48 GB (6.00 → 6.48 GB local) | +0.5 GB | 8 GB |
| whole-submission time (5 workers) | 24.4 min | 1.18 | ≈ 29 min | 40 min |

Conclusion: leaf 8 is a cost-only change worth ≈ −3.7% on the adjusted score, with all projected
quantities inside the caps but with less margin (memory +0.5 GB, wall +18%). Leaf 16 gives
≈ −2.2% with smaller increments. The brief reports leaf 8 as grader-accepted for another
participant's build (not verified by us). Decision for Stage 0: submit leaf 8 as primary and V29
verbatim as fallback (`results/stage0_submission.md`).
