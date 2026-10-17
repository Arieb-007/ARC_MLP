# GPU run of 2026-10-17 — reading of the results (A40, public mini split, organizers' N = 1e9 reference)

> LLM-generated record (Claude Code) for team Arieb-007. The raw report produced on the GPU box by
> `scripts/gpu_run.sh --profile fast` (commit 1be3bb8) was pasted back by Areeb and is stored verbatim
> as `results/2026-10-17_gpu_run_as_received.md`. Every number below is from that report [measured]
> unless labelled otherwise.

## Verdicts against the pre-registered rules

| test | rule (written before the run) | result | verdict |
|---|---|---|---|
| P1: dense K=3 cap 2 at 1024×16 | 2.8e-7 [1.9e-7, 4.2e-7] | 4.29e-7 [3.99e-7, 4.67e-7] | **falsified** (just above the interval) |
| P2: K=3 + coherent κ4 at 1024×16 beats V29 | 1.5e-8 [0.9e-8, 2.6e-8] | 3.93e-8; V29 = 2.26e-8 on the same MLPs (ratio 1.70 [1.61, 1.82]) | **falsified** |
| A go/no-go: K4c3/K3c2 at n = 192, L = 16 | kill if > 0.25 | 0.10 [0.09, 0.12]; 0.46 → 0.16 → 0.10 at n = 128/160/192 | **GO** |
| C corrector (ridge, held-out 20 MLPs) | drop if < 10 % MSE reduction | 0.986 [0.973, 0.999] | **dropped** |
| D knob sweep (8 variants, 20 MLPs, paired) | adopt only if the score ratio's CI excludes 1 below | every variant ≥ 1.000; best is the baseline itself | **dropped**: V29's knobs are a local optimum |
| MC cross-check of the reference | ratio ≈ 1 | 1.12 (N = 1e7), 0.87 (N = 1e8) | reference consistent with N = 1e9 |

## What the numbers say

1. **The width exponents are the integers of the order counting, not the small-n fits.** Between
   n = 256 (results/2026-10-17_w7_coherent_even.md, L = 16: K3c2 6.78e-6, coh2 2.57e-6) and
   n = 1024: K3c2 falls by 15.8× = 4^1.99, coh2 by 65× = 4^3.01. The fitted α = 2.29 / 3.71 from
   n ≤ 256 over-predicted the gain; the asymptotic law is MSE ∝ n^-2 for K=3 and n^-3 for K=3 +
   coherent κ4 [measured at two widths; consistent with the counting rule, proven at the level of
   the diagram orders]. This is the result to put in the write-up, replacing the fitted exponents.
2. **V29 is 1.6–1.7× better than a clean K=3 + coherent κ4 at the official shape**, and the two
   share their error (corr 0.73; a 50/50 blend is 10–15 % worse than V29). V29's extra ingredients —
   the fitted λ·C_off (3,1) proxy (removing it costs 1.17×), the D21 feedback and the fitted
   constants — buy that factor; our coherent model has nothing V29 lacks.
3. **V29 cannot be improved by its own knobs** (every change hurts: ranks, ages, λ scale, the (3,1)
   proxy) **nor by a per-neuron corrector on cheap features** (1.4 % on held-out networks; the
   residual is not predictable from per-neuron scalars, as argued in the plan). Hisernberg's
   reported 27 % with a GRU is not reproduced by our features; left open.
4. **The κ4 gain at depth 16 is real and grows with width** (K4c3/K3c2 0.46 → 0.16 → 0.10 from
   n = 128 to 192, i.e. faster than n^-2), but it lives in the *random-sign* slices: removing all of
   them (paired only) loses 10×; removing (2,1,1) alone loses 12×, (3,1) alone 4.5×, (1,1,1,1)
   1.8×. Adding (3,1) alone to the paired set recovers almost nothing (9.7× vs 10.2×); adding
   (2,1,1) alone recovers most of it (2.8× vs 10.2×). So the cheap A1 variant ((3,1) family only)
   is dead and the program needs the expensive (2,1,1) → mixed-leg κ3 births (2–3 matmuls per
   source-layer in the cost model of docs/notes/w5_cost_model.md).
   Extrapolation [hypothesis]: K4c3 is 3.3× below coh2 at n = 192 with a steeper exponent (≈ 4 vs 3);
   at n = 1024 that is ~2e-9 raw if the trend holds, against V29's 2.26e-8 — a 10× accuracy gain for
   a 2–3× cost increase, i.e. a score of ~1e-9 vs the leader's 1.42e-9, but only after a factored
   (2,1,1) implementation that does not exist (1–2 weeks of work).

## Caveats

- The baseline's C/B here is 0.2592, not the grader's 0.245, because the run used one process per
  MLP and V29 caps Strassen at depth 4 on the first predict of a process (`V28_STRASSEN_FIRST`);
  the grader runs all MLPs in one process. MSE comparisons are unaffected; the sweep's cost column
  for the Strassen knobs is not informative in this run (fixed for future runs: run_v29.py now
  sets `V28_STRASSEN_FIRST` to the Strassen depth unless overridden).
- Residual times are from parallel CPU runs and say nothing about the grader's 0.4 s cap.
- 10 MLPs for the dense references, 20 for the sweep and the corrector test; all CIs are bootstrap
  over MLPs.

## Decision

No new submission comes out of this run: V29 + leaf 8 (AIcrowd 336554, 5.23e-9) stays our Phase-2
entry and should be nominated. The remaining Phase-2 days are better spent on the write-up
(due Oct 28) with the n = 1024 measurements above; the factored random-sign κ4 chain is the
post-deadline research program, now with a measured target (the (2,1,1) slice) and a measured
kill test passed.
