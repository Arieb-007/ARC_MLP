# STATUS

> Maintained by the LLM agent (Claude Code) at the end of every session. Labels: proven / measured /
> hypothesis / external.

## Session 1 — 2026-10-16 (started 12:36 UTC)

### Environment facts (measured)
- Cloud sandbox: 4-core Xeon @ 2.1 GHz, 15 GB RAM, no swap. Network policy blocks
  huggingface.co, aicrowd.com (incl. discourse and gitlab), arxiv.org; public GitHub is readable
  (git), pushing to Arieb-007/ARC_MLP works. No AIcrowd API key in the environment.
- Under the brief's rule (§6.4: < 16 GB) official-shape *evaluations* are notebook work; here we
  only ran cost/timing probes on 4 self-baked 1024×16 MLPs.
- Python 3.13 venv (`/home/user/venv`) and a Python 3.10 venv matching the grader
  (`/home/user/venv310`: numpy 2.2.6, flopscope 0.12.3).

### Stage 0 — ready, awaiting Areeb's submit
- `submission/estimator.py` = V29 + STRASSEN_MIN=8 (primary), `submission_fallback_v29/` = V29
  verbatim. Both validated and packaged under Python 3.10. Details + commands:
  `results/stage0_submission.md`.
- Measured (`results/2026-10-16_stage0_leaf_probe.*`): leaf 8 is a cost-only change (prediction
  delta ≤ 1.1e-11 MSE-equivalent), steady C/B 0.2526 → 0.2432 (−3.7%), +0.48 GB peak RSS, +16%
  residual, +18% wall (local ratios). Our FLOP counts reproduce the grader's V29 min/max exactly.
- Negative result: local `whest run --runner subprocess` (RLIMIT_AS 8 GiB) fails even the
  grader-accepted V29 — not predictive of the grader (client/server split there).

### Proven
- F1 closed forms incl. κ4(ReLU(sZ), t=0) = 0.279708 s⁴ (numerically checked).
- F3 order table (moments of W); the coherent part of the κ4 *partial trace*
  Σ_k κ4(z)_{ijkk} = Σ_m W_im W_jm b_m Σ_k W_km² with Σ_k W_km² = 2 + O(n^{-1/2}).
- Correction to the brief: V29's λ_l multiplies the harmonic-projection (partial-trace) core, not
  the (2,2) slice (504aldo F105; external) — see FINDINGS F3.2.

### Measured (this session)
- Stage-0 leaf probe (above).
- E1–E3 re-runs reproduce the brief's F4–F6 numbers exactly (`results/2026-10-16_e{1,2,3}_*.md`).
- Correction: E1's own "Gaussian-product heuristic" n(1−ε^{2/age}) overpredicts the measured ranks
  by 1.6–3.3× → falsified as a quantitative law (FINDINGS F4); the measured table stands.

### Open / next
1. Stage 0 DONE: 4 graded submissions. A (leaf 8): [AIcrowd 336554](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/336554), [AIcrowd 336558](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/336558) at 5.23e-9; B (V29 verbatim):
   [AIcrowd 336561](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/336561), [AIcrowd 336562](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/336562) at 5.42e-9; raw 2.13e-8 for all (bundle labels inferred from scores).
   Leaf-8 projection (≈ 5.22e-9) confirmed to 3 digits.
2. Stage 1 (2026-10-17): analytic λ attempted, stop rule triggered — no submission
   (`results/2026-10-17_stage1_lambda.md`, FINDINGS F8). Kept for the write-up: true-κ4 one-scalar
   law (R² 0.67–0.95), exact layer-0 λ (1%), memory-dominated depth (transport law ~68%), V29's λ
   is effective (3× below the physical value at layer 0). Open: the missing ~30% source term.
3. Stage 2.1: dense K≤4 reference at n=64/128 + MC verification.

Next acceptance criterion: Stage-1 write-up draft (docs/writeup/writeup.md) with every table backed by results/,
nominating [AIcrowd 336554](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/336554) (pending confirmation of its multiplier on the View page).

## 2026-10-17 (late): GPU run at the official shape — see results/2026-10-17_gpu_run_analysis.md

- [measured, A40, public mini split, N = 1e9 reference] V29+leaf 8 raw 2.26e-8 on 40 MLPs; dense K=3
  4.29e-7; K=3 + coherent κ4 3.93e-8 (1.7× worse than V29, errors correlated 0.73). Width exponents
  256→1024: K3 n^-1.99, coherent κ4 n^-3.01 (the integers of the order counting; the n ≤ 256 fits
  over-predicted — both pre-registered point predictions falsified).
- Knob sweep: no V29 knob improves the score (all CIs above 1). Corrector: 1.4 % on held-out MLPs → dropped.
- κ4 go/no-go at depth 16: K4c3/K3c2 = 0.46/0.16/0.10 at n = 128/160/192 → GO; the gain is in the
  random-sign (2,1,1) slice (removing it: 12×), not (3,1) (adding it alone recovers nothing).
- Decision: nominate AIcrowd 336554 (5.23e-9); write-up (Oct 28) uses the 1024 measurements; the
  factored (2,1,1) κ4 chain is the post-deadline program. Next acceptance criterion: write-up draft
  in docs/writeup/ with the n = 1024 tables.

## 2026-10-18: GPU round 2 — see results/2026-10-18_gpu_round2_analysis.md

- [measured] z-side conditional-coherent proxies: (3,1) ≈ complete (oracle 88–90 %), (2,1,1) partial and
  decreasing with width (74 → 55 % at n = 128 → 192) inside the dense chain; nothing in the light chain at
  1024 (coh2_pboth 1.85× V29). The effect lives in the transported random-sign κ4 memory.
- [measured] Covariance at 1024×16 = rank-16 spike (85 % of off-diagonal energy at layer 16) + bulk; the
  spiked closure is 670–6000× worse: the bulk carries the mean. Spectral energy is the wrong metric.
- K4 ladder (5 seeds, L = 16): 0.34/0.22/0.13 at n = 128/160/192 → GO stands; depth 8: 0.06/0.04/0.03.
- Next: round 3 = structured κ4 memory oracle (κ4(h) as b, B, λ⊗C forms) at n ≤ 192; write-up.

## 2026-10-18/19: GPU round 3 — see results/2026-10-18_gpu_round3_analysis.md

- [measured, n = 192, L = 16, 3 seeds] Cost-3 families by output order (ratio to K4c3 when removed): the
  random-sign read-outs rs_3 4728× (unstable without), k3e_3 65×, the κ3-block + edge κ4 births k3e_4 26×,
  the random-sign transmissions rs_4 13×; c3_3 1.74, p4e_3 1.50; the κ4 births from three edges (c3_4,
  1.03) and from paired κ4 + edge (p4e_4, 0.96) are droppable. Constructive: cheap + rs + k3e + p4e_3 =
  2.11× K4c3 (64 % of the gap); without the k3e_4 births 42×.
- [measured] κ4 generations cannot be ablated by term removal (transmitted κ4 without births diverges
  at every width); question left open.
- n = 256 K4 rows failed twice with a 16 GiB OOM: root cause found on the CPU (the last order-4 term's
  operand list kept the n⁴ pre-activation alive → 3 n⁴ peak), fixed 2026-10-19; rerun pending.
- Factored-κ4 design fixed by the family data: carry rs (transmission + read-out), k3e (births + read-out),
  p4e_3, c3_3; drop c3_4, p4e_4. Width scaling of each family: unmeasured [open].
- Next acceptance criterion: K4c3/K3c2 at n = 256 (kill > 0.09) from the rerun; then the write-up draft.
