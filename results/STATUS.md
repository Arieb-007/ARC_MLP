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
