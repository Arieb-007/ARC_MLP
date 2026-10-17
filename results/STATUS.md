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
1. Stage 0 DONE: #336561 (A, leaf 8) and #336562 (B, V29 verbatim) both GRADED, 5.42e-9 adjusted,
   2.13e-8 raw. Open: why A shows no leaf-8 gain (need multiplier from the report).
2. Stage 1: derive λ_l from the coherent partial trace (tree-level joint cumulants of ReLU outputs +
   W^T Γ W ≈ 2γ̄ I); compare with V29's LAM/REF_R tables; first numerical check at small n.
3. Stage 2.1: dense K≤4 reference at n=64/128 + MC verification.

Next acceptance criterion: Stage-1 write-up draft (docs/writeup/writeup.md) with every table backed by results/.
