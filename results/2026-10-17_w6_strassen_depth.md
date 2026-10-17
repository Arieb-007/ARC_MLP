# W6 — deeper Strassen–Winograd in V29 (2026-10-17)

> LLM-generated record (Claude Code) for team Arieb-007. Tool: `arcmlp/eval/probe.py` (flopscope meter,
> single process). Dataset: self-baked 1024×16 MLPs `data/bake_1024x16_s0-3_n2e4` (seed 0–3), network idx 1
> (idx 0 is the first predict() of a process, where V29 caps Strassen at depth 4: identical cost at any level).
> Raw: `data/runs/probe_L6.log`, `data/runs/probe_leaf8.log` (depth 5 = submitted bundle).

**Falsifier (written before running).** Deeper Strassen is not worth it if the metered C/B at depth 6 is not
below depth 5, if predictions move by more than 1e-10 in mean squared difference, or if residual > 0.4 s.

| Strassen depth | C/B (idx 1) | mean (Δpred)² vs depth 5 | residual (s, CPU contended) |
|---|---|---|---|
| 5 (submitted, AIcrowd 336554) | 0.24419 | 0 | 0.556 |
| 6 | 0.23825 (−2.4%) | 5.6e-13 | 0.581 |
| 7 | not measured (killed: out of memory while other jobs ran) | | |

Findings:
- [measured] Depth 6 lowers billed cost by 2.4% on this network with no accuracy change (5.6e-13 ≪ raw MSE 2e-8).
  Expected score effect if it holds on the grader: 5.23e-9 × 0.976 ≈ 5.10e-9 [extrapolation, hypothesis].
- [proven, accounting] W6's premise "(7/8)^10 ≈ 0.26" ignores additions, which the meter bills and whose volume
  grows as (7/4)^k; V29 already batches each level into one call per combination (the proposed einsum
  reformulation). Estimated product cost at n = 1024 (fraction of dense): depth 5 0.554, 6 0.523, 7 0.525,
  8 0.577 — the optimum is depth 6–7 and the realistic gain is a few percent, not 1.2–1.5×.
- [not established] Residual time: every local residual (including the submitted depth 5, which the grader
  accepted) exceeds 0.4 s on this contended VM, so the residual check is inconclusive here. V28's own ladder
  reports ~+0.02 s per level.

Bundle prepared, NOT submitted: `submission_L6/` (only change vs `submission/`: default Strassen depth 5 → 6).
