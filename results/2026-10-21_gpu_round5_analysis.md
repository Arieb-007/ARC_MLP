# GPU round 5 (2026-10-21) — rank of the (2,1,1) κ4 slice at n = 192: reading

> LLM-generated record (Claude Code) for team Arieb-007. Raw report: `results/2026-10-21_gpu_round5_as_received.md`
> (A40, `scripts/gpu_run.sh --profile round5`, commit 8bd61f2). n = 192, L = 16, seeds 0–2, device MC truth
> N = 1e7 (floor 1.1e-8, 0.5 % of K4c3). Rules and derivation: `docs/notes/r5_cp_memory_theorem.md`.

## n = 192 (3 seeds): the CP memory needs rank 16–64 of n

| configuration | / K4c3 [95 % CI] | / K4c3-no1111 |
|---|---|---|
| bulk dropped, A exact (K4c3-no1111) | 1.80 [1.52, 2.05] | 1.00 |
| A → rank 1 (the λ ⊗ N ceiling) | 2.74 [2.32, 3.71] | 1.52 |
| A → rank 4 | 2.40 [2.13, 2.95] | 1.34 |
| A → rank 16 | 2.15 [1.93, 2.51] | 1.20 |
| A → rank 64 | 1.92 [1.52, 2.26] | 1.07 |
| A → rank 256 (= exact) | 1.80 | 1.00 |

Rank for ≤ 1.3 × the bulk-dropped chain, r(1.3): between 4 and 16 at n = 192; between 4 and 16 at n = 96
(CPU, one seed: 1.44 / 1.07) and ≤ 4 at n = 64, L = 8 (1.12). [measured] r(1.3) is ≈ 8 at both n = 96 and 192 —
no proportional growth over a factor 2 in n (the kill rule's 192 → 256 comparison is still missing).
The quality at fixed rank does degrade with n (rank 16: 1.07 → 1.20; rank 64: 0.98 → 1.07), i.e. the energy
beyond the first few directions matters more at larger n. [hypothesis] r(1.3) ∝ √n or slower.

(j | kl) spectrum of A at n = 192: top direction 45 % (layer 2) → 81 % (layer 14) of the energy, all-ones
j-direction 44 % → 73 %, k90 97 → 7 — the slice becomes nearly rank-one with depth. Compared with n = 96
at the same layers (f_ones 0.43–0.69), the fractions are flat in n: **Proposition B's prediction 0.15–0.35 is
falsified at every width; the (2,1,1) channel is dominated by its partial-trace component** (f_ones ≈ 0.5–0.7),
and the √2-fluctuation bound applies only to the remaining 30–50 %. The partial trace alone (a211:1 is its
ceiling with a free matrix) costs 1.52×; V29's λ·C is a constrained version of it.

Cost of the full CP chain at this quality [hypothesis, Theorem A's accounting]: rank 64 ⇒ transport
3 × 64/n units per term per layer; four birth families ⇒ ≈ 1 unit/layer at n = 1024, ≈ +0.02 B on V29.
What it buys: at n = 192 the rank-64 chain is 1.92 × K4c3 = 0.25 × K3c2, versus V29-class physics (coh2 +
λ·C) at ≈ 0.4 × K3c2 (coh2/K3c2 = 0.41 at 192; V29 itself is not measurable at this width).

## n = 256: every row failed instantly (0–1 s) with OutOfMemoryError for a 16 GiB block

The same failure as round 4, now also for the plain `K4c3-no1111` (a DenseProp with fewer terms than the
`K4c3` that ran at n = 256 in round 3 pass 3, 95 s). The CPU peak of the current code is 2.5 n⁴ (regression
test `test_peak_memory_of_order4_chains`), so the instant failure of the *first* 16 GiB allocation is not the
chain's own footprint. Candidates [hypotheses]: device memory held by a previous configuration in the same
process (an oracle ↔ DenseProp reference cycle delayed release of the final state), pool fragmentation,
or another process on the GPU (`commit … dirty=True`). Mitigation (commit of 2026-10-22): the n = 256
rows run one configuration per process (`k4r5big` stage), each row records the free device memory before
it starts, an OutOfMemoryError is retried once after a full release, and the cycle is broken after each run.
The pass-3 `K4c3` at n = 256 (4.84e-7, 3 seeds) remains the reference for the ratio.

## Status

- Theorem A (CP structure, O(n³)): stands; the oracle rank curve is what it predicts.
- Proposition B (single-matrix ceiling): the bound stands, its energy prediction is falsified (favourably).
- Kill rule (r ∝ n): not yet testable; r(1.3) ≈ 8 at n = 96 and 192. Next: the n = 256 rows.
