# Round 5 CPU preliminary — (j | kl) rank of the (2,1,1) κ4 slice at small widths (2026-10-20)

> LLM-generated record (Claude Code) for team Arieb-007. Tool: `experiments/gpu/k4_gonogo.py --device cpu`
> (commit f26ffad; numpy 2.5.4, scipy 1.18.1; float64 at n = 64, float32 at n = 96). W ~ N(0, 2/n) from
> `default_rng(seed)`; truth = our MC, N = 1e7 (floor ≈ 1e-8, ≤ 0.1 % of every MSE). Rules and derivation:
> `docs/notes/r5_cp_memory_theorem.md`; the rules are stated for n = 192/256 (the GPU run). Raw: `data/r5_cpu/`.

**Falsifier written before the run:** the rank r giving ≤ 1.3 × K4c3-no1111 grows ∝ n; or Proposition B's
energy fractions (f1, f_ones ≈ 0.15–0.35, flat) are off by more than a factor 2.


### Round 5 — rank of the (2,1,1) κ4 slice in its (j | kl) unfolding (rules: docs/notes/r5_cp_memory_theorem.md)

Dense K4c3 with the (1,1,1,1) bulk dropped and the (2,1,1) input slice A[j,k,l] = κ4(z)[j,j,k,l] projected after every linear step onto its top-r (j | kl) directions. Ratios: geometric means over seeds, paired bootstrap CI. 'resid' = ‖μ_cfg − μ_no1111‖² / ‖μ_K3c2 − μ_no1111‖² (truth-free). Theorem A: the chain a CP memory of rank r implements is exactly a211:r.

| n | L | config | meaning | / K4c3 [95% CI] | / K4c3-no1111 | resid | s/run | errors |
|---|---|---|---|---|---|---|---|---|
| 64 | 8 | K4c3-no1111 | bulk dropped, A exact | 1.10 [0.91, 1.23] | 1.00 | — | 17 | 0 |
| 64 | 8 | a211:spec | bulk dropped, A exact (+ spectra; = K4c3-no1111) | 1.10 [0.91, 1.23] | 1.00 | — | 16 | 0 |
| 64 | 8 | a211:1 | A → rank 1 (the λ ⊗ N ceiling) | 1.48 [1.08, 1.79] | 1.35 | — | 16 | 0 |
| 64 | 8 | a211:4 | A → rank 4 | 1.22 [1.05, 1.41] | 1.12 | — | 19 | 0 |
| 64 | 8 | a211:16 | A → rank 16 | 1.11 [0.91, 1.25] | 1.01 | — | 17 | 0 |
| 64 | 8 | a211:64 | A → rank 64 | 1.10 [0.91, 1.23] | 1.00 | — | 17 | 0 |
| 64 | 8 | a211:256 | A → rank 256 | 1.10 [0.91, 1.23] | 1.00 | — | 16 | 0 |
| 96 | 16 | K4c3-no1111 | bulk dropped, A exact | 2.22 [2.22, 2.22] | 1.00 | — | 110 | 0 |
| 96 | 16 | a211:spec | bulk dropped, A exact (+ spectra; = K4c3-no1111) | 2.22 [2.22, 2.22] | 1.00 | — | 115 | 0 |
| 96 | 16 | a211:1 | A → rank 1 (the λ ⊗ N ceiling) | 3.76 [3.76, 3.76] | 1.69 | — | 111 | 0 |
| 96 | 16 | a211:4 | A → rank 4 | 3.19 [3.19, 3.19] | 1.44 | — | 112 | 0 |
| 96 | 16 | a211:16 | A → rank 16 | 2.38 [2.38, 2.38] | 1.07 | — | 113 | 0 |
| 96 | 16 | a211:64 | A → rank 64 | 2.18 [2.18, 2.18] | 0.98 | — | 118 | 0 |
| 96 | 16 | a211:256 | A → rank 256 | 2.22 [2.22, 2.22] | 1.00 | — | 119 | 0 |

(j | kl) spectrum of A = κ4(z)(2,1,1) per layer (means over seeds): f1 = energy fraction of the top direction, f_ones = fraction along the all-ones j-direction (the partial-trace / λ channel of Proposition B), k50/k90 = directions for 50 / 90 %.

| n | layer | f1 | f4 | f16 | f64 | f_ones | k50 | k90 |
|---|---|---|---|---|---|---|---|---|
| 64 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | — | — |
| 64 | 2 | 0.49 | 0.66 | 0.87 | 1.00 | 0.42 | 1 | 21 |
| 64 | 4 | 0.64 | 0.85 | 0.96 | 1.00 | 0.49 | 1 | 8 |
| 64 | 6 | 0.71 | 0.87 | 0.97 | 1.00 | 0.49 | 1 | 6 |
| 96 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | — | — |
| 96 | 2 | 0.48 | 0.60 | 0.78 | 0.97 | 0.43 | 2 | 38 |
| 96 | 4 | 0.49 | 0.66 | 0.85 | 0.98 | 0.41 | 2 | 25 |
| 96 | 6 | 0.73 | 0.86 | 0.95 | 1.00 | 0.56 | 1 | 8 |
| 96 | 8 | 0.70 | 0.84 | 0.94 | 1.00 | 0.57 | 1 | 9 |
| 96 | 10 | 0.73 | 0.84 | 0.96 | 1.00 | 0.60 | 1 | 8 |
| 96 | 12 | 0.78 | 0.86 | 0.96 | 1.00 | 0.67 | 1 | 7 |
| 96 | 14 | 0.80 | 0.89 | 0.97 | 1.00 | 0.69 | 1 | 5 |

Kill rule: the rank r giving ≤ 1.3 × K4c3-no1111 must not grow proportionally with n (192 → 256); Proposition B predicts f_ones and f1 ≈ 0.15–0.35, flat in n.


Per-seed MSE against the MC truth (N = 1e7, mc_seed = 70000 + 100 L + seed):

| n | L | seed | config | MSE |
|---|---|---|---|---|
| 64 | 8 | 0 | K3c2 | 8.899e-05 |
| 64 | 8 | 0 | K4c3 | 3.314e-05 |
| 64 | 8 | 0 | K4c3-no1111 | 3.025e-05 |
| 64 | 8 | 0 | a211:spec | 3.025e-05 |
| 64 | 8 | 0 | a211:1 | 3.586e-05 |
| 64 | 8 | 0 | a211:4 | 3.481e-05 |
| 64 | 8 | 0 | a211:16 | 3.017e-05 |
| 64 | 8 | 0 | a211:64 | 3.025e-05 |
| 64 | 8 | 0 | a211:256 | 3.025e-05 |
| 64 | 8 | 1 | K3c2 | 4.968e-05 |
| 64 | 8 | 1 | K4c3 | 8.661e-06 |
| 64 | 8 | 1 | K4c3-no1111 | 1.063e-05 |
| 64 | 8 | 1 | a211:spec | 1.063e-05 |
| 64 | 8 | 1 | a211:1 | 1.466e-05 |
| 64 | 8 | 1 | a211:4 | 1.072e-05 |
| 64 | 8 | 1 | a211:16 | 1.045e-05 |
| 64 | 8 | 1 | a211:64 | 1.063e-05 |
| 64 | 8 | 1 | a211:256 | 1.063e-05 |
| 64 | 8 | 2 | K3c2 | 5.772e-05 |
| 64 | 8 | 2 | K4c3 | 1.348e-05 |
| 64 | 8 | 2 | K4c3-no1111 | 1.590e-05 |
| 64 | 8 | 2 | a211:spec | 1.590e-05 |
| 64 | 8 | 2 | a211:1 | 2.408e-05 |
| 64 | 8 | 2 | a211:4 | 1.905e-05 |
| 64 | 8 | 2 | a211:16 | 1.691e-05 |
| 64 | 8 | 2 | a211:64 | 1.590e-05 |
| 64 | 8 | 2 | a211:256 | 1.590e-05 |
| 96 | 16 | 0 | K3c2 | 3.051e-05 |
| 96 | 16 | 0 | K4c3 | 2.596e-05 |
| 96 | 16 | 0 | K4c3-no1111 | 5.770e-05 |
| 96 | 16 | 0 | a211:spec | 5.770e-05 |
| 96 | 16 | 0 | a211:1 | 9.758e-05 |
| 96 | 16 | 0 | a211:4 | 8.289e-05 |
| 96 | 16 | 0 | a211:16 | 6.181e-05 |
| 96 | 16 | 0 | a211:64 | 5.651e-05 |
| 96 | 16 | 0 | a211:256 | 5.770e-05 |


## Reading

1. **Rank needed** [measured]: at n = 64, L = 8 (3 seeds) rank 16 of the (j | kl) unfolding reproduces the
   bulk-dropped chain within 1 % (rank 4: 12 %, rank 1: 35 %); at n = 96, L = 16 (1 seed) rank 16 is within
   7 % and rank 64 within 2 %. k90 of the slice is 5–9 at late layers and 21–38 at layers 2–4, decreasing
   with depth. Whether r stays ≈ 16–64 at n = 192/256 is the GPU question (kill rule: r ∝ n).
2. **Proposition B's energy prediction is falsified on the favourable side** [measured]: the top (j | kl)
   direction holds 49–80 % of the slice's energy and the all-ones j-direction (the partial-trace channel)
   40–69 %, both *growing with depth* — not 0.15–0.35. The structural statement (the incoherent part is
   captured at ≤ 1/3 by any single-matrix closure) stands; the incoherent part is simply smaller than
   assumed at these widths. Consequence: a dynamical partial trace P (an n × n state, O(n³)) is worth
   testing against V29's λ·C before the full CP chain — it is the rank-1 all-ones member of the family.
3. **The bulk** [measured]: dropping the (1,1,1,1) slice costs 1.10 [0.91, 1.23] at n = 64, L = 8 but 2.2×
   at n = 96, L = 16 (one seed; round 2: 1.8–1.9× at n = 128–192, L = 16). At depth 16 it is the floor of
   the CP program.
4. n = 96 at depth 16 is still non-perturbative (K4c3/K3c2 = 0.85; the bulk-dropped chain is worse than
   K3c2) — the n = 192/256 run decides; the toy n = 24 dry run (rank 16 worse than rank 4) is the same regime.
