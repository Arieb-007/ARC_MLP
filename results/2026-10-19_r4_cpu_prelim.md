# Round 4 CPU preliminary — linear response of the κ4 memory at small widths (2026-10-19)

> LLM-generated record (Claude Code) for team Arieb-007. Tool: `experiments/gpu/k4_gonogo.py --device cpu`
> (commit ce09028 code; numpy 2.5.4, scipy 1.18.1; float64 at n = 64, float32 at n = 96). Weights
> W ~ N(0, 2/n) from `default_rng(seed)`; truth = our MC, N = 1e7, mc_seed = 70000 + 100 L + seed
> (floor ≈ 1e-8, < 0.1 % of every MSE below). Rules: `docs/notes/r4_prereg_linear_response.md` — they are
> stated for n = 192/256 at depth 16 (the GPU run); these widths are a preliminary, not the test.
> Raw: `data/r4_cpu/k4/` (git-ignored), log `data/r4_cpu/run.log`.

**Falsifier written before the run:** one-way / K4c3 > 1.3, or nonlinearity > 0.1, at a width where K4c3 beats K3c3.

## Construction error found and fixed before any recorded run (negative result, methodological)

The first one-way probe froze the lower power cumulants of its pK → K conversion at the background
(κ4(h_a) = E h⁴ − 3 (E h²)² − …, where E h² itself reads κ4(z_a)). Measured truth-free on single networks
(seed 3; ‖one-way − K4c3‖ / ‖K4c3 − K3c3‖): 0.62–0.85 at depth 4 and 3.9–10.7 at depth 8 for n = 24–48,
i.e. it lost most of the κ4 transmission at order one. With the probe's own same-layer read-outs in its
conversion: 0.07–0.12 at depth 4 and 0.22–0.30 at depth 8 (n = 24–48). Only the corrected chain is used below.

## Not perturbative: n = 64 at depth 16

K4c3 = 8.13e-3 vs K3c3 = 3.38e-4 (seed 0): the cap-3 K = 4 chain is 24× *worse* than K = 3 at this width
and depth; no gate can be read there (run stopped). Depth 16 needs n ≳ 100 (round 1: K4c3 < K3c2 from n = 128).

## Results


### Round 4 — linear response of the κ4 memory (rules: docs/notes/r4_prereg_linear_response.md)

Three chains in lockstep: background S̄ (= K3c3), κ4 probe driven by S̄ (one-way) or by S (two-way = K4c3), perturbed chain S receiving the probe's read-outs. Ratios: geometric means over seeds, paired bootstrap CI. 'recovery' = 1 − log(cfg/K4c3)/log(K3c3/K4c3) against the MC truth (shown only when K3c3/K4c3 ≥ 1.5); 'resid' = ‖μ_cfg − μ_K4c3‖² / ‖μ_K3c3 − μ_K4c3‖² (truth-free; 0 = K4c3, 1 = K3c3).

| n | L | config | meaning | / K4c3 [95% CI] | recovery | resid | s/run | errors |
|---|---|---|---|---|---|---|---|---|
| 64 | 8 | lr:bg | background only (= K3c3) | 3.28 [1.55, 5.01] | 0.00 | 1.000 | 0 | 0 |
| 64 | 8 | lr:twoway | two-way (= K4c3; check) | 1.00 [1.00, 1.00] | 1.00 | 0.000 | 19 | 0 |
| 64 | 8 | lr:oneway | one-way, all on | 0.83 [0.74, 0.93] | 1.15 | 0.037 | 18 | 0 |
| 64 | 8 | lr:oneway:eps=0.5 | read-outs × 0.5 | 0.86 [0.45, 1.43] | 1.12 | 0.317 | 19 | 0 |
| 64 | 8 | lr:oneway:ro=1 | read-outs at the last step only (direct path) | 3.59 [1.53, 5.64] | -0.08 | 1.003 | 19 | 0 |
| 64 | 8 | lr:oneway:ro=2 | read-outs in the last 2 steps | 3.68 [1.47, 6.13] | -0.10 | 0.898 | 18 | 0 |
| 64 | 8 | lr:oneway:ro=4 | read-outs in the last 4 steps | 2.37 [1.10, 5.28] | 0.27 | 0.328 | 20 | 0 |
| 64 | 8 | lr:oneway:b=1 | 1 generation of births | 3.19 [1.42, 4.97] | 0.02 | 0.981 | 17 | 0 |
| 64 | 8 | lr:oneway:b=2 | 2 generations | 3.13 [1.24, 5.10] | 0.04 | 0.952 | 15 | 0 |
| 64 | 8 | lr:oneway:b=4 | 4 generations | 2.19 [0.75, 4.15] | 0.34 | 0.599 | 16 | 0 |
| 64 | 8 | lr:oneway:b=8 | 8 generations | 0.83 [0.74, 0.93] | 1.15 | 0.037 | 18 | 0 |
| 64 | 8 | lr:oneway:ro=1:b=4 | direct path, 4 generations (cheapest add-on) | 3.24 [1.35, 5.19] | 0.01 | 0.983 | 17 | 0 |
| 64 | 8 | lr:oneway:bf=4 | births only in the first 4 steps | 1.05 [0.58, 1.68] | 0.96 | 0.225 | 16 | 0 |
| 64 | 8 | lr:oneway:bf=8 | births only in the first 8 steps | 0.83 [0.74, 0.93] | 1.15 | 0.037 | 20 | 0 |
| 64 | 8 | lr:oneway:kind=coh | read-outs through paired z-slices only | 2.80 [2.50, 3.20] | 0.13 | 0.317 | 17 | 0 |
| 64 | 8 | lr:oneway:kind=rs | read-outs through random-sign z-slices only | 4.30 [2.14, 6.41] | -0.23 | 1.251 | 19 | 0 |
| 64 | 8 | lr:oneway:rank | one-way + spectra (same result as lr:oneway) | 0.83 [0.74, 0.93] | 1.15 | 0.037 | 22 | 0 |
| 96 | 16 | lr:bg | background only (= K3c3) | 1.04 [1.04, 1.04] | — | 1.000 | 1 | 0 |
| 96 | 16 | lr:twoway | two-way (= K4c3; check) | 1.00 [1.00, 1.00] | — | 0.000 | 124 | 0 |
| 96 | 16 | lr:oneway | one-way, all on | 0.55 [0.55, 0.55] | — | 0.085 | 123 | 0 |
| 96 | 16 | lr:oneway:eps=0.5 | read-outs × 0.5 | 0.26 [0.26, 0.26] | — | 0.374 | 118 | 0 |
| 96 | 16 | lr:oneway:ro=1 | read-outs at the last step only (direct path) | 1.04 [1.04, 1.04] | — | 0.865 | 108 | 0 |
| 96 | 16 | lr:oneway:b=4 | 4 generations | 1.01 [1.01, 1.01] | — | 0.885 | 85 | 0 |
| 96 | 16 | lr:oneway:ro=1:b=4 | direct path, 4 generations (cheapest add-on) | 0.98 [0.98, 0.98] | — | 0.935 | 89 | 0 |
| 96 | 16 | lr:oneway:bf=4 | births only in the first 4 steps | 1.45 [1.45, 1.45] | — | 1.185 | 81 | 0 |

Gates (pre-registered):

- n = 64: P0 two-way/K4c3 = 1.000 (must be 1.00 ± 0.02); G1 one-way/K4c3 = 0.83 (≤ 1.3), nonlinearity = 0.001 (≤ 0.1) → **pass**; G2 direct path + 4 generations recovery = 0.01 (≥ 0.5) → **not viable**, truth-free resid = 0.983; G4 bf=4/bf=8: **finite** (round 3's gen:A divergence was the inconsistent filter), probe-norm growth per layer after the births stop = 0.774 / —.
- n = 96: P0 two-way/K4c3 = 1.000 (must be 1.00 ± 0.02); G1 one-way/K4c3 = 0.55 (≤ 1.3), nonlinearity = 0.003 (≤ 0.1) → **pass**; G2 direct path + 4 generations recovery = — (≥ 0.5) → — (K3c3/K4c3 = 1.04 < 1.5: no gap to recover), truth-free resid = 0.935; G4 bf=4/bf=8: **finite** (round 3's gen:A divergence was the inconsistent filter), probe-norm growth per layer after the births stop = 0.751 / —.

Rank of the index-distinct bulk ((ab)(cd) unfolding; randomized SVD, top 256; means over seeds). 'bulk' = share of the tensor's squared norm on index-distinct entries; 'top-k' = share of the bulk's energy in k directions; k90 = directions for 90 % (— = more than 256). G3 rule: k90 must grow < 1.3× from n = 128 to 192.

| n | tensor | layer | bulk | top-16 | top-64 | top-256 | k50 | k90 |
|---|---|---|---|---|---|---|---|---|
| 64 | birth | 4 | 0.52 | 0.68 | 0.89 | 0.99 | 6 | 69 |
| 64 | probe | 4 | 0.50 | 0.76 | 0.93 | 0.99 | 4 | 46 |


Per-seed MSE against the MC truth (N = 1e7, mc_seed = 70000 + 100 L + seed):

| n | L | seed | config | MSE |
|---|---|---|---|---|
| 64 | 8 | 0 | K3c3 | 5.128e-05 |
| 64 | 8 | 0 | K4c3 | 3.314e-05 |
| 64 | 8 | 0 | lr:bg | 5.128e-05 |
| 64 | 8 | 0 | lr:twoway | 3.314e-05 |
| 64 | 8 | 0 | lr:oneway | 2.454e-05 |
| 64 | 8 | 0 | lr:oneway:eps=0.5 | 1.479e-05 |
| 64 | 8 | 0 | lr:oneway:ro=1 | 5.071e-05 |
| 64 | 8 | 0 | lr:oneway:ro=2 | 4.856e-05 |
| 64 | 8 | 0 | lr:oneway:ro=4 | 3.655e-05 |
| 64 | 8 | 0 | lr:oneway:b=1 | 4.691e-05 |
| 64 | 8 | 0 | lr:oneway:b=2 | 4.102e-05 |
| 64 | 8 | 0 | lr:oneway:b=4 | 2.479e-05 |
| 64 | 8 | 0 | lr:oneway:b=8 | 2.454e-05 |
| 64 | 8 | 0 | lr:oneway:ro=1:b=4 | 4.459e-05 |
| 64 | 8 | 0 | lr:oneway:bf=4 | 1.909e-05 |
| 64 | 8 | 0 | lr:oneway:bf=8 | 2.454e-05 |
| 64 | 8 | 0 | lr:oneway:kind=coh | 9.055e-05 |
| 64 | 8 | 0 | lr:oneway:kind=rs | 7.091e-05 |
| 64 | 8 | 0 | lr:oneway:rank | 2.454e-05 |
| 64 | 8 | 1 | K3c3 | 4.338e-05 |
| 64 | 8 | 1 | K4c3 | 8.661e-06 |
| 64 | 8 | 1 | lr:bg | 4.338e-05 |
| 64 | 8 | 1 | lr:twoway | 8.661e-06 |
| 64 | 8 | 1 | lr:oneway | 7.272e-06 |
| 64 | 8 | 1 | lr:oneway:eps=0.5 | 1.234e-05 |
| 64 | 8 | 1 | lr:oneway:ro=1 | 4.888e-05 |
| 64 | 8 | 1 | lr:oneway:ro=2 | 4.820e-05 |
| 64 | 8 | 1 | lr:oneway:ro=4 | 1.973e-05 |
| 64 | 8 | 1 | lr:oneway:b=1 | 4.303e-05 |
| 64 | 8 | 1 | lr:oneway:b=2 | 4.420e-05 |
| 64 | 8 | 1 | lr:oneway:b=4 | 2.940e-05 |
| 64 | 8 | 1 | lr:oneway:b=8 | 7.272e-06 |
| 64 | 8 | 1 | lr:oneway:ro=1:b=4 | 4.498e-05 |
| 64 | 8 | 1 | lr:oneway:bf=4 | 1.457e-05 |
| 64 | 8 | 1 | lr:oneway:bf=8 | 7.272e-06 |
| 64 | 8 | 1 | lr:oneway:kind=coh | 2.774e-05 |
| 64 | 8 | 1 | lr:oneway:kind=rs | 5.035e-05 |
| 64 | 8 | 1 | lr:oneway:rank | 7.272e-06 |
| 64 | 8 | 2 | K3c3 | 6.121e-05 |
| 64 | 8 | 2 | K4c3 | 1.348e-05 |
| 64 | 8 | 2 | lr:bg | 6.121e-05 |
| 64 | 8 | 2 | lr:twoway | 1.348e-05 |
| 64 | 8 | 2 | lr:oneway | 1.258e-05 |
| 64 | 8 | 2 | lr:oneway:eps=0.5 | 1.365e-05 |
| 64 | 8 | 2 | lr:oneway:ro=1 | 7.218e-05 |
| 64 | 8 | 2 | lr:oneway:ro=2 | 8.267e-05 |
| 64 | 8 | 2 | lr:oneway:ro=4 | 7.115e-05 |
| 64 | 8 | 2 | lr:oneway:b=1 | 6.218e-05 |
| 64 | 8 | 2 | lr:oneway:b=2 | 6.542e-05 |
| 64 | 8 | 2 | lr:oneway:b=4 | 5.588e-05 |
| 64 | 8 | 2 | lr:oneway:b=8 | 1.258e-05 |
| 64 | 8 | 2 | lr:oneway:ro=1:b=4 | 6.537e-05 |
| 64 | 8 | 2 | lr:oneway:bf=4 | 1.592e-05 |
| 64 | 8 | 2 | lr:oneway:bf=8 | 1.258e-05 |
| 64 | 8 | 2 | lr:oneway:kind=coh | 3.370e-05 |
| 64 | 8 | 2 | lr:oneway:kind=rs | 8.640e-05 |
| 64 | 8 | 2 | lr:oneway:rank | 1.258e-05 |
| 96 | 16 | 0 | K3c3 | 2.695e-05 |
| 96 | 16 | 0 | K4c3 | 2.596e-05 |
| 96 | 16 | 0 | lr:bg | 2.695e-05 |
| 96 | 16 | 0 | lr:twoway | 2.596e-05 |
| 96 | 16 | 0 | lr:oneway | 1.430e-05 |
| 96 | 16 | 0 | lr:oneway:eps=0.5 | 6.863e-06 |
| 96 | 16 | 0 | lr:oneway:ro=1:b=4 | 2.554e-05 |
| 96 | 16 | 0 | lr:oneway:bf=4 | 3.757e-05 |
| 96 | 16 | 0 | lr:oneway:ro=1 | 2.703e-05 |
| 96 | 16 | 0 | lr:oneway:b=4 | 2.612e-05 |


## Reading [measured at n = 64 L = 8 (3 seeds) and n = 96 L = 16 (1 seed); hypotheses marked]

1. **G1 passes at both widths.** The κ4 effect is a linear response around the K = 3 trajectory:
   nonlinearity 0.001–0.003, and the one-way chain is *better* than the self-consistent K4c3 against the
   truth (0.83× at n = 64 L = 8; 0.55× at n = 96 L = 16). [hypothesis] The dropped second-order terms are
   part of the cap-3 truncation error at these widths.
2. **The cheap add-on fails.** Read-outs at the last step only (the direct path into the final means)
   reproduce 0–13 % of K4c3's shift (resid 1.00 at n = 64, 0.87 at n = 96); with 4 generations, 2–7 %. Read-outs must be carried at
   many layers (last 4 steps: 67 % of the shift at n = 64, L = 8), i.e. through the covariance and κ3 of
   intermediate layers, not through the final κ4(z_i) — consistent with V29's λ·C structure.
3. **Memory is long and the transport contracts.** Births in the first 4 steps alone reproduce 78 % of the
   shift at n = 64 L = 8 (recovery 0.96) and the probe norm decays 0.77× per layer after they stop (0.75× at n = 96, L = 16); the last 4
   generations alone reproduce 40 %. At n = 96, L = 16 neither the first 4 (resid 1.19) nor the last 4
   generations (resid 0.89) suffice: the memory spans the whole depth. Round 3's gen:A divergence was the inconsistent filter (it dropped
   newborn power cumulants but kept the conversion products that cancel them), **not** an expanding
   transport — round 3's reading ("births regulate the transport") is superseded.
4. **Read-out paths cancel.** Paired-slice read-outs alone 2.8× K4c3, random-sign alone 4.3×; together 0.83×.
5. **The κ4 bulk is low-rank at n = 64.** Index-distinct entries are 50 % of the tensor's norm; 46 directions
   (of 4096) hold 90 % of the probe bulk, 69 for the births (layer 4). Width growth is the GPU question (G3).
6. **Half-strength read-outs beat K4c3** (0.86× at n = 64 L = 8; 0.26× at n = 96 L = 16, one seed) — the cap-3
   κ4 correction overshoots at these widths. Not pre-registered; added to the GPU set as an exploratory
   sweep with its own falsifier (prereg addendum).
