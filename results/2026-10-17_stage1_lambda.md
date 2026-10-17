# Stage 1 — can V29's fitted κ4 coefficient λ_l be replaced by an analytic one? (E4, E4b, E4c, E5)

> LLM-generated record (Claude Code), 2026-10-17. Raw outputs and provenance in
> `2026-10-17_stage1_lambda.json`; scripts `experiments/e4_lambda_mc.py`, `e4b_lambda_tree_vs_mc.py`,
> `e4c_lambda_recursion.py`, `e5_lambda_v29_vs_analytic.py`; theory `arcmlp/closure/kappa4_lambda.py`.
> **Outcome: the pre-registered stop rule triggered — no analytic-λ submission.**

**Object.** V29 regenerates the off-diagonal κ4 core as G_off = λ_l C_post,off. In the reference
chain (504aldo `lean_k3_aug.py:1262`) this core is the r=1 harmonic projection
G_ij = c_A Σ_k κ4(h_i, h_j, h_k, h_k), c_A = 6/(n+4) [read from code]. It is measurable exactly by
Monte Carlo through Σ_k κ4_ijkk = E[a_i a_j (|a|² − tr C)] − 2(C²)_ij, a = h − E h [proven identity].

## E4 — is the one-scalar law a property of the *true* κ4? (n = 128, L = 16, 3 networks, N = 4e6)

Falsifier: R² < 0.5 at most layers or > 20% spread across seeds. **Not falsified.**
λ*_l = ⟨G_off, C_off⟩/‖C_off‖² per post-ReLU layer [measured]:

| layer | 0 | 1 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|
| seed 0 λ* | 0.118 | 0.140 | 0.148 | 0.144 | 0.196 | 0.184 | 0.392 | 0.351 | 0.305 | 0.211 |
| seed 1 λ* | 0.109 | 0.124 | 0.121 | 0.131 | 0.170 | 0.190 | 0.177 | 0.183 | 0.146 | 0.152 |
| seed 2 λ* | 0.118 | 0.137 | 0.142 | 0.152 | 0.154 | 0.138 | 0.167 | 0.110 | 0.096 | 0.112 |
| R² range (3 seeds) | 0.85–0.87 | 0.67–0.72 | 0.67–0.70 | 0.72–0.76 | 0.76–0.80 | 0.72–0.80 | 0.85–0.89 | 0.86–0.95 | 0.89–0.95 | 0.87–0.94 |

The true κ4 partial-trace core is ∝ C_post,off with R² 0.67–0.95: one scalar per layer is a real
property of He-ReLU networks, not an artefact of 504aldo's chain. λ* is network-specific at depth
(seed 0 vs 2 differ 2–3× at layers 10–14). MC repeatability: N = 1e6 vs 4e6 on seed 0 agree to ≤ 3%.

## E4b — newborn tree-level theory (no memory) vs λ*

Derivation (FINDINGS F8): at leading order in pre-activation correlations, for Gaussian z,
λ = c_A⟨A + B + D, C_off⟩/‖C_off‖² with A the coherent k-sum, B a mixed term and D the (3,1)
slice, D_ij = c_ij[Φ_j k3'_i + Φ_i k3'_j], k3' = ∂_m κ3(ReLU) = 3(1−Φ)(E h² − 2(E h)²) [proven].
Falsifier: misses λ* by > 25% at > 1/3 of layers. **Falsified as a model of depth; exact at layer 0.**

- Layer 0 (z exactly Gaussian): theory/MC = 0.99, 0.99, 0.99 (seeds 0, 1, 2); 85% from D (the (3,1)
  slice), 14% from A, ≈ 0.5% from B [measured].
- Layer 1: 0.23–0.30; layers ≥ 2: |ratio| ≤ 0.09 except seed 1 layer 2 (0.28); newborn terms change
  sign at depth [measured]. The κ4 core at depth is **not** newborn. The O(n²) mean-field version
  of the A term (`lambda_fast`) agrees with the exact one to ≤ 0.01 in ratio at layer 0.

## E4c — memory transport + newborn recursion

Leading-order transport (FINDINGS F8): the partial trace moves through z = Wh like a covariance with
the coherent weight Σ_k w_k W_kc W_kd ≈ 2 w̄ δ_cd, w_k = Φ_k − μ_kφ_k/s_k, giving
λ_l = 2w̄_l [λ_{l−1}‖C_{l−1,off}‖² + Σ_m g_{l−1,m} v_{l−1,m}]/‖C_{l−1}‖²_F + λ_newborn,l
(g = diagonal of the core; the bracket is exactly the pre-activation regression V29's LAM table was
fitted to). Falsifier: one-step ratio outside [0.8, 1.25] at > 1/3 of layers. **Falsified.**

- One-step ratio (λ_{l−1}, g from MC): layers 1–15, 3 seeds: 0.44–1.07, median ≈ 0.68 — the
  transport law explains about two thirds; a source term of ~30% is missing (candidate: κ3(z)-driven
  contributions to κ4(h), not derived).
- Closed loop: errors compound, ratio decays to 0.07–0.2 by layer 15.

## E5 — analytic λ vs V29's λ on a 1024×16 network (V29 + leaf 8 trajectory)

Falsifier: |analytic/adapted − 1| > 0.2 at > 1/3 of layers. **Falsified (14 of 15 layers).**

| post-layer | 0 | 1 | 2 | 3 | 5 | 8 | 11 | 14 |
|---|---|---|---|---|---|---|---|---|
| V29 λ (adapted, used) | 4.72e-3 | 7.69e-3 | 9.35e-3 | 9.74e-3 | 1.03e-2 | 9.58e-3 | 8.54e-3 | 7.47e-3 |
| newborn analytic λ | 1.46e-2 | 4.10e-3 | 2.21e-3 | 9.7e-4 | 4.2e-4 | 1.4e-4 | 0 (clipped < 0) | 3.5e-4 |
| ratio | 3.10 | 0.53 | 0.24 | 0.10 | 0.04 | 0.01 | 0 | 0.05 |

At layer 0, where the theory is exact (E4b), the physical coefficient at n = 1024 is 1.46e-2
(= c_A·2.5, consistent with the n = 128 value c_A·2.6), but **V29 uses 4.7e-3 — 3× smaller**
[measured]. Hence V29's λ is an *effective* parameter of its truncated K=3 + regeneration chain
(it absorbs the missing terms and the use-side approximations), not the physical κ4/covariance
coefficient [inference; hypothesis until tested by plugging the physical λ into the chain].

## Decision

Per the stop rule agreed before the work (> 20% off at most layers → do not submit), no
analytic-λ submission is made for Phase 2. What stands for the write-up: (1) the one-scalar law
holds for the true κ4 (R² 0.67–0.95); (2) an exact, fit-free derivation of λ at layer 0 (1%),
dominated by the (3,1) slice; (3) at depth the core is transported memory, two thirds of which a
coherent 2w̄ transport law explains; (4) V29's fitted λ is 3× below the physical value where the
physics is exact — so its success is effective, not mechanistic.
