# W5 — Cost model for diagram terms in a factorized representation (n = 1024, L = 16)

> Written by an LLM agent (Claude Code) for team Arieb-007. Status: **[hypothesis]** — a FLOP model
> calibrated on one external point (504aldo's V29 ledger), not a measurement of an implementation.
> Code: `arcmlp/compress/cost_model.py`; per-term table: `results/2026-10-17_w5_term_registry.md`.

## Unit and representation

1 unit = one dense n×n×n matmul = 2n³ FLOPs (2^31 at n = 1024); budget B = 1024 units; the
scoring multiplier is frozen below 0.1 B ≈ 102 units.

Factorized representation (paper §4.3 as implemented in V29 and ARC's `factor_k3.py`): mean and a
dense covariance; the third cumulant as a sum of sources, one born per post-ReLU layer, each carried
as an n×n leg P_s (identity at birth) transported P_s ← T P_s by every later layer, with per-neuron
weights a_s. Live source-layers over the network: S = Σ_{l=1}^{L−1} l = L(L−1)/2 = 120 at L = 16.

## Rule

A term's cost is not its own arithmetic (all diagram terms inside the cap are elementwise or
rank-one, O(n²) = 0 units at n = 1024) but the **families** of factored objects it forces to exist:

| family | provides | raw units (L = 16) | derivation |
|---|---|---|---|
| base | mean, covariance transport W C Wᵀ | 2L = 32 | two dense products per layer |
| k3_legs | κ3 sources (diagonal κ3 = Σ_m P_im³ a_m, the (1,1,1) slice implicitly) | S = 120 | 1 transport per source-layer |
| k3_21 | (2,1) slice D21 = Σ_s (P_s∘P_s) diag(a_s) P_sᵀ | S = 120 | 1 Hadamard-type contraction per source-layer |
| k4_diag | κ4 (4,) slice (W∘W∘W∘W) b | 0 | matrix-vector product |
| k4_coherent | rank-one (2,2) = (4 b̄/n + 4 B̄) 11ᵀ | 0 | O(n²) |
| k4_22_layer | exact (2,2) of this layer's diagonal source (W∘W) diag(b) (W∘W)ᵀ | L = 16 | 1 matmul per layer |
| k4_22_exact | exact (2,2) with κ4 memory: Σ_s (P∘P) diag(b_s)(P∘P)ᵀ | S = 120 (+k3_legs) | 1 per source-layer, legs shared with κ3 |
| k4_31 | (3,1) slice Σ_s (P∘P∘P) diag(b_s) Pᵀ | S = 120 (+k3_legs) | 1 per source-layer |
| k4_211 | (2,1,1) and (1,1,1,1) inputs | 2S = 240 (+k3_legs) | two contractions per source-layer |
| k4_top_newborn | writing multi-neuron κ4 outputs | S = 120 | a new leg family per source |

A selection's raw cost = Σ over the union of families; families are shared, so costs are not
additive over terms (this makes the selection a knapsack with set-up costs).

## Calibration against V29 [external numbers, 504aldo F86 ledger]

V29 = base + k3_legs + k3_21 with the exact 7 → 4 collapse: 4 units per source-layer, i.e.
32 + 4 × 120 = 512 raw units in this model, and it bills 260 units (0.254 B) after the shared-basis
compression of old sources (ranks 384 / 224) and Strassen–Winograd (7/8 per level). Billed cost is
modelled as ENG × raw with ENG = 260/512 = 0.51. The model therefore reproduces V29 by construction;
its predictions for other families (κ4 contractions) inherit the assumption that the same
compression and Strassen savings apply to them (**unverified**).

## Consequences for the knapsack

* Any term reading a κ3 slice needs k3_legs (≥ 77 billed units with base); reading the (2,1) slice
  adds k3_21 (138 billed). The O(1/n)-completeness constraint (every cost ≤ 2 term) includes
  K3(2,1)-reading terms, so **the complete O(1/n) set already costs ≈ 138 billed units > 100**
  under this model — it can only fit the 100-unit floor with additional compression beyond V29's.
* Coherent κ4 terms ((4,) diagonal and rank-one (2,2)) are free; exact (2,2) costs 16–120 raw.
