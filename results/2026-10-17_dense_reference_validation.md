# Dense cumulant reference — validation (2026-10-17)

> LLM-generated record (Claude Code). Code: `arcmlp/reference/dense_cumulants.py` (transcribed from ARC's
> reference implementation, MIT, commit 93d091a; the paper PDF was unreachable). Scripts:
> `experiments/v1_dense_validation.py`, `experiments/v1_report.py`; unit tests `tests/test_dense.py`,
> `tests/test_relu_moments.py`. Raw outputs: `data/v1/` (git-ignored) mirrored in the .json.

**Falsifier (written before running).** The reference is wrong if (A) at n = 16 or 32 the Kt=4 entrywise
deviation of an order exceeds the Kt=3 deviation, or the Kt=4 deviation does not shrink from n = 16 to
n = 32; or (B) Kt=4 has a larger final-layer MSE than Kt=3 on average at n = 64, L = 8.

**Independent oracle.** Running ARC's own code needs torch, whose CPU index is blocked here; the
transcription is validated by exact identities instead: Kt=2 equals the starter-kit closure (1e-15);
covariance-only diagrams converge to the exact bivariate ReLU covariance (2-D quadrature) with error
1.9e-4 -> 3.5e-18 as the cap grows 1 -> 9 at c = 0.05; single-neuron kappa3/kappa4 coefficients give
the predicted c^3 error law on z = a + b g + c (g^2 - 1) (slope 3.11 with cap 4, 1.99 without kappa3^2).

## A. Entrywise vs MC full cumulant tensors, n = 16, L = 4, seeds [0, 1, 2], N = 4e+07 (10 independent MC runs → per-entry SE)

Relative RMS deviation ‖K_prop − K_MC‖/‖K_MC‖ (mean over seeds); MC SE/RMS in the last column.

| layer | order | Kt=2 | Kt=3 | Kt=4 | MC SE/RMS |
|---|---|---|---|---|---|
| 0 | κ2 | 0.072 | 0.002 | 0.002 | 0.0006 |
| 0 | κ3 | 1.000 | 0.022 | 0.009 | 0.0014 |
| 0 | κ4 | 1.000 | 1.000 | 0.052 | 0.0040 |
| 1 | κ2 | 0.273 | 0.059 | 0.025 | 0.0005 |
| 1 | κ3 | 1.000 | 0.240 | 0.102 | 0.0011 |
| 1 | κ4 | 1.000 | 1.000 | 0.259 | 0.0026 |
| 2 | κ2 | 0.340 | 0.088 | 0.055 | 0.0004 |
| 2 | κ3 | 1.000 | 0.237 | 0.154 | 0.0010 |
| 2 | κ4 | 1.000 | 1.000 | 0.410 | 0.0028 |
| 3 | κ2 | 0.421 | 0.116 | 0.107 | 0.0004 |
| 3 | κ3 | 1.000 | 0.295 | 0.259 | 0.0009 |
| 3 | κ4 | 1.000 | 1.000 | 0.573 | 0.0021 |

Per-slice Kt=4 deviation at the last layer (layer 3), relative to that slice's RMS:

| slice | rel. RMS dev (Kt=3) | rel. RMS dev (Kt=4) | MC SE/RMS |
|---|---|---|---|
| κ2 (2,) | 0.060 | 0.067 | 0.0003 |
| κ2 (1, 1) | 0.174 | 0.150 | 0.0005 |
| κ3 (3,) | 0.262 | 0.132 | 0.0007 |
| κ3 (2, 1) | 0.287 | 0.236 | 0.0009 |
| κ3 (1, 1, 1) | 0.390 | 0.455 | 0.0011 |
| κ4 (4,) | 1.000 | 0.298 | 0.0019 |
| κ4 (3, 1) | 1.000 | 0.419 | 0.0021 |
| κ4 (2, 2) | 1.000 | 0.678 | 0.0023 |
| κ4 (2, 1, 1) | 1.000 | 0.660 | 0.0023 |
| κ4 (1, 1, 1, 1) | 1.000 | 0.915 | 0.0023 |

## A. Entrywise vs MC full cumulant tensors, n = 32, L = 4, seeds [0, 1, 2], N = 4e+06 (10 independent MC runs → per-entry SE)

Relative RMS deviation ‖K_prop − K_MC‖/‖K_MC‖ (mean over seeds); MC SE/RMS in the last column.

| layer | order | Kt=2 | Kt=3 | Kt=4 | MC SE/RMS |
|---|---|---|---|---|---|
| 0 | κ2 | 0.054 | 0.002 | 0.002 | 0.0024 |
| 0 | κ3 | 1.000 | 0.020 | 0.009 | 0.0075 |
| 0 | κ4 | 1.000 | 1.000 | 0.045 | 0.0262 |
| 1 | κ2 | 0.205 | 0.035 | 0.012 | 0.0019 |
| 1 | κ3 | 1.000 | 0.184 | 0.059 | 0.0058 |
| 1 | κ4 | 1.000 | 1.000 | 0.201 | 0.0159 |
| 2 | κ2 | 0.271 | 0.062 | 0.022 | 0.0018 |
| 2 | κ3 | 1.000 | 0.260 | 0.110 | 0.0048 |
| 2 | κ4 | 1.000 | 1.000 | 0.273 | 0.0121 |
| 3 | κ2 | 0.347 | 0.072 | 0.029 | 0.0016 |
| 3 | κ3 | 1.000 | 0.251 | 0.128 | 0.0041 |
| 3 | κ4 | 1.000 | 1.000 | 0.283 | 0.0108 |

Per-slice Kt=4 deviation at the last layer (layer 3), relative to that slice's RMS:

| slice | rel. RMS dev (Kt=3) | rel. RMS dev (Kt=4) | MC SE/RMS |
|---|---|---|---|
| κ2 (2,) | 0.035 | 0.013 | 0.0010 |
| κ2 (1, 1) | 0.087 | 0.035 | 0.0018 |
| κ3 (3,) | 0.231 | 0.044 | 0.0027 |
| κ3 (2, 1) | 0.225 | 0.095 | 0.0037 |
| κ3 (1, 1, 1) | 0.278 | 0.166 | 0.0048 |
| κ4 (4,) | 1.000 | 0.099 | 0.0078 |
| κ4 (3, 1) | 1.000 | 0.191 | 0.0094 |
| κ4 (2, 2) | 1.000 | 0.214 | 0.0088 |
| κ4 (2, 1, 1) | 1.000 | 0.254 | 0.0107 |
| κ4 (1, 1, 1, 1) | 1.000 | 0.385 | 0.0122 |

## B. Final-layer MSE vs MC, n = 64, L = 8, N = 2e7 (debiased; ± MC SE)

| seed | MC floor | Kt=2 | Kt=3 | Kt=4 | Kt=3/Kt=4 |
|---|---|---|---|---|---|
| 0 | 1.9e-08 | 7.103e-04 ± 4e-06 | 8.781e-05 ± 5e-07 | 3.363e-05 ± 3e-07 | 2.61 |
| 1 | 4.9e-09 | 2.866e-04 ± 1e-06 | 5.003e-05 ± 3e-07 | 8.468e-06 ± 7e-08 | 5.91 |
| 2 | 1.6e-08 | 4.620e-04 ± 3e-06 | 5.685e-05 ± 8e-07 | 1.328e-05 ± 1e-07 | 4.28 |
| 3 | 1.1e-08 | 8.163e-04 ± 4e-06 | 4.869e-05 ± 7e-07 | 6.023e-06 ± 1e-07 | 8.08 |
| 4 | 1.5e-08 | 7.325e-04 ± 5e-06 | 3.401e-05 ± 4e-07 | 1.701e-05 ± 1e-07 | 2.00 |
| geo-mean | | 5.623e-04 | 5.288e-05 | 1.311e-05 | 4.03 |

## C. Cross-check against experiments/conditional.py (seed 3, n = 256, L = 8, same MC protocol)

Kt=2 final-layer MSE = 3.5635e-05 (conditional.py: 3.563e-05).

## §1.2 ReLU closed forms (tests/test_relu_moments.py, all passing)

* κ2…κ8 of ReLU(m + sZ) from the raw-moment recurrence agree with adaptive quadrature to 1e-12 and with a
  61-digit mpmath evaluation; Wick coefficients w(k, p) = E[∂^k ReLU(z)^p] (k ≤ 8, p ≤ 4) agree with the
  Hermite-projection quadrature to 1e-12. [proven identities, checked numerically]
* **Positivity check (falsified).** The brief's hypothesis that κ4 and κ6 of ReLU(t + Z) are ≥ 0 on
  t ∈ [−8, 8] is false: κ4 ≥ 0 iff t ≤ 0.7556, and κ6 changes sign at t ≈ −0.316, 0.961, 2.329
  (measured at 61 digits and confirmed by 2e7-sample MC; pinned by `test_kappa4_kappa6_sign_structure`).
  Consequence: per-neuron κ4 at pre-activation means t > 0.76 is negative, so 'coherent' κ4 sums are
  not sign-definite and can cancel across neurons.

## Outcome

Falsifier (A) is triggered at the strict reading in the cells listed (all at n = 16, the deepest layer, by ≤ 0.011 absolute; not at n = 32); the per-seed-mean ordering Kt=4 < Kt=3 and the shrinkage with n hold. (A) Kt=4 entrywise deviation exceeds Kt=3 (beyond 3 MC SE) in 2 (n, seed, layer, order) cells: [(16, 1, 3, 3), (16, 2, 3, 2)]; Kt=4 deviation at the last layer, mean over seeds, n = 16 → 32: κ2 0.11 → 0.03, κ3 0.26 → 0.13, κ4 0.57 → 0.28. (B) Kt=4 < Kt=3 final-layer MSE on 5/5 networks at n = 64, L = 8 (geo-mean ratio 4.03).

**Caveats [measured, see results/2026-10-17_w7_coherent_even.md].** (i) The deviations grow with depth at
fixed n (n = 16: Kt=4 κ4 deviation 0.05 → 0.57 over four layers), as expected of a truncation whose dropped
terms are O(n^{-2}) per layer but accumulate. (ii) At L = 16 the paper-faithful Kt=4, cap-3 truncation is
numerically unstable for n ≤ 64 (final-layer MSE 1e-3 to 3e-1 on several seeds at n = 64, divergence at
n = 32), while Kt=3 stays stable; §1.3's 'stop if K=4 is worse than K=3' rule was evaluated at n = 64,
L = 8 as specified, where it is not triggered.

