# W7 — coherent even-order cumulant terms (2026-10-17)

> LLM-generated record (Claude Code). Runner `experiments/w7_coherent_even.py`, aggregation
> `experiments/w7_report.py`, theory `docs/notes/w7_coherent_even_orders.md`. Weights: W ~ N(0, 2/n) from
> `default_rng(seed)`; MC truth N = 4e6 per network, MC seed 70000 + 100 L + seed (cached in data/mc).

**Falsifier (written before running).** W7 is not a win if at n = 128 '+κ4 coherent' recovers < 50% of the
final-layer MSE gap between K=3 and dense '+all κ4 slices' at the same diagram cap (bootstrap CI over
networks excluding 50%); κ6/κ8 are reported either way.

MSE = debiased final-layer MSE (MC floor subtracted); mean over networks with 95% bootstrap CI.

## n = 32, L = 8 (10 networks, seeds 0–9; mean MC floor 8.5e-08)

| variant | mean MSE [95% CI] | ratio to K3c2 [95% CI] | mean per-net SE |
|---|---|---|---|
| K3c2 | 3.570e-04 [2.15e-04, 5.19e-04] | 1.000 [1.000, 1.000] | 5.9e-06 |
| coh2 | 4.036e-03 [1.40e-03, 8.48e-03] | 11.306 [3.480, 29.694] | 8.4e-06 |
| ex22 | 4.161e-03 [1.34e-03, 9.14e-03] | 11.655 [3.220, 32.052] | 8.3e-06 |
| ex4c2 | 2.291e-03 [8.70e-04, 4.63e-03] | 6.417 [2.140, 16.396] | 7.2e-06 |
| exallc2 | 2.180e-03 [8.34e-04, 4.46e-03] | 6.107 [2.039, 15.511] | 7.8e-06 |
| K4c2 | 1.773e-03 [9.84e-04, 2.73e-03] | 4.966 [2.752, 8.467] | 7.7e-06 |
| K3c3 | 4.674e-04 [2.60e-04, 7.22e-04] | 1.309 [1.029, 1.532] | 7.6e-06 |
| coh3 | 4.084e-02 [1.31e-02, 7.69e-02] | 114.390 [38.801, 260.109] | 1.3e-05 |
| ex4c3 | 1.581e-02 [5.06e-03, 3.25e-02] | 44.298 [12.261, 122.945] | 1.0e-05 |
| exallc3 | 1.204e-02 [5.55e-03, 1.92e-02] | 33.739 [13.130, 73.619] | 1.1e-05 |
| K4c3 | 1.442e-03 [3.52e-04, 3.41e-03] | 4.039 [0.943, 10.192] | 3.5e-06 |
| coh2_k6 | 4.268e+01 [1.57e-02, 1.28e+02] | 119558.662 [39.211, 459316.918] | 4.5e-05 |
| coh2_k6_k8 | 2.454e+01 [1.51e-02, 7.36e+01] | 68733.935 [36.692, 271896.695] | 3.7e-05 |
| coh2_k3sq | 3.522e-03 [1.51e-03, 6.57e-03] | 9.867 [3.792, 22.960] | 9.4e-06 |
| ex4c2_k6 | 9.443e+00 [6.63e-03, 2.83e+01] | 26450.076 [15.720, 100791.965] | 2.4e-05 |
| ex4c2_k6_k8 | 4.246e-01 [6.39e-03, 1.25e+00] | 1189.278 [14.602, 4592.658] | 1.1e-05 |

Gap recovery at cap 2: (K3 − coh2)/(K3 − K4 dense) = **2.60** [0.92, 5.18] (mean gap -1.42e-03).

Gap recovery at cap 2: (K3 − ex4c2)/(K3 − K4 dense) = **1.37** [0.43, 2.74] (mean gap -1.42e-03).

Gap recovery at cap 2: (K3 − exallc2)/(K3 − K4 dense) = **1.29** [0.37, 2.65] (mean gap -1.42e-03).

Gap recovery at cap 3: (K3 − coh3)/(K3 − K4 dense) = **41.43** [-1469.63, 1634.44] (mean gap -9.74e-04).

Gap recovery at cap 3: (K3 − ex4c3)/(K3 − K4 dense) = **15.75** [-387.99, 536.30] (mean gap -9.74e-04).

Gap recovery at cap 3: (K3 − exallc3)/(K3 − K4 dense) = **11.88** [-350.49, 389.24] (mean gap -9.74e-04).

Recovery of the gap between the standard K=3 (cap 2) and the paper's K=4 (dense, cap 3), mean gap -1.08e-03:
(K3c2 − coh2)/(K3c2 − K4c3) = 3.39 [-36.95, 91.40]; 
(K3c2 − ex22)/(K3c2 − K4c3) = 3.51 [-45.38, 122.02]; 
(K3c2 − ex4c2)/(K3c2 − K4c3) = 1.78 [-20.78, 51.14]; 
(K3c2 − exallc2)/(K3c2 − K4c3) = 1.68 [-21.39, 47.82]; 
(K3c2 − K4c2)/(K3c2 − K4c3) = 1.31 [-13.65, 26.60]; 
(K3c2 − K3c3)/(K3c2 − K4c3) = 0.10 [-1.89, 3.11]; 
(K3c2 − coh3)/(K3c2 − K4c3) = 37.32 [-601.20, 1089.64]; 
(K3c2 − ex4c3)/(K3c2 − K4c3) = 14.25 [-85.09, 278.33]; 
(K3c2 − exallc3)/(K3c2 − K4c3) = 10.77 [-98.96, 241.92]; 

K4c3 worse than K3c2 on 7/10 networks; MSE > 1e-2 (diverged) on 1/10.

coh2_k6: RMS of the added final-layer mean correction = 2.17e+00 (its square 4.7e+00 vs MC floor 8.5e-08).

coh2_k6_k8: RMS of the added final-layer mean correction = 1.67e+00 (its square 2.8e+00 vs MC floor 8.5e-08).

coh2_k3sq: RMS of the added final-layer mean correction = 1.05e-02 (its square 1.1e-04 vs MC floor 8.5e-08).

ex4c2_k6: RMS of the added final-layer mean correction = 1.03e+00 (its square 1.1e+00 vs MC floor 8.5e-08).

ex4c2_k6_k8: RMS of the added final-layer mean correction = 2.81e-01 (its square 7.9e-02 vs MC floor 8.5e-08).

Per-layer coherent κ4 constant β_l = 4 b̄/n + 4 B̄ (z-side, mean over networks) and b̄_l:

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| β | 0.00e+00 | 1.95e-01 | 2.22e-01 | 2.09e-01 | 1.99e-01 | 1.61e-01 | 1.30e-01 | 7.40e-02 |
| b̄ | 0.000 | 1.157 | 0.570 | 0.466 | 0.419 | 0.344 | 0.339 | 0.330 |

## n = 32, L = 16 (10 networks, seeds 0–9; mean MC floor 5.6e-08)

| variant | mean MSE [95% CI] | ratio to K3c2 [95% CI] | mean per-net SE |
|---|---|---|---|
| K3c2 | 3.003e-03 [9.42e-04, 6.25e-03] | 1.000 [1.000, 1.000] | 8.7e-06 |
| coh2 | 4.256e-01 [1.39e-01, 7.88e-01] | 141.750 [31.430, 593.542] | 8.6e-05 |
| ex22 | 1.474e+01 [8.49e-02, 4.02e+01] | 4909.457 [20.930, 32412.893] | 2.6e-04 |
| ex4c2 | 4.961e-01 [3.80e-02, 1.16e+00] | 165.215 [11.790, 623.399] | 6.7e-05 |
| exallc2 | 7.633e-01 [3.87e-02, 2.07e+00] | 254.204 [12.384, 1514.106] | 3.8e-05 |
| K4c2 | 6.152e-01 [1.40e-01, 1.23e+00] | 204.882 [37.154, 888.335] | 9.0e-05 |
| K3c3 | 9.040e-04 [4.81e-04, 1.52e-03] | 0.301 [0.123, 1.049] | 7.5e-06 |
| coh3 | 1.230e+08 [1.06e+03, 3.69e+08] | 40978034352.273 [805712.294, 222461502327.843] | 2.9e-01 |
| ex4c3 | 2.420e+03 [1.54e+01, 7.03e+03] | 805912.950 [11118.935, 5309502.925] | 1.5e-03 |
| exallc3 | 6.537e+04 [2.61e+00, 1.96e+05] | 21770469.511 [608.098, 121862800.076] | 8.6e-03 |
| K4c3 | 4.769e+01 [6.63e+00, 1.12e+02] | 15882.201 [3453.337, 60244.206] | 1.5e-03 |
| coh2_k6 | nan [nan, nan] | nan [nan, nan] | nan |
| coh2_k6_k8 | nan [nan, nan] | nan [nan, nan] | nan |
| coh2_k3sq | nan [nan, nan] | nan [nan, nan] | nan |
| ex4c2_k6 | nan [nan, nan] | nan [nan, nan] | nan |
| ex4c2_k6_k8 | nan [nan, nan] | nan [nan, nan] | nan |

Gap recovery at cap 2: (K3 − coh2)/(K3 − K4 dense) = **0.69** [0.18, 3.31] (mean gap -6.12e-01).

Gap recovery at cap 2: (K3 − ex4c2)/(K3 − K4 dense) = **0.81** [0.05, 2.91] (mean gap -6.12e-01).

Gap recovery at cap 2: (K3 − exallc2)/(K3 − K4 dense) = **1.24** [0.06, 8.37] (mean gap -6.12e-01).

Gap recovery at cap 3: (K3 − coh3)/(K3 − K4 dense) = **2580172.01** [124.27, 3656546.36] (mean gap -4.77e+01).

Gap recovery at cap 3: (K3 − ex4c3)/(K3 − K4 dense) = **50.74** [0.52, 673.31] (mean gap -4.77e+01).

Gap recovery at cap 3: (K3 − exallc3)/(K3 − K4 dense) = **1370.77** [0.07, 17047.15] (mean gap -4.77e+01).

Recovery of the gap between the standard K=3 (cap 2) and the paper's K=4 (dense, cap 3), mean gap -4.77e+01:
(K3c2 − coh2)/(K3c2 − K4c3) = 0.01 [0.00, 0.08]; 
(K3c2 − ex22)/(K3c2 − K4c3) = 0.31 [0.00, 4.13]; 
(K3c2 − ex4c2)/(K3c2 − K4c3) = 0.01 [0.00, 0.07]; 
(K3c2 − exallc2)/(K3c2 − K4c3) = 0.02 [0.00, 0.19]; 
(K3c2 − K4c2)/(K3c2 − K4c3) = 0.01 [0.00, 0.11]; 
(K3c2 − K3c3)/(K3c2 − K4c3) = -0.00 [-0.00, 0.00]; 
(K3c2 − coh3)/(K3c2 − K4c3) = 2580285.56 [101.05, 3663584.09]; 
(K3c2 − ex4c3)/(K3c2 − K4c3) = 50.75 [0.51, 672.66]; 
(K3c2 − exallc3)/(K3c2 − K4c3) = 1370.83 [0.07, 19310.69]; 

K4c3 worse than K3c2 on 10/10 networks; MSE > 1e-2 (diverged) on 10/10.

coh2_k6: RMS of the added final-layer mean correction = nan (its square nan vs MC floor 5.6e-08).

coh2_k6_k8: RMS of the added final-layer mean correction = nan (its square nan vs MC floor 5.6e-08).

coh2_k3sq: RMS of the added final-layer mean correction = nan (its square nan vs MC floor 5.6e-08).

ex4c2_k6: RMS of the added final-layer mean correction = nan (its square nan vs MC floor 5.6e-08).

ex4c2_k6_k8: RMS of the added final-layer mean correction = nan (its square nan vs MC floor 5.6e-08).

Per-layer coherent κ4 constant β_l = 4 b̄/n + 4 B̄ (z-side, mean over networks) and b̄_l:

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| β | 0.00e+00 | 1.95e-01 | 2.22e-01 | 2.09e-01 | 1.99e-01 | 1.61e-01 | 1.30e-01 | 7.40e-02 | 2.64e-02 | -1.06e-01 | -2.86e-01 | -2.12e-01 | -2.84e+00 | -4.76e+00 | -4.10e+00 | -4.73e+00 |
| b̄ | 0.000 | 1.157 | 0.570 | 0.466 | 0.419 | 0.344 | 0.339 | 0.330 | 0.362 | 0.625 | 0.387 | -0.117 | -21.961 | -19.834 | -9.107 | -22.145 |

## n = 64, L = 8 (10 networks, seeds 0–9; mean MC floor 5.5e-08)

| variant | mean MSE [95% CI] | ratio to K3c2 [95% CI] | mean per-net SE |
|---|---|---|---|
| K3c2 | 4.760e-05 [3.65e-05, 6.03e-05] | 1.000 [1.000, 1.000] | 9.9e-07 |
| coh2 | 8.064e-05 [5.84e-05, 1.03e-04] | 1.694 [1.162, 2.473] | 9.6e-07 |
| ex22 | 7.874e-05 [5.71e-05, 1.01e-04] | 1.654 [1.144, 2.388] | 9.4e-07 |
| ex4c2 | 6.011e-05 [4.32e-05, 7.94e-05] | 1.263 [0.868, 1.874] | 9.9e-07 |
| exallc2 | 5.354e-05 [3.83e-05, 7.22e-05] | 1.125 [0.783, 1.694] | 9.4e-07 |
| K4c2 | 5.123e-05 [3.72e-05, 6.82e-05] | 1.076 [0.742, 1.616] | 9.6e-07 |
| K3c3 | 4.220e-05 [3.49e-05, 4.93e-05] | 0.886 [0.749, 1.051] | 1.0e-06 |
| coh3 | 2.324e-04 [1.51e-04, 3.27e-04] | 4.882 [3.429, 7.054] | 1.3e-06 |
| ex4c3 | 1.464e-04 [1.06e-04, 1.89e-04] | 3.076 [2.327, 4.099] | 1.3e-06 |
| exallc3 | 1.308e-04 [9.28e-05, 1.74e-04] | 2.747 [2.085, 3.555] | 1.2e-06 |
| K4c3 | 1.259e-05 [8.49e-06, 1.79e-05] | 0.265 [0.200, 0.333] | 2.9e-07 |
| coh2_k6 | 1.778e-04 [1.10e-04, 2.54e-04] | 3.734 [2.100, 6.131] | 9.9e-07 |
| coh2_k6_k8 | 1.789e-04 [1.10e-04, 2.56e-04] | 3.758 [2.128, 6.193] | 9.9e-07 |
| coh2_k3sq | 8.238e-05 [6.10e-05, 1.04e-04] | 1.731 [1.194, 2.521] | 9.7e-07 |
| ex4c2_k6 | 1.016e-04 [6.08e-05, 1.51e-04] | 2.134 [1.206, 3.460] | 1.0e-06 |
| ex4c2_k6_k8 | 1.019e-04 [6.17e-05, 1.50e-04] | 2.141 [1.223, 3.478] | 1.0e-06 |

Gap recovery at cap 2: (K3 − coh2)/(K3 − K4 dense) = **9.10** [-38.67, 46.09] (mean gap -3.63e-06).

Gap recovery at cap 2: (K3 − ex4c2)/(K3 − K4 dense) = **3.45** [-12.70, 14.69] (mean gap -3.63e-06).

Gap recovery at cap 2: (K3 − exallc2)/(K3 − K4 dense) = **1.64** [-2.48, 4.36] (mean gap -3.63e-06).

Gap recovery at cap 3: (K3 − coh3)/(K3 − K4 dense) = **-6.42** [-11.50, -3.29] (mean gap 2.96e-05).

Gap recovery at cap 3: (K3 − ex4c3)/(K3 − K4 dense) = **-3.52** [-5.95, -1.89] (mean gap 2.96e-05).

Gap recovery at cap 3: (K3 − exallc3)/(K3 − K4 dense) = **-2.99** [-5.29, -1.53] (mean gap 2.96e-05).

Recovery of the gap between the standard K=3 (cap 2) and the paper's K=4 (dense, cap 3), mean gap 3.50e-05:
(K3c2 − coh2)/(K3c2 − K4c3) = -0.94 [-2.11, -0.20]; 
(K3c2 − ex22)/(K3c2 − K4c3) = -0.89 [-2.00, -0.19]; 
(K3c2 − ex4c2)/(K3c2 − K4c3) = -0.36 [-1.25, 0.17]; 
(K3c2 − exallc2)/(K3c2 − K4c3) = -0.17 [-0.97, 0.29]; 
(K3c2 − K4c2)/(K3c2 − K4c3) = -0.10 [-0.86, 0.33]; 
(K3c2 − K3c3)/(K3c2 − K4c3) = 0.15 [-0.08, 0.36]; 
(K3c2 − coh3)/(K3c2 − K4c3) = -5.28 [-8.83, -3.05]; 
(K3c2 − ex4c3)/(K3c2 − K4c3) = -2.82 [-4.49, -1.67]; 
(K3c2 − exallc3)/(K3c2 − K4c3) = -2.38 [-3.71, -1.39]; 

K4c3 worse than K3c2 on 0/10 networks; MSE > 1e-2 (diverged) on 0/10.

coh2_k6: RMS of the added final-layer mean correction = 7.13e-03 (its square 5.1e-05 vs MC floor 5.5e-08).

coh2_k6_k8: RMS of the added final-layer mean correction = 7.17e-03 (its square 5.1e-05 vs MC floor 5.5e-08).

coh2_k3sq: RMS of the added final-layer mean correction = 1.45e-03 (its square 2.1e-06 vs MC floor 5.5e-08).

ex4c2_k6: RMS of the added final-layer mean correction = 4.16e-03 (its square 1.7e-05 vs MC floor 5.5e-08).

ex4c2_k6_k8: RMS of the added final-layer mean correction = 4.20e-03 (its square 1.8e-05 vs MC floor 5.5e-08).

Per-layer coherent κ4 constant β_l = 4 b̄/n + 4 B̄ (z-side, mean over networks) and b̄_l:

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| β | 0.00e+00 | 9.70e-02 | 9.65e-02 | 8.07e-02 | 7.30e-02 | 5.82e-02 | 4.75e-02 | 3.66e-02 |
| b̄ | 0.000 | 1.167 | 0.425 | 0.242 | 0.170 | 0.141 | 0.113 | 0.095 |

## n = 64, L = 16 (10 networks, seeds 0–9; mean MC floor 3.6e-08)

| variant | mean MSE [95% CI] | ratio to K3c2 [95% CI] | mean per-net SE |
|---|---|---|---|
| K3c2 | 1.630e-04 [9.38e-05, 2.49e-04] | 1.000 [1.000, 1.000] | 1.6e-06 |
| coh2 | 4.394e-04 [2.86e-04, 6.17e-04] | 2.696 [1.832, 4.409] | 2.5e-06 |
| ex22 | 4.403e-04 [2.77e-04, 6.24e-04] | 2.702 [1.770, 4.574] | 2.5e-06 |
| ex4c2 | 3.554e-04 [2.23e-04, 5.04e-04] | 2.181 [1.561, 3.369] | 2.5e-06 |
| exallc2 | 3.198e-04 [1.95e-04, 4.57e-04] | 1.962 [1.425, 3.002] | 2.3e-06 |
| K4c2 | 3.784e-04 [2.50e-04, 5.27e-04] | 2.322 [1.700, 3.564] | 2.8e-06 |
| K3c3 | 9.348e-05 [4.94e-05, 1.58e-04] | 0.574 [0.416, 0.727] | 1.7e-06 |
| coh3 | 4.034e-01 [3.45e-02, 9.85e-01] | 2475.383 [218.636, 8765.746] | 1.7e-05 |
| ex4c3 | 2.649e+01 [8.07e-03, 7.91e+01] | 162520.234 [54.415, 707415.833] | 6.3e-05 |
| exallc3 | 1.030e-01 [1.72e-02, 2.21e-01] | 631.887 [137.436, 1154.581] | 1.4e-05 |
| K4c3 | 3.512e-02 [1.05e-03, 1.02e-01] | 215.492 [9.253, 667.850] | 2.9e-06 |
| coh2_k6 | 1.423e-03 [4.58e-04, 2.63e-03] | 8.729 [2.452, 21.819] | 2.6e-06 |
| coh2_k6_k8 | 1.467e-03 [4.67e-04, 2.69e-03] | 9.003 [2.515, 22.573] | 2.6e-06 |
| coh2_k3sq | 4.483e-04 [2.96e-04, 6.22e-04] | 2.751 [1.897, 4.548] | 2.6e-06 |
| ex4c2_k6 | 1.151e-03 [3.40e-04, 2.26e-03] | 7.062 [1.881, 19.762] | 2.6e-06 |
| ex4c2_k6_k8 | 1.159e-03 [3.46e-04, 2.31e-03] | 7.114 [1.907, 19.107] | 2.6e-06 |

Gap recovery at cap 2: (K3 − coh2)/(K3 − K4 dense) = **1.28** [0.94, 1.77] (mean gap -2.15e-04).

Gap recovery at cap 2: (K3 − ex4c2)/(K3 − K4 dense) = **0.89** [0.63, 1.20] (mean gap -2.15e-04).

Gap recovery at cap 2: (K3 − exallc2)/(K3 − K4 dense) = **0.73** [0.49, 0.94] (mean gap -2.15e-04).

Gap recovery at cap 3: (K3 − coh3)/(K3 − K4 dense) = **11.52** [0.68, 648.87] (mean gap -3.50e-02).

Gap recovery at cap 3: (K3 − ex4c3)/(K3 − K4 dense) = **756.19** [0.18, 51454.17] (mean gap -3.50e-02).

Gap recovery at cap 3: (K3 − exallc3)/(K3 − K4 dense) = **2.94** [0.37, 75.65] (mean gap -3.50e-02).

Recovery of the gap between the standard K=3 (cap 2) and the paper's K=4 (dense, cap 3), mean gap -3.50e-02:
(K3c2 − coh2)/(K3c2 − K4c3) = 0.01 [0.00, 0.34]; 
(K3c2 − ex22)/(K3c2 − K4c3) = 0.01 [0.00, 0.34]; 
(K3c2 − ex4c2)/(K3c2 − K4c3) = 0.01 [0.00, 0.23]; 
(K3c2 − exallc2)/(K3c2 − K4c3) = 0.00 [0.00, 0.19]; 
(K3c2 − K4c2)/(K3c2 − K4c3) = 0.01 [0.00, 0.24]; 
(K3c2 − K3c3)/(K3c2 − K4c3) = -0.00 [-0.05, -0.00]; 
(K3c2 − coh3)/(K3c2 − K4c3) = 11.54 [0.67, 690.51]; 
(K3c2 − ex4c3)/(K3c2 − K4c3) = 757.69 [0.18, 55263.51]; 
(K3c2 − exallc3)/(K3c2 − K4c3) = 2.94 [0.34, 77.10]; 

K4c3 worse than K3c2 on 10/10 networks; MSE > 1e-2 (diverged) on 1/10.

coh2_k6: RMS of the added final-layer mean correction = 2.38e-02 (its square 5.7e-04 vs MC floor 3.6e-08).

coh2_k6_k8: RMS of the added final-layer mean correction = 2.45e-02 (its square 6.0e-04 vs MC floor 3.6e-08).

coh2_k3sq: RMS of the added final-layer mean correction = 1.33e-03 (its square 1.8e-06 vs MC floor 3.6e-08).

ex4c2_k6: RMS of the added final-layer mean correction = 1.91e-02 (its square 3.7e-04 vs MC floor 3.6e-08).

ex4c2_k6_k8: RMS of the added final-layer mean correction = 1.97e-02 (its square 3.9e-04 vs MC floor 3.6e-08).

Per-layer coherent κ4 constant β_l = 4 b̄/n + 4 B̄ (z-side, mean over networks) and b̄_l:

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| β | 0.00e+00 | 9.70e-02 | 9.65e-02 | 8.07e-02 | 7.30e-02 | 5.82e-02 | 4.75e-02 | 3.66e-02 | 2.53e-02 | 1.70e-02 | 1.22e-02 | 6.68e-03 | 2.64e-03 | -5.65e-04 | -5.40e-03 | -1.04e-02 |
| b̄ | 0.000 | 1.167 | 0.425 | 0.242 | 0.170 | 0.141 | 0.113 | 0.095 | 0.084 | 0.057 | 0.045 | 0.037 | 0.031 | 0.029 | 0.036 | 0.024 |

## n = 128, L = 8 (10 networks, seeds 0–9; mean MC floor 4.9e-08)

| variant | mean MSE [95% CI] | ratio to K3c2 [95% CI] | mean per-net SE |
|---|---|---|---|
| K3c2 | 1.339e-05 [1.12e-05, 1.58e-05] | 1.000 [1.000, 1.000] | 4.0e-07 |
| coh2 | 8.414e-06 [7.63e-06, 9.18e-06] | 0.628 [0.531, 0.750] | 2.5e-07 |
| ex22 | 8.235e-06 [7.48e-06, 8.97e-06] | 0.615 [0.523, 0.733] | 2.5e-07 |
| ex4c2 | 6.709e-06 [6.09e-06, 7.39e-06] | 0.501 [0.413, 0.619] | 2.4e-07 |
| exallc2 | 6.590e-06 [5.85e-06, 7.38e-06] | 0.492 [0.395, 0.617] | 2.3e-07 |
| K4c2 | 6.181e-06 [5.33e-06, 7.16e-06] | 0.462 [0.365, 0.592] | 2.3e-07 |
| K3c3 | 1.203e-05 [9.93e-06, 1.43e-05] | 0.899 [0.835, 0.971] | 4.1e-07 |
| coh3 | 1.435e-05 [1.25e-05, 1.62e-05] | 1.071 [0.910, 1.265] | 2.8e-07 |
| ex4c3 | 1.118e-05 [9.62e-06, 1.28e-05] | 0.835 [0.688, 1.007] | 2.6e-07 |
| exallc3 | 1.107e-05 [9.52e-06, 1.26e-05] | 0.826 [0.674, 1.006] | 2.5e-07 |
| K4c3 | 8.908e-07 [8.07e-07, 9.77e-07] | 0.067 [0.059, 0.075] | 7.5e-08 |
| coh2_k6 | 9.334e-06 [8.29e-06, 1.05e-05] | 0.697 [0.583, 0.844] | 2.5e-07 |
| coh2_k6_k8 | 9.334e-06 [8.30e-06, 1.05e-05] | 0.697 [0.585, 0.840] | 2.5e-07 |
| coh2_k3sq | 8.740e-06 [7.99e-06, 9.46e-06] | 0.653 [0.554, 0.773] | 2.6e-07 |
| ex4c2_k6 | 7.197e-06 [6.39e-06, 8.09e-06] | 0.537 [0.440, 0.664] | 2.4e-07 |
| ex4c2_k6_k8 | 7.197e-06 [6.43e-06, 8.09e-06] | 0.537 [0.442, 0.666] | 2.4e-07 |

Gap recovery at cap 2: (K3 − coh2)/(K3 − K4 dense) = **0.69** [0.55, 0.80] (mean gap 7.21e-06).

Gap recovery at cap 2: (K3 − ex4c2)/(K3 − K4 dense) = **0.93** [0.86, 1.02] (mean gap 7.21e-06).

Gap recovery at cap 2: (K3 − exallc2)/(K3 − K4 dense) = **0.94** [0.87, 1.01] (mean gap 7.21e-06).

Gap recovery at cap 3: (K3 − coh3)/(K3 − K4 dense) = **-0.21** [-0.47, 0.02] (mean gap 1.11e-05).

Gap recovery at cap 3: (K3 − ex4c3)/(K3 − K4 dense) = **0.08** [-0.13, 0.27] (mean gap 1.11e-05).

Gap recovery at cap 3: (K3 − exallc3)/(K3 − K4 dense) = **0.09** [-0.13, 0.28] (mean gap 1.11e-05).

Recovery of the gap between the standard K=3 (cap 2) and the paper's K=4 (dense, cap 3), mean gap 1.25e-05:
(K3c2 − coh2)/(K3c2 − K4c3) = 0.40 [0.27, 0.50]; 
(K3c2 − ex22)/(K3c2 − K4c3) = 0.41 [0.28, 0.51]; 
(K3c2 − ex4c2)/(K3c2 − K4c3) = 0.53 [0.41, 0.63]; 
(K3c2 − exallc2)/(K3c2 − K4c3) = 0.54 [0.40, 0.64]; 
(K3c2 − K4c2)/(K3c2 − K4c3) = 0.58 [0.44, 0.67]; 
(K3c2 − K3c3)/(K3c2 − K4c3) = 0.11 [0.03, 0.18]; 
(K3c2 − coh3)/(K3c2 − K4c3) = -0.08 [-0.28, 0.10]; 
(K3c2 − ex4c3)/(K3c2 − K4c3) = 0.18 [-0.00, 0.33]; 
(K3c2 − exallc3)/(K3c2 − K4c3) = 0.19 [-0.01, 0.34]; 

K4c3 worse than K3c2 on 0/10 networks; MSE > 1e-2 (diverged) on 0/10.

coh2_k6: RMS of the added final-layer mean correction = 4.92e-04 (its square 2.4e-07 vs MC floor 4.9e-08).

coh2_k6_k8: RMS of the added final-layer mean correction = 4.92e-04 (its square 2.4e-07 vs MC floor 4.9e-08).

coh2_k3sq: RMS of the added final-layer mean correction = 4.41e-04 (its square 1.9e-07 vs MC floor 4.9e-08).

ex4c2_k6: RMS of the added final-layer mean correction = 3.44e-04 (its square 1.2e-07 vs MC floor 4.9e-08).

ex4c2_k6_k8: RMS of the added final-layer mean correction = 3.44e-04 (its square 1.2e-07 vs MC floor 4.9e-08).

Per-layer coherent κ4 constant β_l = 4 b̄/n + 4 B̄ (z-side, mean over networks) and b̄_l:

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| β | 0.00e+00 | 4.83e-02 | 4.56e-02 | 3.68e-02 | 2.97e-02 | 2.27e-02 | 1.72e-02 | 1.44e-02 |
| b̄ | 0.000 | 1.136 | 0.345 | 0.159 | 0.091 | 0.064 | 0.043 | 0.033 |

## n = 128, L = 16 (5 networks, seeds 0–4; mean MC floor 2.1e-08)

| variant | mean MSE [95% CI] | ratio to K3c2 [95% CI] | mean per-net SE |
|---|---|---|---|
| K3c2 | 2.579e-05 [1.40e-05, 3.76e-05] | 1.000 [1.000, 1.000] | 3.8e-07 |
| coh2 | 2.331e-05 [1.42e-05, 3.24e-05] | 0.904 [0.471, 1.735] | 4.4e-07 |
| ex22 | 2.307e-05 [1.41e-05, 3.20e-05] | 0.894 [0.458, 1.681] | 4.3e-07 |
| ex4c2 | 2.099e-05 [1.35e-05, 3.00e-05] | 0.814 [0.457, 1.356] | 4.2e-07 |
| exallc2 | 2.031e-05 [1.27e-05, 3.09e-05] | 0.787 [0.438, 1.187] | 3.9e-07 |
| K4c2 | 2.003e-05 [1.39e-05, 2.75e-05] | 0.777 [0.448, 1.312] | 3.9e-07 |
| K3c3 | 2.137e-05 [1.27e-05, 3.00e-05] | 0.829 [0.690, 1.060] | 4.2e-07 |
| coh3 | 1.349e-04 [8.54e-05, 2.06e-04] | 5.230 [3.303, 7.799] | 7.3e-07 |
| ex4c3 | 1.214e-04 [7.21e-05, 1.99e-04] | 4.706 [2.653, 6.881] | 7.0e-07 |
| exallc3 | 1.160e-04 [6.17e-05, 1.98e-04] | 4.499 [2.400, 6.611] | 6.5e-07 |
| K4c3 | 1.021e-05 [4.26e-06, 1.88e-05] | 0.396 [0.176, 0.618] | 1.7e-07 |
| coh2_k6 | 2.348e-05 [1.44e-05, 3.26e-05] | 0.910 [0.456, 1.751] | 4.4e-07 |
| coh2_k6_k8 | 2.348e-05 [1.44e-05, 3.26e-05] | 0.910 [0.456, 1.751] | 4.4e-07 |
| coh2_k3sq | 2.343e-05 [1.43e-05, 3.26e-05] | 0.908 [0.456, 1.742] | 4.3e-07 |
| ex4c2_k6 | 2.114e-05 [1.38e-05, 3.01e-05] | 0.820 [0.460, 1.368] | 4.2e-07 |
| ex4c2_k6_k8 | 2.114e-05 [1.38e-05, 2.95e-05] | 0.820 [0.470, 1.366] | 4.2e-07 |

Gap recovery at cap 2: (K3 − coh2)/(K3 − K4 dense) = **0.43** [-5.57, 5.39] (mean gap 5.76e-06).

Gap recovery at cap 2: (K3 − ex4c2)/(K3 − K4 dense) = **0.83** [-1.31, 1.90] (mean gap 5.76e-06).

Gap recovery at cap 2: (K3 − exallc2)/(K3 − K4 dense) = **0.95** [-2.43, 1.65] (mean gap 5.76e-06).

Gap recovery at cap 3: (K3 − coh3)/(K3 − K4 dense) = **-10.17** [-26.60, -4.33] (mean gap 1.12e-05).

Gap recovery at cap 3: (K3 − ex4c3)/(K3 − K4 dense) = **-8.96** [-26.05, -3.31] (mean gap 1.12e-05).

Gap recovery at cap 3: (K3 − exallc3)/(K3 − K4 dense) = **-8.48** [-26.71, -2.68] (mean gap 1.12e-05).

Recovery of the gap between the standard K=3 (cap 2) and the paper's K=4 (dense, cap 3), mean gap 1.56e-05:
(K3c2 − coh2)/(K3c2 − K4c3) = 0.16 [-1.44, 0.71]; 
(K3c2 − ex22)/(K3c2 − K4c3) = 0.17 [-1.33, 0.70]; 
(K3c2 − ex4c2)/(K3c2 − K4c3) = 0.31 [-0.71, 0.70]; 
(K3c2 − exallc2)/(K3c2 − K4c3) = 0.35 [-0.40, 0.68]; 
(K3c2 − K4c2)/(K3c2 − K4c3) = 0.37 [-0.61, 0.72]; 
(K3c2 − K3c3)/(K3c2 − K4c3) = 0.28 [-0.11, 0.55]; 
(K3c2 − coh3)/(K3c2 − K4c3) = -7.00 [-15.66, -2.83]; 
(K3c2 − ex4c3)/(K3c2 − K4c3) = -6.14 [-14.60, -2.03]; 
(K3c2 − exallc3)/(K3c2 − K4c3) = -5.79 [-14.46, -1.72]; 

K4c3 worse than K3c2 on 0/5 networks; MSE > 1e-2 (diverged) on 0/5.

coh2_k6: RMS of the added final-layer mean correction = 2.39e-04 (its square 5.7e-08 vs MC floor 2.1e-08).

coh2_k6_k8: RMS of the added final-layer mean correction = 2.39e-04 (its square 5.7e-08 vs MC floor 2.1e-08).

coh2_k3sq: RMS of the added final-layer mean correction = 4.22e-04 (its square 1.8e-07 vs MC floor 2.1e-08).

ex4c2_k6: RMS of the added final-layer mean correction = 2.53e-04 (its square 6.4e-08 vs MC floor 2.1e-08).

ex4c2_k6_k8: RMS of the added final-layer mean correction = 2.53e-04 (its square 6.4e-08 vs MC floor 2.1e-08).

Per-layer coherent κ4 constant β_l = 4 b̄/n + 4 B̄ (z-side, mean over networks) and b̄_l:

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| β | 0.00e+00 | 4.73e-02 | 4.30e-02 | 3.34e-02 | 2.73e-02 | 2.12e-02 | 1.64e-02 | 1.37e-02 | 1.19e-02 | 8.87e-03 | 6.58e-03 | 5.59e-03 | 4.07e-03 | 2.94e-03 | 1.94e-03 | 1.31e-03 |
| b̄ | 0.000 | 1.122 | 0.338 | 0.146 | 0.087 | 0.060 | 0.043 | 0.032 | 0.028 | 0.025 | 0.017 | 0.015 | 0.013 | 0.010 | 0.007 | 0.006 |

## n = 256, L = 8 (10 networks, seeds 0–9; mean MC floor 4.4e-08)

| variant | mean MSE [95% CI] | ratio to K3c2 [95% CI] | mean per-net SE |
|---|---|---|---|
| K3c2 | 3.397e-06 [2.89e-06, 3.97e-06] | 1.000 [1.000, 1.000] | 1.6e-07 |
| coh2 | 7.907e-07 [7.10e-07, 8.71e-07] | 0.233 [0.196, 0.275] | 5.2e-08 |
| ex22 | 7.886e-07 [7.07e-07, 8.68e-07] | 0.232 [0.196, 0.276] | 5.2e-08 |
| ex4c2 | 6.322e-07 [5.81e-07, 6.86e-07] | 0.186 [0.162, 0.216] | 4.7e-08 |
| exallc2 | 6.174e-07 [5.77e-07, 6.62e-07] | 0.182 [0.158, 0.211] | 4.7e-08 |
| K3c3 | 3.104e-06 [2.66e-06, 3.61e-06] | 0.914 [0.897, 0.934] | 1.6e-07 |
| coh3 | 1.114e-06 [9.51e-07, 1.30e-06] | 0.328 [0.266, 0.405] | 5.6e-08 |
| ex4c3 | 8.921e-07 [7.75e-07, 1.04e-06] | 0.263 [0.215, 0.325] | 4.9e-08 |
| exallc3 | 8.754e-07 [7.76e-07, 1.01e-06] | 0.258 [0.214, 0.315] | 5.0e-08 |
| coh2_k6 | 8.178e-07 [7.30e-07, 9.10e-07] | 0.241 [0.202, 0.286] | 5.2e-08 |
| coh2_k6_k8 | 8.178e-07 [7.29e-07, 9.09e-07] | 0.241 [0.202, 0.285] | 5.2e-08 |
| coh2_k3sq | 8.153e-07 [7.33e-07, 8.96e-07] | 0.240 [0.203, 0.285] | 5.3e-08 |
| ex4c2_k6 | 6.482e-07 [5.94e-07, 7.06e-07] | 0.191 [0.165, 0.222] | 4.7e-08 |
| ex4c2_k6_k8 | 6.482e-07 [5.93e-07, 7.05e-07] | 0.191 [0.166, 0.222] | 4.7e-08 |

coh2_k6: RMS of the added final-layer mean correction = 6.51e-05 (its square 4.2e-09 vs MC floor 4.4e-08).

coh2_k6_k8: RMS of the added final-layer mean correction = 6.51e-05 (its square 4.2e-09 vs MC floor 4.4e-08).

coh2_k3sq: RMS of the added final-layer mean correction = 1.13e-04 (its square 1.3e-08 vs MC floor 4.4e-08).

ex4c2_k6: RMS of the added final-layer mean correction = 5.29e-05 (its square 2.8e-09 vs MC floor 4.4e-08).

ex4c2_k6_k8: RMS of the added final-layer mean correction = 5.29e-05 (its square 2.8e-09 vs MC floor 4.4e-08).

Per-layer coherent κ4 constant β_l = 4 b̄/n + 4 B̄ (z-side, mean over networks) and b̄_l:

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| β | 0.00e+00 | 2.39e-02 | 2.25e-02 | 1.90e-02 | 1.51e-02 | 1.20e-02 | 9.62e-03 | 7.42e-03 |
| b̄ | 0.000 | 1.121 | 0.311 | 0.129 | 0.068 | 0.041 | 0.027 | 0.020 |

## n = 256, L = 16 (10 networks, seeds 0–9; mean MC floor 2.3e-08)

| variant | mean MSE [95% CI] | ratio to K3c2 [95% CI] | mean per-net SE |
|---|---|---|---|
| K3c2 | 6.776e-06 [5.66e-06, 7.88e-06] | 1.000 [1.000, 1.000] | 2.4e-07 |
| coh2 | 2.565e-06 [2.23e-06, 2.95e-06] | 0.379 [0.318, 0.455] | 1.0e-07 |
| ex22 | 2.573e-06 [2.24e-06, 2.96e-06] | 0.380 [0.320, 0.456] | 1.1e-07 |
| ex4c2 | 2.333e-06 [2.02e-06, 2.69e-06] | 0.344 [0.296, 0.404] | 9.6e-08 |
| exallc2 | 2.282e-06 [1.97e-06, 2.64e-06] | 0.337 [0.295, 0.388] | 9.3e-08 |
| K3c3 | 5.581e-06 [4.93e-06, 6.24e-06] | 0.824 [0.757, 0.906] | 2.3e-07 |
| coh3 | 6.436e-06 [4.98e-06, 7.95e-06] | 0.950 [0.813, 1.117] | 1.3e-07 |
| ex4c3 | 5.687e-06 [4.44e-06, 7.01e-06] | 0.839 [0.721, 0.979] | 1.2e-07 |
| exallc3 | 5.613e-06 [4.34e-06, 6.94e-06] | 0.828 [0.713, 0.951] | 1.1e-07 |
| coh2_k6 | 2.590e-06 [2.25e-06, 2.97e-06] | 0.382 [0.321, 0.459] | 1.0e-07 |
| coh2_k6_k8 | 2.590e-06 [2.26e-06, 2.98e-06] | 0.382 [0.319, 0.459] | 1.0e-07 |
| coh2_k3sq | 2.594e-06 [2.27e-06, 2.97e-06] | 0.383 [0.324, 0.454] | 1.0e-07 |
| ex4c2_k6 | 2.353e-06 [2.04e-06, 2.70e-06] | 0.347 [0.298, 0.407] | 9.6e-08 |
| ex4c2_k6_k8 | 2.353e-06 [2.04e-06, 2.71e-06] | 0.347 [0.298, 0.407] | 9.6e-08 |

coh2_k6: RMS of the added final-layer mean correction = 7.97e-05 (its square 6.4e-09 vs MC floor 2.3e-08).

coh2_k6_k8: RMS of the added final-layer mean correction = 7.97e-05 (its square 6.4e-09 vs MC floor 2.3e-08).

coh2_k3sq: RMS of the added final-layer mean correction = 1.34e-04 (its square 1.8e-08 vs MC floor 2.3e-08).

ex4c2_k6: RMS of the added final-layer mean correction = 6.79e-05 (its square 4.6e-09 vs MC floor 2.3e-08).

ex4c2_k6_k8: RMS of the added final-layer mean correction = 6.79e-05 (its square 4.6e-09 vs MC floor 2.3e-08).

Per-layer coherent κ4 constant β_l = 4 b̄/n + 4 B̄ (z-side, mean over networks) and b̄_l:

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| β | 0.00e+00 | 2.39e-02 | 2.25e-02 | 1.90e-02 | 1.51e-02 | 1.20e-02 | 9.62e-03 | 7.42e-03 | 5.71e-03 | 4.55e-03 | 3.63e-03 | 2.84e-03 | 2.33e-03 | 1.93e-03 | 1.64e-03 | 1.24e-03 |
| b̄ | 0.000 | 1.121 | 0.311 | 0.129 | 0.068 | 0.041 | 0.027 | 0.020 | 0.015 | 0.011 | 0.008 | 0.007 | 0.005 | 0.004 | 0.004 | 0.003 |

## Scaling MSE(n) ∝ n^{-α} (per variant and depth; fit of log mean MSE on log n over n ≥ 64 — n = 32 is
excluded because the K=4 truncations are unstable there; bootstrap over networks)

| L | variant | widths | α [95% CI] |
|---|---|---|---|
| 8 | K3c2 | [64, 128, 256] | 1.90 [1.69, 2.11] |
| 8 | coh2 | [64, 128, 256] | 3.34 [3.09, 3.51] |
| 8 | ex22 | [64, 128, 256] | 3.32 [3.09, 3.52] |
| 8 | ex4c2 | [64, 128, 256] | 3.29 [3.03, 3.49] |
| 8 | exallc2 | [64, 128, 256] | 3.22 [2.96, 3.44] |
| 8 | K4c2 | [64, 128] | 3.05 [2.52, 3.50] |
| 8 | K3c3 | [64, 128, 256] | 1.88 [1.70, 2.05] |
| 8 | coh3 | [64, 128, 256] | 3.85 [3.52, 4.13] |
| 8 | ex4c3 | [64, 128, 256] | 3.68 [3.44, 3.89] |
| 8 | exallc3 | [64, 128, 256] | 3.61 [3.35, 3.84] |
| 8 | K4c3 | [64, 128] | 3.82 [3.23, 4.36] |
| 8 | coh2_k6 | [64, 128, 256] | 3.88 [3.52, 4.15] |
| 8 | coh2_k6_k8 | [64, 128, 256] | 3.89 [3.54, 4.15] |
| 8 | coh2_k3sq | [64, 128, 256] | 3.33 [3.10, 3.52] |
| 8 | ex4c2_k6 | [64, 128, 256] | 3.65 [3.30, 3.94] |
| 8 | ex4c2_k6_k8 | [64, 128, 256] | 3.65 [3.29, 3.93] |
| 16 | K3c2 | [64, 128, 256] | 2.29 [1.87, 2.62] |
| 16 | coh2 | [64, 128, 256] | 3.71 [3.39, 3.98] |
| 16 | ex22 | [64, 128, 256] | 3.71 [3.37, 3.99] |
| 16 | ex4c2 | [64, 128, 256] | 3.63 [3.29, 3.90] |
| 16 | exallc2 | [64, 128, 256] | 3.57 [3.21, 3.85] |
| 16 | K4c2 | [64, 128] | 4.24 [3.49, 4.98] |
| 16 | K3c3 | [64, 128, 256] | 2.03 [1.57, 2.41] |
| 16 | coh3 | [64, 128, 256] | 7.97 [6.31, 8.66] |
| 16 | ex4c3 | [64, 128, 256] | 11.08 [5.37, 11.92] |
| 16 | exallc3 | [64, 128, 256] | 7.08 [5.77, 7.67] |
| 16 | K4c3 | [64, 128] | 11.75 [6.48, 13.89] |
| 16 | coh2_k6 | [64, 128, 256] | 4.55 [3.67, 4.99] |
| 16 | coh2_k6_k8 | [64, 128, 256] | 4.57 [3.76, 5.03] |
| 16 | coh2_k3sq | [64, 128, 256] | 3.72 [3.39, 3.98] |
| 16 | ex4c2_k6 | [64, 128, 256] | 4.47 [3.56, 4.98] |
| 16 | ex4c2_k6_k8 | [64, 128, 256] | 4.47 [3.62, 4.97] |
## Decomposition: coherent replacement of one κ4 slice inside the dense reference

Slice hooks replace the pre-activation (4,) slice by W⁴b + 12B̄ and/or the (2,2) slice by 4b̄/n + 4B̄
(b, B from the dense post-ReLU κ4), or zero the random-sign slices (3,1), (2,1,1), (1,1,1,1) (cap 3).
Falsifier (written before running) for '(2,2) rank-one is adequate, (4,) replacement is what fails':
replacing only (2,2) moves the MSE more than replacing only (4,).

n = 64, L = 8, seeds [0, 1, 2, 3, 4] — mean MSE (ratio of means to exact at the same cap):

| variant | mean MSE | ratio to exact |
|---|---|---|
| c2_ex_ex | 4.460e-05 | 1.00 [1.00, 1.00] |
| c2_ex4_coh22 | 5.681e-05 | 1.27 [1.08, 1.44] |
| c2_coh4_ex22 | 7.316e-05 | 1.64 [1.30, 1.88] |
| c2_coh_coh | 9.029e-05 | 2.02 [1.62, 2.37] |
| c3_full | 1.556e-05 | 1.00 [1.00, 1.00] |
| c3_ex4_coh22 | 5.401e-05 | 3.47 [2.45, 5.28] |
| c3_coh_coh | 2.962e-04 | 19.04 [14.74, 26.45] |
| c3_rs0 | 1.294e-04 | 8.32 [7.25, 9.95] |

## n = 1024: newborn coherent κ4 constant β_l vs V29's fitted λ_l (4 self-baked MLPs, seeds 0–3)

β_l = 4 b̄/n + 4 B̄ from the Gaussian-input (cap 2) newborn κ4 of each post-ReLU layer along the K=2 closure;
λ_l = V29's table × 0.95. Falsifier for 'β is V29's λ': λ/β varies by > 2× across layers.

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| β | 6.0e-03 | 1.8e-03 | 8.3e-04 | 4.4e-04 | 2.2e-04 | 1.1e-04 | 9.6e-05 | 4.6e-05 | 3.0e-05 | 1.6e-05 | 2.2e-05 | 7.6e-07 | 5.5e-06 | 6.3e-06 | -1.9e-06 | 3.9e-06 |
| λ (V29) | 4.7e-03 | 7.7e-03 | 9.3e-03 | 1.0e-02 | 1.0e-02 | 1.0e-02 | 1.0e-02 | 9.5e-03 | 9.2e-03 | 8.7e-03 | 8.3e-03 | 8.0e-03 | 7.7e-03 | 7.3e-03 | 6.9e-03 | 6.9e-03 |
| λ/β | 0.792 | 4.25 | 11.4 | 23 | 46.9 | 94.8 | 110 | 268 | 355 | 946 | 661 | 1.36e+03 | 332 | 1.34e+03 | 3.05e+03 | 4.22e+03 |
| frac b<0 | 0.00 | 0.14 | 0.22 | 0.26 | 0.30 | 0.32 | 0.33 | 0.36 | 0.38 | 0.38 | 0.39 | 0.39 | 0.40 | 0.40 | 0.41 | 0.42 |

Outcome: falsified — λ/β grows from 0.79 to 4224: the newborn
β decays ~10³× over 16 layers (a growing fraction of neurons have t > 0.76, where κ4 < 0, so b̄ cancels)
while λ is flat; V29's λ multiplies C_off in a partial-trace core and is not the coherent κ4 of z.

