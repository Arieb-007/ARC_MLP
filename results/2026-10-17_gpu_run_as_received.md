# GPU run — V29 baseline, dense references at 1024×16, κ4 go/no-go, knob sweep, corrector (2026-10-17)

> Verbatim report produced on Areeb's A40 box by `scripts/gpu_run.sh --profile fast` (commit 1be3bb8)
> and pasted back into the session; the reading is in `results/2026-10-17_gpu_run_analysis.md`.
> LLM-generated record (Claude Code) for team Arieb-007.

## Environment

- GPU: NVIDIA A40, 49140 MiB, driver 580.126.20 (4 listed); CUDA Version 13.0; device used: cuda; cupy 14.2.0
- CPU: 64 cores, 3.8e+02 GB; python 3.13.11; profile `fast`
- commit 1be3bb8fce9a (dirty=True); versions {"numpy": "2.4.6", "scipy": "1.18.1", "flopscope": "0.12.3", "whestbench": "0.16.2", "opt-einsum": "3.4.0", "threadpoolctl": "3.7.0"}
- dataset: hf://aicrowd/arc-whestbench-public-2026@v2-phase2 split mini (100 MLPs; reference n_samples = 1000000000; reference MC floor ≈ 7.6e-11 per neuron — every MSE below includes it)

| stage | state | seconds | note |
|---|---|---|---|
| setup | ok | 4 | cupy 14.2.0 |
| smoke | ok | 3 |  |
| data | ok | 31 |  |
| mc_check | ok | 451 |  |
| dense | ok | 1474 |  |
| k4 | ok | 1450 |  |
| cpu_baseline | ok | 336 | idx 0-39 |
| sweep_s6 | ok | 154 | V26_STRASSEN=6 |
| sweep_lam105 | ok | 131 | V17_LAM_SCALE=1.05 |
| corrector | ok | 88 |  |
| report | ok | 0 |  |
| sweep_rold256 | ok | 120 | V21_R_OLD=256 |
| sweep_rold2_128 | ok | 153 | V24_R_OLD2=128 |
| sweep_age3 | ok | 173 | V21_AGE_OLD=3 |
| sweep_age2off | ok | 153 | V24_AGE_OLD2=0 |
| sweep_lam085 | ok | 115 | V17_LAM_SCALE=0.85 |
| sweep_nowk431 | ok | 145 | V17_NO_WK431=1 |

## V29 (+ leaf 8) baseline on the dataset reference

40 MLPs (idx 0–39; 0 failed): raw MSE **2.261e-08** [2.18e-08, 2.34e-08], C/B 0.2592, score 5.859e-09 [5.64e-09, 6.07e-09]; residual max 1.887 s (parallel CPU runs — not the grader's timing), wall 307 s/MLP.
Grader record for comparison: AIcrowd 336554 raw 2.13e-8 at C/B 0.245 → 5.23e-9.

Mean all-layer MSE per layer (V29 vs reference): 6.9e-10, 7.1e-10, 1.1e-09, 1.8e-09, 2.8e-09, 3.9e-09, 5.2e-09, 6.7e-09, 8.1e-09, 9.7e-09, 1.2e-08, 1.3e-08, 1.5e-08, 1.8e-08, 2.0e-08, 2.3e-08

## Knob sweep (Algorithm D): V29 variants on the same MLPs, paired against the baseline

| variant | env | n | raw MSE | C/B | score | MSE ratio vs base [95% CI] | score ratio vs base [95% CI] |
|---|---|---|---|---|---|---|---|
| age2off | `V24_AGE_OLD2=0` | 20 | 2.203e-08 | 0.2714 | 5.979e-09 | 0.988 [0.983, 0.993] | **1.034** [1.029, 1.039] |
| age3 | `V21_AGE_OLD=3` | 20 | 2.519e-08 | 0.2486 | 6.261e-09 | 1.129 [1.101, 1.155] | **1.083** [1.056, 1.108] |
| lam085 | `V17_LAM_SCALE=0.85` | 20 | 2.265e-08 | 0.2592 | 5.871e-09 | 1.015 [1.009, 1.021] | **1.015** [1.009, 1.021] |
| lam105 | `V17_LAM_SCALE=1.05` | 20 | 2.245e-08 | 0.2592 | 5.819e-09 | 1.007 [1.000, 1.012] | **1.007** [1.000, 1.012] |
| nowk431 | `V17_NO_WK431=1` | 20 | 2.624e-08 | 0.2590 | 6.796e-09 | 1.175 [1.140, 1.210] | **1.174** [1.139, 1.209] |
| rold256 | `V21_R_OLD=256` | 20 | 2.947e-08 | 0.2295 | 6.764e-09 | 1.320 [1.291, 1.352] | **1.169** [1.143, 1.198] |
| rold2_128 | `V24_R_OLD2=128` | 20 | 2.573e-08 | 0.2489 | 6.402e-09 | 1.152 [1.131, 1.176] | **1.106** [1.086, 1.129] |
| s6 | `V26_STRASSEN=6` | 20 | 2.231e-08 | 0.2592 | 5.783e-09 | 1.000 [1.000, 1.000] | **1.000** [1.000, 1.000] |

Best score ratio: **s6** (1.000 [1.000, 1.000]) — i.e. the baseline itself (the Strassen knob is masked by the first-predict cap in this harness, see the analysis).

## Dense / light references at the official shape (GPU), against the dataset reference

| variant | n MLPs | raw MSE [95% CI] | ratio to V29 [95% CI] (paired) | ratio to K3c2 (paired) | s/MLP | peak GB | errors |
|---|---|---|---|---|---|---|---|
| K2 | 10 | 4.445e-06 [3.78e-06, 5.08e-06] | 187.630 [160.771, 216.919] (n=10) | 10.110 [8.582, 11.954] | 0 | 4.0 | 0 |
| K3c2 | 10 | 4.291e-07 [3.99e-07, 4.67e-07] | 18.560 [16.956, 20.340] (n=10) | — | 32 | 16.0 | 0 |
| K3c3 | 10 | 4.098e-07 [3.82e-07, 4.43e-07] | 17.740 [16.415, 19.240] (n=10) | 0.956 [0.944, 0.970] | 53 | 16.0 | 0 |
| coh2 | 10 | 3.932e-08 [3.67e-08, 4.20e-08] | 1.703 [1.609, 1.821] (n=10) | 0.092 [0.082, 0.102] | 33 | 24.0 | 0 |
| ex4c2 | 10 | 3.595e-08 [3.37e-08, 3.82e-08] | 1.559 [1.480, 1.658] (n=10) | 0.084 [0.076, 0.093] | 33 | 24.0 | 0 |

Orientation check: DenseProp(K=2) vs the starter-kit covariance closure on the dataset's weights, max |Δ| = 1.8e-06.

Pre-registered predictions (plan of 2026-10-17, from the n ≤ 256 scaling fits):

- dense K=3 cap 2 at 1024x16 = 2.8e-7 [1.9e-7, 4.2e-7]: measured **4.291e-07** [3.99e-07, 4.67e-07] → **falsified**
- K=3 + coherent κ4 at 1024x16 = 1.5e-8 [0.9e-8, 2.6e-8], i.e. below V29's 2.13e-8: measured **3.932e-08** [3.67e-08, 4.20e-08] → **falsified**

Error structure vs V29 (same final-layer neurons): K2: corr(e_K2, e_V29) = -0.03, MSE of the 50/50 blend / V29 = 47.011 [40.252, 54.381]; K3c2: corr = +0.14, blend 5.174 [4.709, 5.644]; K3c3: corr = +0.10, blend 4.886 [4.494, 5.291]; coh2: corr = +0.73, blend 1.152 [1.113, 1.198]; ex4c2: corr = +0.74, blend 1.100 [1.068, 1.137] (n=10 each).

coh2 / V29 paired ratio = 1.703 [1.609, 1.821]: a clean K=3 + coherent κ4 does not beat V29 on these networks.

## Monte-Carlo cross-check of the dataset reference (our device MC vs the organizers' means)

| idx | N ours | MSE(ours, ref) | floor ours | floor ref (n_samples) | expected = sum | ratio | var/neuron | s |
|---|---|---|---|---|---|---|---|---|
| 0 | 1e+07 | 8.923e-09 | 7.867e-09 | 7.867e-11 | 7.945e-09 | 1.12 | 0.079 | 46 |
| 1 | 1e+08 | 8.276e-10 | 8.599e-10 | 8.599e-11 | 9.459e-10 | 0.87 | 0.086 | 449 |

## Factored-κ4 go/no-go (Algorithm A): dense K=4 cap 3 at depth 16 vs device MC truth

Kill rule (pre-registered): K4c3 / K3c2 > 0.25 at the largest width run → the program is dead. Ratios are geometric means over seeds with bootstrap CIs; per-slice rows are ratios to K4c3 (> 1 = removing the slice hurts).

| n | L | seeds | floor | K3c2 | K4c3 | K4c3/K3c2 [95% CI] | K4c2/K3c2 | coh2/K3c2 | no(3,1)/K4c3 | no(2,1,1)/K4c3 | no(1,1,1,1)/K4c3 | paired/K4c3 | +(3,1)/K4c3 | +(2,1,1)/K4c3 | s/run (K4c3) | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 16 | 3 | 7.4e-09 | 2.121e-05 | 1.180e-05 | **0.46** [0.37, 0.66] | 1.30 | 1.56 | 4.47 | 15.12 | 1.89 | 11.90 | 12.10 | 2.64 | 7 | 0 |
| 160 | 16 | 3 | 1.1e-08 | 2.439e-05 | 3.905e-06 | **0.16** [0.09, 0.22] | 0.54 | 0.54 | 4.45 | 8.84 | 1.87 | 7.87 | 7.04 | 2.01 | 18 | 0 |
| 192 | 16 | 3 | 1.1e-08 | 1.927e-05 | 1.989e-06 | **0.10** [0.09, 0.12] | 0.37 | 0.33 | 4.53 | 11.80 | 1.80 | 10.16 | 9.68 | 2.84 | 38 | 0 |

Verdict at the largest width run (n = 192): K4c3/K3c2 = 0.10 vs kill threshold 0.25 → **GO**.

## Residual corrector (Algorithm C) on V29, fitted on networks disjoint from the test networks

Train: 20 networks (20480 neurons); test: 20 networks; features: 23; ridge λ = 0.01.

| model | split | n | MSE before | MSE after | after/before [95% CI] | wins |
|---|---|---|---|---|---|---|
| ridge | cv | 20 | 2.290e-08 | 2.234e-08 | **0.975** [0.967, 0.983] | 17/20 |
| mlp | cv | 20 | 2.290e-08 | 2.256e-08 | **0.984** [0.975, 0.995] | 16/20 |
| ridge | test | 20 | 2.231e-08 | 2.200e-08 | **0.986** [0.973, 0.999] | 15/20 |
| mlp | test | 20 | 2.231e-08 | 2.221e-08 | **0.996** [0.981, 1.011] | 12/20 |

Largest standardized ridge coefficients: d_k2 -3.06e-05, e4 -2.14e-05, D3 +1.66e-05, mu_v29 -1.55e-05, mu_k2 -1.54e-05, phi -1.06e-05, e33 -8.97e-06, m -8.41e-06

Kill rule: < 10% MSE reduction on held-out networks → drop the corrector.
