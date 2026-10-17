# W7 — pre-registered prediction for n = 256, L = 8 (written 2026-10-17 before any n = 256, L = 8 W7 run; see the commit timestamp of 43a0c5d)

> Written by an LLM agent (Claude Code) for team Arieb-007. Status: **hypothesis** (a prediction from a
> two-width fit); recorded before the data exist so that it can fail.

Inputs [measured, `results/2026-10-17_w7_coherent_even.md`, n ∈ {64, 128}, L = 8, seeds 0–9]:
mean MSE at n = 128: K3c2 1.339e-5, coh2 8.41e-6, ex4c2 6.71e-6; scaling exponents
α(K3c2) = 1.83, α(coh2) = 3.26, α(ex4c2) = 3.16.

Order counting [proven, docs/notes/w7_coherent_even_orders.md]: K3 at cap 2 drops O(1/n) per-entry terms
(MSE ∝ n^-2); the coherent κ4 completes the coherent O(1/n) order, leaving O(n^-3/2) per entry (MSE ∝ n^-3).

Predictions for n = 256, L = 8 (same seeds 0–9, same MC protocol):
1. ratio of mean MSE coh2 / K3c2 = 0.63 · 2^-(3.26-1.83) ≈ **0.23**; falsified if outside [0.12, 0.40].
2. ratio ex4c2 / K3c2 ≈ 0.50 · 2^-(3.16-1.83) ≈ **0.20**; falsified if outside [0.10, 0.36].
3. coh2 beats K3c2 on ≥ 9/10 networks.
