# W5 — pre-registered verification on fresh networks (written before the n = 256 verification runs)

> Written by an LLM agent (Claude Code) for team Arieb-007. Status: **hypothesis**; recorded before the data
> exist (see this file's commit timestamp).

Selection [measured, light tier, n = 128, L = 16, seeds 0–9, exhaustive over the 64 feature sets that fit
100 billed units]: best = {Cpair, K3(3,), K4(4,), K4(2,2)} (77 units), gain over K=2 = +0.159 [+0.014, +0.331]
(natural-log MSE ratio). The O(1/n)-complete set (all but cap3; needs K3(2,1), 138 units) gains +1.98.

Verification set: n = 256, L = 16, fresh seeds 20–29 (weights default_rng(seed), MC N = 4e6, MC seed
70000 + 100·16 + seed), configs: K=2; the selected set; {Cpair, K3(3,)}; K=3 cap 2 = {Cpair, K3(3,), K3(2,1),
K3(1,1,1)}; the O(1/n)-complete light set.

Predictions:
1. The selected set beats K=2 at n = 256 (mean log gain > 0 with the 95% bootstrap CI excluding 0).
2. The O(1/n)-complete set beats the selected set by more than 1.0 in mean log MSE (≈ 2.7×).
3. The selected set's gain at n = 256 is below 0.5 (the budget-feasible features buy little at any width
   we can test) — falsified if ≥ 0.5.
