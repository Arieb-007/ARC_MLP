# GPU round 2 (2026-10-18) — conditional-coherent proxies, spiked covariance: reading of the results

> LLM-generated record (Claude Code) for team Arieb-007. Raw report (A40, `scripts/gpu_run.sh
> --profile round2`, commit 134dcfb) pasted back by Areeb and stored verbatim as
> `results/2026-10-18_gpu_round2_as_received.md`. Truth = organizers' N = 1e9 reference at 1024×16;
> device MC (N = 1e7) for the n ≤ 192 ladders. All numbers [measured] unless labelled.

## Verdicts against the pre-registered rules

| test | rule | result | verdict |
|---|---|---|---|
| CC (2,1,1) proxy, oracle fit, dense chain | dead if < 30 % recovery of the paired→K4c3 log-gap | 74 % / 70 % / 55 % at n = 128/160/192 (L = 16); formula ≈ oracle | alive by the rule — but **recovery falls with width** |
| CC (3,1) proxy | — | oracle 90 % / 90 % / 88 %; formula 91 % / 91 % / 73 % | the λ_a·C_ab form is nearly complete for (3,1) (V29's proxy is justified); the formula loses the memory part at n = 192 |
| light chain + proxies at 1024 vs V29 | CI below 1 → submission candidate | coh2 1.72×, coh2_pboth 1.85×, ex4c2 1.58×, ex4c2_pboth 1.71× **worse** than V29 | **no candidate; the proxies do not help the cheap chain** |
| Algorithm E premise (top-64 eigenvectors < 15 % of the off-diagonal energy; 64 anchors < 10 %) | — | top-16: 85 %, top-64: 94 %, top-256: 98 % at layer 16; 64 anchors explain 48 % | **premise falsified the other way: the spike exists** |
| Algorithm E closure (spike-64 chain > 10× worse) | — | r = 256: 669×, r = 64: 1873×, r = 0: 6009× worse than the full chain | **spiked closure dead**: the 2 % bulk carries the mean |
| K4 kill rule at n = 192, L = 16 (5 seeds) | > 0.25 → dead | 0.13 [0.10, 0.17] (round 1, 3 seeds: 0.10) | GO stands; ladder 0.34 → 0.22 → 0.13 (≈ n^-2.4); at L = 8: 0.06 → 0.04 → 0.03 |

## What the numbers say

1. **Where the (2,1,1) effect lives: in the κ4 memory of h, not in the newborn.** Inside the dense
   chain (which carries every κ4(h) slice exactly) the λ_a·C_bc proxy of the z-slice recovers 55–74 %
   of the gap and the cheap formula matches the oracle (R² ≈ 0.43 of the slice's variance at every
   width). In the light chain (κ4(h) = diagonal + (2,2) only) the same proxies recover 50–63 % at
   n ≤ 192 and nothing at 1024 (coh2_pboth is 7 % *worse* than coh2). The part the formula cannot
   see — the transported random-sign slices of κ4(h) — is what matters at the official shape, and
   its share grows with width (oracle recovery 0.74 → 0.55 from n = 128 to 192). Same lesson as F8:
   at depth the memory dominates the source.
2. **The single random-sign blocks are not enough**: with exact slices but only the single-block
   cost-3 terms the chain is 3–5× worse than the full cap-3 set — the κ4 blocks act together with the
   covariance-edge diagrams of the same order. Zeroing the (1,1,1,1) transmission costs nothing once
   the proxies are used (no1111 ≈ with), unlike in the exact chain (1.8×).
3. **The covariance at 1024×16 is a rank-16 spike plus a bulk, and the bulk is what the mean
   needs.** Off-diagonal energy grows from 0.5× to 9.6× the diagonal energy over 16 layers (the ReLU
   correlation map drives all pairs toward a common mode); 16 eigenvectors hold 85 % of it. But the
   bulk — O(n^-1/2) pairwise correlations — enters the next layer's variances as an incoherent sum of
   n² terms, O(n^-1/2) of the variance, i.e. ~3 %, and the final mean must be right to 1e-4: dropping
   the bulk costs 2.6e-5 even at rank 256. Spectral energy is the wrong metric for this problem; the
   same holds for the κ3 (2,1) matrix (96 % in 64 eigenvectors) and for B (90 %). Any low-rank scheme
   has to keep the bulk's *diagonal effect* exactly (which is what V29's per-source legs do).
4. **The exact paired (4,) slice is the only cheap term that pays at 1024** (ex4c2 = 0.92× coh2,
   one matmul per layer), and no light variant comes within 1.5× of V29. V29's advantage is its
   (3,1) proxy with *memory* (the fitted λ absorbs the transported part the formula misses — point 1)
   plus the D21 feedback.
5. **Noise level of the ladders**: 5 seeds move K4c3/K3c2 at n = 128 from 0.46 to 0.34; the n = 160
   row 0.16 → 0.22; CIs of 3-seed rows were optimistic. The depth-8 ratios (0.06/0.04/0.03) are 4–5×
   smaller than at depth 16: the κ4 truncation's relative gain shrinks with depth.

## Decision

No submission candidate from round 2; AIcrowd 336554 remains the Phase-2 entry. The cheap route to
the κ4 gain is closed in its simplest form (z-side proxies); the next question is whether a
*structured κ4 memory* (κ4(h) carried as diagonal + (2,2) + λ-vector⊗C forms for (3,1) and (2,1,1),
O(n²) state, exactly transportable) recovers the memory part — testable with the same oracle
machinery at n ≤ 192 before any 1024 implementation. The alternative remains the full factored
random-sign κ4 chain (weeks). Everything above is write-up material (due Oct 28).
