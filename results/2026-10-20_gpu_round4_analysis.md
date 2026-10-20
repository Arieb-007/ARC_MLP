# GPU round 4 (2026-10-20) — linear response of the κ4 memory: reading

> LLM-generated record (Claude Code) for team Arieb-007. Raw report: `results/2026-10-20_gpu_round4_as_received.md`
> (A40, `scripts/gpu_run.sh --profile round4`, commit 8281983). Depth 16, seeds 0–2, device MC truth N = 1e7
> (floor ≈ 1e-8, ≤ 0.5 % of every K4c3 MSE). Rules: `docs/notes/r4_prereg_linear_response.md`.
> 'resid' = ‖μ_cfg − μ_K4c3‖² / ‖μ_K3c3 − μ_K4c3‖² (truth-free; median over seeds; 0 = reproduces K4c3's shift).

## n = 256 did not run: a memory regression of mine

Every n = 256 linear-response row failed within 1 s (OutOfMemoryError, 16 GiB). Cause [measured on the CPU,
`tracemalloc`, n = 72]: the round-4 refactor moved the order-4 mode product into a helper while the caller
still held the input tensor, so three n⁴ buffers coexisted — peak 3.47 n⁴ (dense K4c3) and 3.50 n⁴ (lr chains)
instead of 2.5 n⁴; at n = 256 that is 3 × 16 GiB. Fixed 2026-10-20 (the tensor is handed over in a holder):
peaks 2.50 / 2.55 / 2.56 n⁴; regression test `test_peak_memory_of_order4_chains`. The cached round-3 K4c3 at
n = 256 predates the regression and stands. The only n = 256 number from this run: K3c3 / K4c3 = 13.4
[10.4, 21.8] (lr:bg, 3 seeds).

## Gates at n = 128 and 192

| rule | n = 128 | n = 192 | verdict |
|---|---|---|---|
| P0 two-way / K4c3 (1.00 ± 0.02) | 0.998 | 1.000 | pass — the partition is exact on the GPU |
| G1 one-way / K4c3 (≤ 1.3) | **0.53** [0.47, 0.60] | **0.63** [0.57, 0.70] | pass |
| G1 nonlinearity (≤ 0.1) | 0.001 | 0.000 | pass |
| one-way resid | 0.067 | 0.034 | the dropped feedback shrinks ~ n^-1.7 |
| G2 direct path + 4 generations: recovery (≥ 0.5) | 0.00 (resid 0.97) | 0.00 (resid 0.98) | **fail** — cheap add-on dead |
| G4 births in the first 4 / 8 steps only | finite | finite | pass — round 3's gen:A divergence was my filter |
| G3 k90 growth 128 → 192 (< 1.3×) | | | **fail** at layers 4–12 (1.9–2.5×); pass at layer 14 (1.1×) |

## What the κ4 effect is made of (resid, n = 128 → 192)

| configuration | n = 128 | n = 192 | reading |
|---|---|---|---|
| read-outs at the last step only | 0.94 | 0.96 | the direct path carries 4–6 % |
| read-outs in the last 4 steps | 0.57 | 0.77 | read-out depth needed **grows** with n |
| births in the last 8 generations only | 0.53 | 0.66 | late births matter **less** at larger n |
| births in the first 8 steps only | 0.38 | 0.23 | early births matter **more** |
| births in the first 4 steps only | 0.68 | 0.57 | |
| read-outs through paired z-slices only | 0.42 | **0.18** | the paired-slice channel carries most, and more with n |
| read-outs through random-sign z-slices only | 1.55 | 1.46 | worse than none (cancels against the paired channel) |
| probe-norm factor per layer after births stop (bf = 4 / 8) | 0.73 / 0.74 | 0.84 / 0.95 | transport contraction **weakens** with n |

Reading [measured, 3 seeds, two widths]: the κ4 effect on the final means is born mostly in the first
half of the network, carried by a transport whose per-layer contraction approaches 1 as n grows, and
read out at every layer, chiefly through the paired slices κ4(z)_{aaaa}, κ4(z)_{aabb} of the transported
tensor (into the variances and covariances). Every trend points the same way at n = 1024: longer memory,
deeper read-out, early-born. [hypothesis] At 1024 the full depth must be carried.

## Rank of the index-distinct κ4 bulk ((ab)(cd) unfolding; k90 = directions for 90 % of its energy)

| layer | probe k90, n = 128 | n = 192 | growth | births k90, 128 | 192 | growth |
|---|---|---|---|---|---|---|
| 4 | 206 | > 256 | > 1.24 | > 256 | > 256 | — |
| 8 | 48 | 107 | 2.2× | 62 | 156 | 2.5× |
| 12 | 21 | 39 | 1.9× | 44 | 86 | 2.0× |
| 14 | 21 | 23 | 1.1× | 43 | 58 | 1.35× |

n grew 1.5× (n² 2.25×): at mid depth the rank grows roughly like n² — a fixed-rank compressed memory fails
(G3). At late layers one direction holds > 50 % of the bulk's energy (k50 = 1–2) — [hypothesis] a C ⊗ C-like
component, the structure V29's λ·C proxy represents. Energy share is not read-out relevance: whether the
high-rank remainder matters for the means is not measured (a rank-truncated probe would measure it).

## Read-out strength (post-hoc sweep; addendum to the rules)

| ε | 0.25 | 0.5 | 0.75 | 1 |
|---|---|---|---|---|
| one-way / K4c3, n = 128 | 1.24 | 0.61 | **0.36** | 0.53 |
| one-way / K4c3, n = 192 | 3.92 | 1.74 | 0.65 | **0.63** |

(n = 96, CPU, one seed: best ε = 0.5.) The optimum moves to ε = 1 as n grows and the gain over ε = 1 falls
from 1.5× (n = 128) to 1.0× (n = 192): a finite-width overshoot of the cap-3 truncation, not a transferable
damping constant — the addendum's falsifier is met already between 128 and 192 (it was stated for 192 → 256).

## Consequences

1. **Linear response is the right description** [measured]: the κ4 effect can be computed on a fixed K = 3
   trajectory, and doing so is 1.6–1.9× *more* accurate than the self-consistent K4c3 at n = 128–192.
2. **No cheap add-on exists** [measured]: the direct path and the last few generations carry ≤ 16 % of the
   effect, and the needed depth grows with n.
3. **A fixed-rank compressed κ4 memory fails** [measured, two widths]: mid-depth rank grows ~ n². What remains
   open is whether the read-out-relevant part (the paired-slice channel carrying 82 % at n = 192) is
   low-dimensional even though the bulk is not — the only remaining route for the factored program.
4. **Phase 2** closes 2026-10-21 23:59 UTC: none of this can be implemented in flopscope, validated and
   submitted in time. The program's status for the write-up: physics map complete at n ≤ 192; feasibility
   at 1024 unproven, with the rank growth as the measured obstacle.
