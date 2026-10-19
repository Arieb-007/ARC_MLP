# Round 4 — pre-registration: linear response of the κ4 memory (written 2026-10-19, before any round-4 run)

> Written by an LLM agent (Claude Code) for team Arieb-007. Status: **hypothesis** throughout; recorded
> before the data exist so that each rule can fail. Results go to `results/<date>_gpu_round4_*.md`.

## Question

Can the effect of the κ4 memory on the final-layer means be computed as a *linear response around a
K = 3 trajectory* — births and transport evaluated on that trajectory, read-outs added afterwards — and
if so, through which paths and over how many generations? This is the premise of the adjoint read-out
add-on to V29 (proposal of 2026-10-19): if it fails, that add-on is dead; if it passes, its cost is set by
the read-out depth and the generation depth measured here.

## Construction (`arcmlp/estimators/linear_response.py`)

The dense K = 4 cap-3 chain (K4c3) is split, term by term, into three chains run in lockstep:

- background S̄: every term with output order ≤ 3 and no κ4 block = the dense K = 3 cap-3 chain (K3c3);
- κ4 probe P: every order-4 output term — transmission (reads a κ4 block) and births (no κ4 block, plus
  the products of lower cumulants in the power-cumulant → cumulant conversion); coefficients, edges and
  κ3 taken from the *driver* (S̄ for "one-way", S for "two-way");
- perturbed chain S: background terms on its own state + ε × the read-out terms (order ≤ 3 output with a
  κ4 block read from P).

Two-way (driver S, ε = 1) is K4c3 exactly (identity of the partition; checked in `tests/test_linear_response.py`).
One-way (driver S̄) is affine in the births: P^{l+1} = T_l(S̄) P^l + β_l(S̄); its only approximation is the
feedback of the read-outs on the probe's coefficients (first order in κ4 times first order in κ4 for the
coefficients; first order for the conversion products — the size of that is what G1 measures).

## Configurations (depth 16; n = 128, 192, 256; seeds 0–2; device MC truth N = 1e7, cached)

| config | meaning |
|---|---|
| lr:bg | background only (= K3c3) |
| lr:twoway | driver S, ε = 1 (= K4c3; implementation check) |
| lr:oneway | driver S̄, everything on |
| lr:oneway:eps=0.5 | read-outs scaled by 0.5 (linearity of the response of S) |
| lr:oneway:ro=k | read-outs only in the last k ReLU steps (k = 1 is the direct path: κ4(z^L_i) = ⟨w_i^{⊗4}, κ4(h^{L-1})⟩) |
| lr:oneway:b=g | births only in the last g + 1 ReLU steps (g generations that can reach the final mean) |
| lr:oneway:ro=1:b=4 | the cheapest add-on: direct path, 4 generations |
| lr:oneway:bf=A | births only in the first A ReLU steps (the consistent version of round 3's gen:A) |
| lr:oneway:kind=coh / kind=rs | read-outs only through paired / only through random-sign z-slices |
| lr:oneway:rank | spectra of the random-sign bulk of P and of the births (n ≤ 192) |

Recovery = 1 − log(cfg / K4c3) / log(K3c3 / K4c3), paired over seeds (1 = K4c3, 0 = K3c3).

## Rules (each can fail)

- **P0 (implementation)**: lr:twoway / K4c3 = 1.00 ± 0.02 and lr:bg / K3c3 = 1.00 ± 0.02. A failure voids the round.
- **G1 (linear response)**: lr:oneway / K4c3 ≤ 1.3 at n = 192 and 256, and the nonlinearity
  ‖μ(1) − 2μ(½) + μ̄‖² / ‖μ(1) − μ̄‖² ≤ 0.1 (median over seeds). Falsified → the adjoint add-on is dead.
  Prediction: ratio 1.0–1.15, nonlinearity ≤ 0.05.
- **G2 (cheap add-on)**: lr:oneway:ro=1:b=4 recovers ≥ 0.5 of the log-gap at n = 256 → a direct-path,
  4-generation add-on is viable. No point prediction: round 3's rs_3 removal (4504×) suggests large
  cancellations between read-out paths, in which case ro=1 can even be worse than K3c3 (recovery < 0).
  Secondary: the read-out depth k and generation depth g needed for recovery ≥ 0.7, and whether they
  grow from n = 192 to 256 (growth → the cost at 1024 is higher than at 256).
- **G4 (stability)**: lr:oneway:bf=4 and bf=8 stay finite (MSE < 10 × K3c3). Prediction: finite, i.e. round 3's
  gen:A divergence came from the inconsistent filter (it dropped the newborn power cumulants but kept the
  products of lower cumulants that cancel them in the conversion), not from an expanding transport.
  Falsified → the transport is expanding and the births regulate it (round 3's reading stands).
- **G3 (rank, exploratory)**: energy fraction of the random-sign bulk of P (index-distinct entries,
  (ab)(cd) unfolding) in its top 64 singular directions at n = 128 and 192. A compressed-memory chain
  is viable only if the rank for 90 % energy grows by less than 1.3× from 128 to 192.
