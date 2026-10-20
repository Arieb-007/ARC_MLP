# Round 5 — the random-sign κ4 memory is a rank-n CP object, and why matrix proxies plateau (2026-10-20)

> Written by an LLM agent (Claude Code) for team Arieb-007, at Areeb's request to find mathematics that
> changes the estimator rather than the write-up. Status labels: **proven** (derivation below, checked
> against the term registry), **measured** (results/ files named), **hypothesis**. Literature: I know no
> reference that states either result for cumulant propagation; ARC's reference implementation carries
> dense tensors truncated by diagram cost, 504aldo's V29 uses a λ·C ansatz. "No authoritative reference known."

Setting: x ~ N(0, I_n), z^l = W^l h^{l−1}, h^l = ReLU(z^l), W^l fixed draws with i.i.d. N(0, 2/n) entries.
Slices of a symmetric 4-tensor K: b_j = K_jjjj, B_jk = K_jjkk, S_jk = K_jjjk, A_{j,kl} = K_jjkl (the (2,1,1)
slice, viewed as an n × n² matrix "(j | kl)"), K° = index-distinct entries ("bulk", the (1,1,1,1) slice).

## Theorem A (structural; proven at the level of the cap-3 term registry)

Under the cap-3 ReLU closure with the bulk K° set to zero, the state
  (μ, C, κ3 in V29's factored form, b, B, S, A)
is closed under the layer map, where A is carried as a sum of CP terms of rank n,
  A_{j,kl} = Σ_g Σ_{r=1}^{n} U^g_{jr} V^g_{kr} W^g_{lr}        (one term g per birth family and generation),
and every operation on it costs O(n³) per term:

(i) **Transport is exactly CP-preserving.** κ4(z)_{aa kl} = Σ_j W_aj² A_{j,kl} + [S, B, b patterns] + [bulk, dropped],
    so U^g ← (W∘W) U^g, V^g ← W V^g, W^g ← W W^g: three matrix products (3 units of 2n³) per term.
    The S, B, b patterns are exact O(n³) contractions of n × n matrices (e.g. Σ_j W_aj³ W_bm S_jm = [(W^{∘3}) S Wᵀ]_ab).
(ii) **ReLU vertex factors are diagonal scalings** of U, V, W (by w(k,p)(j), (k), (l)): free.
(iii) **Births are CP of rank ≤ n per family** when written with the doubled vertex as the CP index:
    two edges C_jk C_jl → (U, V, W) = (I, C, C); κ3 block + edge D21_jk C_jl → (I, D21, C) with D21_jk = κ3(z)_jjk;
    paired κ4 + edge B_jk C_jl → (I, B, C); the triangle C_jk C_jl C_kl → rank n² (Hadamard with C): keep only
    if measured necessary. The κ3 (1,1,1)-block + edge births have rank n·r3 (r3 = κ3's factored rank).
(iv) **Read-outs are O(n³) per term**: into κ3(h): δκ3_{jkl} = w2_j w1_k w1_l A_{j,kl} + perms (CP, handed to
    V29's factored κ3); into the paired slices: Σ_j W_aj² W_bk W_bl A_{j,kl} = [((W∘W)U)((WV)∘(WW))ᵀ]_ab (2 units).

Proof: inspection of the registry (`arcmlp/reference/dense_cumulants.py`, `enumerate_terms`): every cap-3 term
with a (2,1,1) or (3,1) block has exactly one doubled vertex, and the linear step contracts each index
with one row of W, so the doubled index carries W∘W and the single indices carry W. ∎

Cost of dropping the bulk: measured 1.80–1.89× in MSE at n = 128/160/192, L = 16, flat in n
(`K4c3-no1111 / K4c3`, results/2026-10-18_gpu_round2_as_received.md). [hypothesis] it stays ≈ 1.8× at 1024.

Merging generations: the (j | kl) unfolding of a CP term is U (V ⊙ W)ᵀ (Khatri–Rao product V ⊙ W, an
n² × n matrix), and (V ⊙ W)ᵀ(V ⊙ W) = (VᵀV) ∘ (WᵀW) (n × n, 2 units); with that Gram matrix the exact
rank-r truncation of the concatenated generations costs O(n³) (an n × n eigenproblem, no n² × n² object).
So the cost knob is r = kept rank of A's (j | kl) unfolding, as V29's r_old is for κ3.

## Proposition B (statistical; derived here — why λ·C and every matrix-valued proxy plateau)

Write the (2,1,1) transport as κ4(z)_{aa kl} = Σ_j W_aj² A_{j,kl}. Split W_aj² = 2/n + ε_aj with E ε = 0,
Var ε_aj = 2 (2/n)² (Gaussian fourth moment). Then
  κ4(z)_{aa kl} = (2/n) P_kl + Σ_j ε_aj A_{j,kl},     P_kl = Σ_j A_{j,kl} (partial trace).
Split A into its j-coherent part (energy fraction f: the best rank-1 (j | kl) approximation) and the rest.
For the coherent part the fluctuation term is O(n^{-1/2}) relative (Hanson–Wright). For the rest the
summands A_{j,kl} have random signs across j, so the mean-weight sum is (2/n)√n·a and the fluctuation sum is
√2·(2/n)√n·a: **the same order, ratio √2**. Hence any closure that represents the (2,1,1) channel by one
n × n matrix times per-neuron scalars (λ_a C_kl, λ_a·anything, a dynamical partial trace) has
  R²  ≤  f + (1 − f)/3        (equality when the incoherent part is independent of W).
Measured [results/2026-10-18_gpu_round2_as_received.md]: the per-neuron λ_a oracle on the (2,1,1) z-slice
gives R² = 0.43–0.44 at n = 128–192, flat → f ≈ 0.15 if the matrix is well aligned. Prediction: the best
rank-1 (j | kl) energy fraction f of A is ≈ 0.15–0.35 and does not grow with n; no O(n²)-state closure
will pass R² ≈ 0.5. The rank-n CP form has R² = 1 by construction.

Corollary (shrinkage): when a channel is represented only partially (proxy p = ρ t + noise for the true
term t), the MSE-optimal coefficient is β = Cov(t,p)/Var(p) < 1 (Wiener). This explains two measured facts:
V29's λ is 3× below the physical value (Stage 1, FINDINGS F8) and round 4's read-out strength optimum
ε = 0.5 → 0.75 → 1 for n = 96 → 128 → 192 as the cap-3 representation becomes more complete. In a chain
with truncated rank r, each channel should carry a fitted β_r (two scalars per layer, fitted on the public
split — a legitimate learned closure).

## What decides it (dense-chain oracles, slice hooks; a few GPU minutes per width)

For n = 192 and 256, L = 16, 3 seeds, device MC truth:
1. `K4c3-no1111` at 256 (bulk dropped): prediction 1.8 [1.5, 2.2] × K4c3.
2. `a211:r` (bulk dropped, A projected to its top-r (j | kl) directions after every linear step),
   r ∈ {1, 4, 16, 64, 256}: the MSE-vs-r curve. **Kill rule**: if the r giving ≤ 1.3 × `K4c3-no1111` grows
   proportionally with n from 192 to 256, the CP chain costs ≥ 0.5 B at 1024 and cannot top the board.
   Prediction [hypothesis]: r ≈ 64 suffices at both widths (the (j | kl) spectrum is dominated by the
   generation-coherent directions).
3. Spectrum of A's (j | kl) unfolding per layer: f (top-1 fraction), k50, k90, and the fraction along the
   all-ones j-direction (the partial-trace channel) — the direct test of Proposition B.

Budget at n = 1024 (1 unit = 2n³; B = 1024 units; V29 = 251): per kept CP term ≈ 5 units/layer; at rank r the
transport is 3r/n units → with r = 64 and 4 families ≈ 2–3 units/layer ≈ +40 units ≈ +0.04 B. If the oracle
confirms r = 64: V30 raw ≈ 1.8 × K4c3(1024) ≈ 1.5e-9 … 1.2e-8 [hypothesis, from the n^-2.5 … n^-1 ratio laws]
at C/B ≈ 0.29 → score 0.4 … 3.4e-9 against the leader's 1.42e-9. Phase 2 closes 2026-10-21 23:59 UTC; this
is for the next phase.
