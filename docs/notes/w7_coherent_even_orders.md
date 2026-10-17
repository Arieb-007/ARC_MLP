# W7 — Coherent even-order cumulant terms: derivation

> Written by an LLM agent (Claude Code) for team Arieb-007. Status labels: **[proven]** (derived
> here from the diagram rule, which is itself unit-tested), **[measured]** (file in `results/`),
> **[hypothesis]**. Code: `arcmlp/estimators/coherent_even.py`; reference
> `arcmlp/reference/dense_cumulants.py`.

## 0. The diagram rule (transcribed, validated)

Source: ARC's reference implementation of the companion paper (MIT, commit `93d091a`,
`kprop_ds.nonlin_kprop`, `kprop_harmonic.get_all_terms/get_vec_cond`, `partitions.vec_part_coef`).
The paper PDF was not reachable from the sandbox; the code is the transcription source.

For pre-activations z with mean m, variances s² (reference Gaussian: independent N(m_i, s_i²)) and
deviation cumulants (off-diagonal covariance, κ3, κ4, …), the power cumulant of ReLU outputs on r
distinct neurons with powers p = (p_1..p_r) is

  pK_p = Σ_ν coef(ν) · Π_{v∈ν} κ_v · Π_a w(k_a, p_a),   coef(ν) = 1 / Π_{distinct v} (c(v)! Π_i v_i!^{c(v)}),

summed over connected multisets ν of blocks v (vectors of leg multiplicities over the r neurons;
|v| = 2 only between distinct neurons), with k_a the number of legs at neuron a and
w(k, p) = E[∂^k ReLU(z)^p] (closed forms in `closure/relu_moments.py`). Ordinary cumulants follow
by the power-cumulant → cumulant conversion. Truncation: Σ_v cost(v) ≤ cap, cost(v) = |v| − 1 −
[all v_i even], |κ_v| = O(n^{−cost/2}).

Validation [measured, `tests/test_dense.py`, `results/*_dense_reference_validation.md`]:
covariance-only diagrams reproduce the exact bivariate ReLU covariance (Mehler series) with error
→ 1e-18 as the cap grows; the single-neuron κ3/κ4 coefficients (1/6, 1/24, 1/72, 1/144) give the
predicted c³ error law on a Gaussian-polynomial test; Kt = 2 equals the starter-kit closure to 1e-15.

## 1. Pre-activation κ4 after a Gaussian linear layer — coherent part [proven]

Let h have per-neuron fourth cumulants b_m = κ4(h_m) and (2,2) slice B_mp = κ(h_m,h_m,h_p,h_p)
(m ≠ p). For z = W h, W_ij iid N(0, 2/n), E W² = 2/n, E W⁴ = 12/n²:

| z-slice | dominant source sum | E_W (coherent part) | fluctuation |
|---|---|---|---|
| (4,): κ4(z)_aaaa | Σ_m W_am⁴ b_m + 3 Σ_{m≠p} W_am² W_ap² B_mp | 12 b̄/n + 12 B̄ | O(n^{−3/2}) |
| (2,2): κ4(z)_aabb | Σ_m W_am² W_bm² b_m + Σ_{m≠p} W_am² W_bp² B_mp | 4 b̄/n + 4 B̄ | O(n^{−3/2}) |
| (3,1), (2,1,1), (1,1,1,1) | (no paired structure) | 0 | O(n^{−3/2}) random-sign |

with b̄ = n^{−1} Σ_m b_m and B̄ = n^{−2} Σ_{m≠p} B_mp (B_mp = O(1/n), so B̄ = O(1/n)).
Hence the coherent part is **isotropic**: κ4(z)_abcd ≈ β (δ_ab δ_cd + δ_ac δ_bd + δ_ad δ_bc),
β = 4b̄/n + 4B̄ — the cumulant of a Gaussian scale mixture, i.e. a random global factor on all
pre-activation variances. One number per layer. (Derivation: each entry is a sum of
independent terms; its mean is non-zero only when every row index appears an even number of
times; the (4,) entry counts 3 pairings.)

**Sign structure [measured, `results/2026-10-17_dense_reference_validation.md`].** The brief's
claim "κ4(ReLU(X)) ≥ 0, which makes even orders coherent" is false: κ4(ReLU(t + sZ))/s⁴ ≥ 0 iff
t ≤ 0.7556 (min −0.286 at t = 1.58); κ6 changes sign at t = −0.316, 0.961, 2.329. Coherence comes
from W_am² ≥ 0, not from b_m ≥ 0; b̄ is a mixed-sign average and can be small.

## 2. Effect of the (4,) and (2,2) slices on the next layer [proven]

Single-block diagrams (cost 2, i.e. O(n^{−1})). Write w_k ≡ w(k, 1), so w_2 = φ/s,
w_3 = −tφ/s², w_4 = (t² − 1)φ/s³, and w(k, p) = p!·w_{k−p+1} for k ≥ p.

(a) **Mean** (pK(1,)): Δμ_a = κ4(z)_aaaa w_4(a)/24. Coherent part: (3β/24) w_4 = (β/8)(t²−1)φ/s³.
    Order n^{−1}, combinatorial constant 1/4! = 1/24. Cost O(n).

(b) **Covariance.** Off-diagonal: Δκ(h_a,h_b) = κ4(z)_aabb w_2(a) w_2(b)/(2!·2!) → coherent
    rank-one (β/4) u uᵀ, u_a = φ_a/s_a. Variance: ΔpK(2,)_a = κ4_aaaa w(4,2)/24 = κ4_aaaa·2w_3/24,
    coherent −(β/4) t φ/s², and Δκ2 = ΔpK(2,) − 2μ Δμ. Order n^{−1}. Cost O(n²) if the rank-one
    term is materialised, O(n) if kept factored.

(c) **Diagonal κ3.** ΔpK(3,)_a = κ4_aaaa w(4,3)/24 = κ4_aaaa·6 w_2/24 (coherent (3/4)β φ/s), and
    Δκ3 = ΔpK3 − 3(ΔpK2·μ + pK2·Δμ) + 6μ²Δμ. Also the (2,1) slice gets
    ΔpK(2,1)_ab = κ4_aabb w(2,2)_a w_2(b)/4 = κ4_aabb·2Φ_a w_2(b)/4 (rank-one when coherent).
    Order n^{−1}. Cost O(n) / O(n²).

At cap 3 the same blocks also enter with one covariance edge (cost 3, O(n^{−3/2})).

## 3. Order n^{−2} and n^{−3}: κ6, κ8 at the level of the mean [proven; mean only]

Coherent κ6 of z from a diagonal source c6_m = κ6(h_m): κ6(z)_{a…} ≈ γ6 Σ_{15 pairings} δδδ with
γ6 = 8 c̄6/n² (E W⁶ = 120/n³: (6,) = 15γ6, (4,2) = 3γ6, (2,2,2) = γ6). Mean effect
Δμ = κ6_aaaaaa w_6/6! = (15γ6/720) w_6 = (γ6/48) w_6, O(n^{−2}).
At the same order the product of two coherent (4,) blocks contributes (κ4_aaaa)² w_8/(2!·4!²)
(coefficient 1/1152 for the 9β² coherent value: β² w_8/128), and the square of the random-sign
κ3 diagonal contributes (κ3_aaa)² w_6/(2!·3!²) = κ3² w_6/72 — a square, hence coherent.
κ8: γ8 = 16 c̄8/n³ (E W⁸ = 1680/n⁴), (8,) = 105 γ8, Δμ = 105 γ8 w_8/8!, O(n^{−3}).
In `coherent_even.py` c6, c8 are taken from the Gaussian-ReLU closed form of the previous layer
(newborn source only; transported κ6/κ8 memory is not included) [approximation].

## 4. What this predicts [hypothesis, tested in W7]

If the field sits at an O(1/n)-complete floor, adding the coherent κ4 (O(n^{−1}) per neuron, O(n^{−2})
in MSE) to K=3 should close most of the gap to the dense K=4 at the same cap, and κ6/κ8 should be
below the MC floor at n ≥ 128. The random-sign slices (3,1), (2,1,1), (1,1,1,1) are O(n^{−3/2}).
