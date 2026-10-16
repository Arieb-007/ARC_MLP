# FINDINGS — what we know, with status labels

> Written by an LLM agent (Claude Code) directed by Areeb. Labels: **[proven]** = derivation
> given here (or a cited theorem); **[measured]** = produced by a script in this repo, output file
> in `results/` (seed, versions, commit recorded there); **[hypothesis]** = interpretation or
> extrapolation not yet tested; **[external]** = reported by another team, not re-measured by us.
> Numbers in F4–F6 marked *(prompt)* come from Areeb's earlier runs as stated in the task brief;
> they become **[measured]** once our re-run in `results/` reproduces them (see the
> "reproduced?" column).

Notation: `Φ, φ` standard normal CDF/PDF; `X ~ N(m, s²)`, `t = m/s`,
`psi(m, s) = m Φ(t) + s φ(t)`. Layers `z_l = W_l h_{l-1}`, `h_l = ReLU(z_l)`, `W ~ N(0, 2/n)`.
"Unit" = one dense n×n×n matmul = 2n³ FLOPs; B = 1024 units at n = 1024.

---

## F1 — Closed forms [proven; numerically checked in `arcmlp/closure/relu_moments.py`]

For `X ~ N(m, s²)`:

| quantity | closed form |
|---|---|
| `E[ReLU(X)]` | `psi = m Φ(t) + s φ(t)` |
| `E[ReLU(X) X]` (= `E[ReLU(X)²]`) | `(m² + s²) Φ + m s φ` |
| `E[ReLU(X) X²]` | `(m³ + 3 m s²) Φ + s (m² + 2 s²) φ` |
| `∂_m psi, ∂²_m psi, ∂³_m psi, ∂⁴_m psi` | `Φ, φ/s, −tφ/s², (t² − 1)φ/s³`; in general `∂^k_m psi = (−1)^k He_{k−2}(t) φ(t)/s^{k−1}` for k ≥ 2 |

*Derivation.* Write `X = m + sZ`. `E[X^j 1{X>0}] = s^j Σ_i C(j,i) t^{j−i} I_i(t)`, with
`I_i(t) = ∫_{−t}^∞ z^i φ(z) dz`, `I_0 = Φ(t)`, `I_1 = φ(t)`, and by parts
`I_i = (−t)^{i−1} φ(t) + (i−1) I_{i−2}`. Substituting `t = m/s` gives the table (the
`m² s φ` terms cancel in the third row: `3m²s − 3m s² t + s³ t² = m² s`). Derivatives: `∂_m psi =
Φ + tφ − tφ = Φ`, then `φ' = −tφ`. Numerical check (quadrature at m=0.3, s=1.7; finite differences):
agreement to ≤ 1e-9 relative for the moments and ≤ 1e-7 for ∂¹–∂³ (∂⁴ limited by FD roundoff).

**Price's theorem** [proven, Price 1958]: for jointly Gaussian `(X_i, X_j)` with covariance `c`,
`∂_c E[f(X_i) g(X_j)] = E[f'(X_i) g'(X_j)]`; hence the leading bivariate ReLU covariance is
`Cov(ReLU X_i, ReLU X_j) = Φ(t_i)Φ(t_j) c + (φ(t_i)φ(t_j)/(2 s_i s_j)) c² + O(c³)`
(the c² term is `E[f'' ]E[g'']·c²/2` with `E[ReLU''] = φ/s`).

**Edgeworth / cumulant expansion** [proven as an asymptotic expansion]: for `Z` with mean m,
variance s², third and fourth cumulants κ3, κ4,
`E[ReLU(Z)] ≈ psi + κ3/6·∂³_m psi + κ4/24·∂⁴_m psi + κ3²/72·∂⁶_m psi + …
= psi − κ3 tφ/(6s²) + κ4 (t² − 1)φ/(24 s³) + …`
(from `E[f(Z)] = exp(Σ_{r≥3} κ_r ∂^r_m / r!) E_G[f]` with `E_G[f^{(k)}] = ∂^k_m E_G[f]`).

**Cumulants of ReLU(sZ) at t = 0** [proven, closed form; `relu_cumulants_std()`]:
κ1 = 0.398942 s, κ2 = 0.340845 s², κ3 = 0.326458 s³, **κ4 = 0.279708 s⁴** (the brief's
"0.278 s⁴" is this number rounded down).

## F2 — Cost atom [proven]

With `T_m = diag(Φ(t_m)) W_m` (the Jacobian of the closure's mean map), every K ≥ 3 method needs

`d_i^{(s,l)} = Σ_m (Q^{(s→l)})_{im}³ a_m^{(s)}`,   `Q^{(s→l)} = T_l ⋯ T_{s+1}`,   `a_m^{(s)} = κ3(h_{s,m})`

(the diagonal of the third cumulant at layer l contributed by the skew born at layer s, to leading
order in which each source is diagonal at birth). *Why it is costly:* the entrywise cube forces
`Q` to exist explicitly in the neuron basis, so every interval product `Q^{(s→l)}` (s < l) is
needed: with one new source per layer and one transport per live source per layer, the dense cost
is Σ_l (l − 1) ≈ L²/2 = 128 units at L = 16 — above the ≈ 102-unit floor. The (2,1) slice
`Q^{⊙2} diag(a) Qᵀ` has rank ≤ rank(Q), but `Q^{⊙2}` (Hadamard square) has rank up to
rank(Q)² ("Hadamard wall": `rank(A ⊙ B) ≤ rank(A)·rank(B)`). A rank-r Tucker treatment of
the cube costs `O(n r³)` per source-layer for the contraction plus `O(n² r)` for transports, which
beats one dense unit (`2n³`) only for `r ≲ n^{2/3}` ≈ 100 at n = 1024.

504aldo's V16b/F63 [external, parity-checked by them] brings the per-source-layer cost of the
exact K3-simple (2,1) update to 4 units (2 transports + 2 contractions), and V29 brings the whole
estimator to ≈ 260 units ≈ 0.254 B with Strassen–Winograd pricing (their F86 ledger: 95% of the
bill is κ3 source machinery).

## F3 — Order hierarchy [proven]; floor claim [hypothesis]

Push a **diagonal** cumulant of `h` (independent-source approximation: `κ_r(h)` supported on
`m_1 = … = m_r`) through `z = W h`, `W_ij` iid `N(0, 2/n)`. Then
`κ_r(z)_{i_1…i_r} = Σ_m W_{i_1 m} ⋯ W_{i_r m} c_m`, `c_m = κ_r(h_m)`. Moments of `W`:
`E W² = 2/n`, `E W⁴ = 12/n²`, `E W⁶ = 120/n³`, odd moments 0. Let `a_m = κ3(h_m)`,
`b_m = κ4(h_m) ≥ 0`, `ā = mean_m a_m`, `b̄ = mean_m b_m`, and the layer sum `B := Σ_m b_m = n b̄`.

| contraction | mean over W | fluctuation (std) | character |
|---|---|---|---|
| (3,) `Σ_m W_im³ a_m` | 0 | `√(n·120/n³)·rms(a) = O(n^{-1})` | random-sign |
| (2,1) `Σ_m W_im² W_jm a_m`, i≠j | 0 | `√(n·12/n³)·rms(a) = O(n^{-1})` | random-sign |
| (2,2) `Σ_m W_im² W_jm² b_m`, i≠j | `4 b̄/n` (= `4B/n²`) | `O(n^{-3/2})` | **coherent** |
| (4,) `Σ_m W_im⁴ b_m` | `12 b̄/n` | `O(n^{-3/2})` | **coherent** |
| (3,1) `Σ_m W_im³ W_jm b_m` | 0 | `O(n^{-3/2})` | random-sign |
| (2,1,1) `Σ_m W_im² W_jm W_km b_m` | 0 | `O(n^{-3/2})` | random-sign |
| (1,1,1,1) | 0 | `O(n^{-3/2})` | random-sign |
| κ5 slices | 0 or coherent `O(n^{-2})` (even-power sub-slices) | `≲ O(n^{-2})` | — |

*Proof sketch.* Each entry is a sum of n independent terms; the mean is `n·E[product]·(mean c)`,
nonzero only when every row index appears an even number of times; the std is
`√n·√E[product²]·rms(c)`. E.g. (3,): `E[W⁶] = 15(2/n)³ = 120/n³`.

*Notation correction.* The brief writes "`Σ_m W_im² W_jm² b_m ≈ 4b̄/n`" and later "the analytic
`b̄_l = Σ_m κ4(h_{l,m})`". These are consistent only if `b̄` is the **mean** in the first and we
call the **sum** `B_l`; the coherent value is `4 b̄/n = 4 B/n²`. With `b_m ≈ 0.28 s_m⁴` and
`s_m² ≈ O(1)` under He scaling, the coherent (2,2) entry is `≈ 1.1 s⁴/n`.

**F3.1 Consequence for the closure** [proven at leading order]. Odd orders (κ3 slices, κ4 (3,1),
(2,1,1)) enter the next layer as random-sign `O(n^{-1})` / `O(n^{-3/2})` matrices that depend on
the specific W ("network-specific, diffuse"). Even orders have a coherent leading part that is
**one scalar per layer** (`b̄_l`), plus random-sign remainders. Through Price/Edgeworth:
the coherent (4,) part shifts each mean by `12 b̄/n · (t²−1)φ/(24 s³)` and the coherent (2,2)
part adds a rank-one term `(4 b̄/n)·¼ v vᵀ`, `v_i = φ(t_i)/s_i`, to the post-ReLU covariance —
both `O(1/n)` per entry and both computable at `O(n²)` cost.

**F3.2 Relation to V29's `λ_l` — correction to the brief** [proven for the structural statement;
hypothesis for the numerical identification]. The brief states that "V29's single fitted `λ_l`"
captures the coherent κ4 term and that "the principled replacement is the analytic `b̄_l` with the
rank-one term inserted exactly". This conflates two objects:

- 504aldo's own correction F105 (2026-09-22, after an outside replication by AndreasHad04 on
  forum topic 18218) states that V29's `G` is the **r = 1 "harmonic-projection" matrix core** of
  the augmented chain, `G_off = c_A (K211_cap + hub source contributions) + c_S Sym(K31)`, with
  `c_A = 6/(n+4)`; the (2,2) slice enters G only through its row sums on the diagonal. The
  replication found the off-diagonal (2,2) slice of post-activation κ4 is covariance-shaped only
  at layer 1 (R² 0.998 → 0.045 at L15) [external].
- The harmonic projection of a symmetric 4-tensor onto `Sym(δ ⊗ G)` is built from the
  **partial trace** `Σ_k κ_{ijkk}`. For a diagonal source pushed through W,
  `Σ_k κ4(z)_{ijkk} = Σ_m W_im W_jm b_m (Σ_k W_km²)`, and `Σ_k W_km² = 2 + O(n^{-1/2})` is a
  **coherent** sum [proven]. So `G_off ≈ 2 c_A Σ_m W_im W_jm b_m`: a random-sign `O(n^{-1/2})`
  matrix with the same sparsity pattern as the diagonal-source part of
  `C_pre,off = Σ_m W_im W_jm var(h_m) + …`. If `b_m / var(h_m) ≈ ρ_l` across neurons, then
  `G_off ≈ λ_l C_pre,off` with `λ_l ≈ 2 c_A ρ_l` — **one scalar per layer, and it is a
  coherent-sum constant**, which is the mechanism F3 predicts, but acting through the partial
  trace, not through a rank-one (2,2) insertion. [hypothesis until Stage 1 measures it]
- Consistent with this, V25's adaptive rule sets `λ_l ∝ mean(dG)/mean(var)` (a ratio of a κ4
  diagonal to a variance), and from V29's own tables `LAM[l]/REF_R[l]` = 0.76 (L0), 0.99, 1.15,
  1.26, 1.34, 1.40, 1.44, 1.45, 1.47, 1.48, 1.50, 1.51, 1.51, 1.52, 1.52 (L14) — the fitted
  slope settles at ≈ 1.5 with depth [measured from the constants in `third_party/v29`; the
  interpretation is a hypothesis]. Explaining that factor (e.g. by the residual off-diagonal
  covariance of `h`, or by the var-weighting of the least-squares fit when `b_m/var_m` varies
  with `t_m`) is the concrete Stage-1 target.

**F3.3 Floor claim** [hypothesis]. The leaders' RMS error `√(1.1e-8) ≈ 1.05e-4` is close to
`4 n^{-3/2} = 1.2e-4` at n = 1024; hypothesis: the field sits at the floor of an `O(1/n)`-complete
theory, and the next 10× needs the random-sign κ4 slices, which ride on the same legs `Q` as κ3
(extra cost only in contractions: (4,) is O(n²), (3,1) and (2,2) one matmul each; (2,1,1) spawns
mixed sources needing their own leg transports). Not tested; Stage 2.2 is the test.

## F4 — Lyapunov rank law [prompt numbers; re-run: `results/*_e1_predictability.*`]

Heuristic [hypothesis, from Newman 1986-type Gaussian-product spectra]: the interval product of
`age` iid Gaussian-like maps has singular values `σ_k ≈ (k/n)^{-age/2}`-like decay, giving the
rank at relative Frobenius error ε ≈ `n(1 − ε^{2/age})`. Measured on the closure's transfer
maps `T = diag(Φ(t)) W` at n = 1024, L = 16 (`experiments/predictability.py`, seed 0):

| age | rank @1% | @10% | @30% | heuristic @1% / 10% / 30% |
|---|---|---|---|---|
| 1 | 656 | 507 | 351 | see results file |
| 2 | 563 | 388 | 236 | |
| 4 | 437 | 264 | 145 | |
| 8 | 300 | 160 | 80 | |
| 15 | 199 | 98 | 47 | |

(*prompt*; reproduced? → see `results/STATUS.md`.) Young sources need 1–2% accuracy (504aldo:
extra MSE ≈ 4.2e-6·ε² on D21 [external]) → rank 400–650 → dense is cheaper; old sources tolerate
~30% → rank 50–80 → marginal. This is consistent with 504aldo's "cliff" (rank 3n/8 at age 4,
7n/32 at age 7) [interpretation: hypothesis].

## F5 — Where the error lives [prompt numbers; re-run: `results/*_e2_conditional.*`, `*_e3_richardson.*`]

`experiments/conditional.py`, seed 3, n = 256, L = 8, 1.6M-sample truth, MC floor ≈ 1.2e-7:

| estimator | final-layer MSE |
|---|---|
| K=2 closure, propagated | 3.56e-5 |
| K=2 at last layer with oracle `μ_{L−1}, C_{L−1}` | 4.69e-6 (→ 87% of the error is accumulated drift) |
| oracle-conditional on top-k modes of `h_{L−1}`, Gaussian bath, k = 1/4/16/64/128 | 4.24e-6 / 3.75e-6 / 2.16e-6 / 1.05e-6 / 7.5e-7 (last ≈ MC floor of the 200k oracle samples) |

62% of the squared error lies in the top eigenmode of Cov(h_L), which carries 10% of the variance.
`experiments/richardson.py` (seeds 3, 4, 5): cos(top eigenvector, μ_L) = 0.95 / 0.87 / 0.94;
cos(error, μ_L) = 0.84 / 0.76 / 0.80 — the drift is radial.
Interpretation [hypothesis]: the Jacobian chain projects every per-layer error onto its top
Lyapunov direction; per-layer scalar calibration exploits this; the network-specific remainder is
caused by diffuse random-sign κ3 contractions.

## F6 — Dead ends (do not retry without a new idea)

| idea | evidence | status |
|---|---|---|
| any sampling component | output variance ≈ 0.08/neuron; 6,500 passes at 0.1B → raw 1.2e-5; 1e-8 needs ≈ 7e6 passes = 100 budgets; Rao–Blackwellising on slow modes is worse; randomized sketches of the cube need S ≳ n/ε² | prompt; arithmetic [proven] given the variance |
| Richardson extrapolation across truncation orders | corr(K2 error, K2−K1) = +0.03 / −0.21 / +0.03 (seeds 3, 4, 5) | prompt → re-run E3 |
| spherical designs / cubature at the input | O(n^{-1/2}) per-neuron 1-D error; Kerdock–MUB negative result | prompt |
| conditioning on a few slow modes as a replacement for dense κ3 | E2: k=16 modes only halves the oracle K=2 error | prompt → re-run E2 |
| Gaussian scale mixtures | 1.7× | prompt |
| exact bivariate covariance alone | 7% | prompt |

504aldo's 25 dead ends (their `docs/community_post.md` §5, `docs/ablation_table.md`) [external]
are the larger catalogue; we do not repeat them here.

## F7 — Exact identities recorded, no saving found [proven]

Without biases `h_L` is positively 1-homogeneous in `x`, so Euler gives `h_L(x) = x·∇h_L(x)`
(a.e.), and Stein's lemma `E[x·∇u] = E[Δu]` for `x ~ N(0, I)` gives
`E[h_L] = E[Δh_L] = Σ_{l,k} p_{l,k}(0)·E[[A_l]_{ik} |∇z_{l,k}|² | z_{l,k} = 0]`,
`A_l = D_L W_L ⋯ D_{l+1} W_{l+1}` (the distributional Laplacian of ReLU at the kinks; `p_{l,k}(0)`
the density of `z_{l,k}` at 0). Stein control variates `E[Δu − x·∇u] = 0` remove the
closed-form-mean requirement, but sampling is dead anyway (F6).

---

## Stage-0 facts

See `results/stage0_submission.md` and `results/STATUS.md`.
