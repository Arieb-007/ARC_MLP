# Standing task brief — ARC White-Box Estimation Challenge 2026 (team Arieb-007)

> Written by an LLM agent (Claude Code) from Areeb's task prompt of 2026-10-16. Sections 1–4 and
> 6–8 of that prompt, reorganised. Status labels: **proven** / **measured** / **hypothesis**.
> Where we found a statement of the prompt to be wrong or ambiguous, a boxed *Note* says so; the
> original statement is kept.

## 0. Deadlines (UTC) and priorities

| event | deadline |
|---|---|
| Phase-2 submissions close | **2026-10-17 23:59** |
| Algorithmic-contribution write-ups due | **2026-10-24 23:59** |
| Winners announced | ~2026-11-15 |

Rule: if the submission deadline has not passed, Stage 0 (a graded submission) is the top
priority and everything else waits. We are registered for Phase 2. 10 submissions per team per
UTC day.

## 1. The task (verified facts)

**Model.** Fully connected ReLU MLP, `L = 16` weight matrices `n x n`, `n = 1024`, He init
`W_ij ~ N(0, 2/n)`, input `x ~ N(0, I_n)`, no biases:

```
h_0 = x,  z_l = W_l h_{l-1},  h_l = ReLU(z_l),  l = 1..16
```

The dataset stores only weight matrices (`weights`: `(16, 1024, 1024)` float32 per MLP, per the
starter-kit docs; confirm when a public MLP is loaded — HF is not reachable from the cloud
sandbox, see `results/STATUS.md`).

**Target.** Per-neuron post-ReLU means `mu_{l,i} = E_x[h_{l,i}]`. `predict(mlp, budget)` returns
a `(depth, width)` array (`(16, 1024)` in the suite; read the shape off `mlp`). Only the final
layer is graded, against a Monte-Carlo reference with `N = 1e9` samples.

**Score.** `adjusted = final_layer_mse * max(0.1, C/B)`, `B = 2**41 ≈ 2.2e12` FLOPs per MLP,
`C` = FLOPs counted by `flopscope`. `C > B` → that MLP's prediction is zeroed. Below `0.1 B` the
multiplier is frozen, so the game is "maximum accuracy at or just above 10% of budget".
Unit: one dense `n x n x n` matmul = `2 n^3 = 2^31` FLOPs = 1 unit; `B = 1024` units; the floor
is ≈ 102 units.

**Contract** (from `whest-starterkit/docs/reference/estimator-contract.md`, whestbench 0.16.1):
`class Estimator(BaseEstimator)` with `predict(self, mlp: MLP, budget: int) -> fnp.ndarray`,
optional `setup(self, context: SetupContext)` (off-budget, ≤ 5 s per worker process, runs
5–15 times per submission) and `teardown(self)`. `MLP` has `width`, `depth`, `weights` (list of
`(width, width)` arrays), `seed`. Output is coerced to float32; must be finite.

**Hard rules.** All computation through `flopscope`; bundled numpy/scipy, BLAS/LAPACK, compiled
kernels, ctypes/cffi, threads, subprocesses and any tampering with the flopscope client are
disqualifying. Allowed: the grader's Python, the flopscope client API, `whestbench` contract types,
the pure-Python stdlib, and data files (weights, lookup tables, bases). "Packing independent
values into one machine element to reduce billed work is not permitted" (fair-accounting rule).
Per-MLP wall clock ≤ 120 s on one physical core; residual (non-flopscope) wall time ≤ 0.4 s per
MLP (a cap, crossing it zeroes that MLP); process memory ≤ 8 GB; `setup()` ≤ 5 s (exceeding it
fails the whole submission); bundle ≤ 50 MiB and ≤ 50 files. The grader's smoke test uses shapes
other than 16 x 1024 (e.g. 256x32, 64x16, 1024x3 per 504aldo's log): guard every layer- or
width-indexed constant with a fallback.

**Tooling and data.** `pip install "flopscope>=0.12.1,<0.13" "whestbench>=0.16.0"` (pure-Python
wheels; numpy>=2.0,<2.5, opt-einsum). Dataset `aicrowd/arc-whestbench-public-2026`, revision
`v2-phase2`, weights float32 (≈ 67 MB per MLP; mini split = 100 MLPs ≈ 7.0 GB; full = 1000,
≈ 70 GB, do not download). CLI: `whest init | validate | run | smoke-test | package | submit |
login | dataset {download, bake, ...}`. `whest dataset bake` makes our own datasets at any
width/depth with MC ground truth. Submission: `whest login` (AIcrowd API key) then
`whest submit --estimator <file-or-folder>`, or `whest submit submission.tar.gz`.

**Prizes.** Score-based 50k/20k/10k USD. Algorithmic contribution 20k USD: exactly one *graded*
submission ID plus a PDF write-up, judged at ARC's discretion on which novel *mechanistic* ideas
most improved performance over known methods; negative results and ablations welcome; LLM use
encouraged but must be disclosed; unhedged dubious claims make judges distrust the whole write-up.

## 2. Public state of the art (as of 2026-10-15; participants' numbers are self-reported)

| Method | raw final-layer MSE | C/B | adjusted |
|---|---|---|---|
| Kit covariance propagation (K=2 Gaussian closure) | 3.9e-6 | 0.02 (billed 0.1) | 3.9e-7 |
| K=2 + exact bivariate ReLU covariance + per-layer ridge calibration (subm. 329951) | ~1.3e-6 | ≤ 0.1 | 1.3e-7 |
| K=2 + rank-4608 inherited-κ3 channel, zero fitted constants (subm. 330001) | ~1.1e-6 | 0.099 | 1.1e-7 |
| Pure sampling (Sobol', antithetic, control variates all tried, all fail) | — | — | ~1.2e-6 |
| **504aldo "V29"** (MIT, public): factorized K=3 (paper §4.3), memoryless κ4 `G_off = λ_l·C_off`, rank-16 D21 feedback, shared low-rank bases for old sources, Strassen–Winograd | 2.13e-8 | 0.25 | 5.4e-9 (5.2e-9 with `STRASSEN_MIN=8`) |
| Leaderboard top 15 (methods not public) | 1.15–2.0e-8 | 0.10–0.16 | 1.4–2.1e-9 |

Independent grader record for V29 verbatim (Hisernberg/arc-whitebox archive, #333467):
raw 2.1327e-8, multiplier 0.2542, adjusted 5.418e-9, residual 0.183 s mean / 0.195 s max,
100/100 MLPs. Hisernberg's later V32+GRU-corrector variants reach ≈ 3.96e-9 adjusted
(their README; not verified by us).

> *Note (fair accounting).* 504aldo asked ARC on 2026-09-05 whether Strassen–Winograd written as
> flopscope ops is within the fair-accounting rule and had no reply as of their post. Many graded
> submissions use it, but a ruling against it would move V29 back to C/B ≈ 0.37 (their V25).

Sources: companion paper arXiv:2605.05179; forum posts 18197 (Phase 2), 18041 (prize
guidelines), 18218 (504aldo write-up; full text mirrored in
github.com/504aldo/whest-p2-cumulant-k3 `docs/community_post.md`); see `docs/REFERENCES.md`.

## 3. What we already know

See `docs/FINDINGS.md` (F1–F7, expanded with derivations, status labels and the measured tables).

## 4. The plan

### Stage 0 — secure a graded Phase-2 submission (only if before 2026-10-17 23:59 UTC)

1. Vendor 504aldo's public V29 (MIT) into `third_party/v29/` with its LICENSE.
2. Install packages; read the `whest init` template / contract docs; `whest validate` and smoke
   test V29 unchanged on a 2–4-MLP local shard.
3. Set `STRASSEN_MIN = 8` (reported 5.42e-9 → 5.2e-9, grader-accepted by another participant),
   re-validate, confirm residual < 0.4 s and setup < 5 s under `threadpool_limits(1)`.
4. `whest package`; check ≤ 50 MiB / ≤ 50 files. Stop and tell Areeb the exact submit command
   unless credentials are configured.
5. `results/stage0_submission.md` (submission ID, commit, local raw MSE, C/B, residual seconds).

Acceptance: a graded, non-failed submission ID in `results/stage0_submission.md`.

### Stage 1 — algorithmic-contribution write-up (due 2026-10-24 23:59 UTC)

Optional, only before the Stage-0 cutoff with ≥ 12 h margin: replace V29's fitted `λ_l` by an
analytic coherent-κ4 constant (F3), validate on ≥ 20 MLPs with a paired bootstrap; submit only if
not worse. Otherwise the submission is plain V29(+leaf 8) and the write-up documents it plus our
findings.

> *Note.* V29's `G` is not the (2,2) slice of κ4 but the r=1 harmonic-projection core of the
> augmented chain (504aldo F105). The analytic counterpart of `λ_l` is therefore a
> partial-trace (coherent) contraction, not `b̄_l` inserted as a rank-one (2,2) term; see
> FINDINGS F3.2. Stage 1 compares the right object.

Write-up (`docs/writeup/writeup.md` → PDF via pandoc), every table backed by a `results/` file:
(1) coherent/random-sign decomposition of the cumulant hierarchy, why one scalar per layer
captures the leading κ4 effect, numerical comparison of the analytic constant vs V29's fitted
`λ_l`; (2) oracle ablation of the κ4 slices with the dense exact K=4 reference at n=128
(Stage 2.1), which slices move the final error below the `n^{-3/2}` floor; (3) Lyapunov rank law
vs measured ranks; (4) radial-drift decomposition and negative results; (5) reproduction commands,
seeds, CIs, hedged claims, LLM disclosure. Send to arc-whestbench@aicrowd.com or post on the forum
before the deadline — ask Areeb which.

### Stage 2 — the program to reach the top (weeks; in order, each with a falsifying experiment)

| # | Deliverable | Expected effect |
|---|---|---|
| 2.1 | Truth machines: dense exact cumulant propagation K≤4 at n=128 (κ4 tensor ≈ 1 GB float32 at n=128 if stored fully; switches to zero individual slices / cross terms); chunked MC truth (float32 passes, float64 accumulators, up to 1e8 samples) at n=256; `whestbench`-baked small datasets. Verify dense K=3/K=4 against MC at n=64 to MC precision. | enabling |
| 2.2 | Term ablation at small n: inject exact κ4 slices ((4,), (3,1), (2,2), (2,1,1)) and κ3⊗C cross terms into a K=3 pipeline; final error per term with CIs. | decides design |
| 2.3 | κ4 on κ3's legs in the factorized estimator: same `Q`, extra weights `b_m`; (4,), (3,1), (2,2) exact; (2,1,1) only if 2.2 says it matters. | 3–10× MSE if 2.2 favourable (hypothesis) |
| 2.4 | Goal-oriented ranks: reachability `P_{l+1} = T P Tᵀ + B Bᵀ` and observability `O_l = Tᵀ O T + Cᵀ C` Gramian recursions; Hankel values choose dense / low-rank / drop per (source, layer). Transports and covariance corrections only — the cube needs full rows of `Q`. | 1.3–2× cost at fixed MSE (hypothesis) |
| 2.5 | Skewness sparsity: drop columns of `Q diag(a)` with negligible `a_m` before (2,1) products, with an error budget. | ≤ 2× on old sources (hypothesis) |
| 2.6 | Derived closure constants; fit only the residual, cross-validated on self-baked datasets. | 1.1–1.5× (hypothesis) |
| 2.7 | Cost accounting: Strassen–Winograd depth and minimum block vs the 0.4 s residual cap; buffer reuse; batched families. | 1.2–1.5× score (hypothesis) |

Target arithmetic: top ≈ 1.4e-9 adjusted = raw 1.2–1.4e-8 at C/B 0.10–0.13. First milestone:
a clean re-implementation (`arcmlp/estimators/k3_factored.py`) reproducing V29 on 20 MLPs
(raw ≈ 2.1e-8, C/B ≈ 0.25) before any change.

> *Note (2.1 memory).* A dense symmetric κ4 at n=128 has n⁴ = 2.7e8 entries = 1.07 GB in float32
> (2.1 GB float64) if stored unsymmetrised; the symmetric-unique count is ≈ n⁴/24 ≈ 1.1e7.

## 6. Evaluation protocol (binding)

1. Small-n suite: `n ∈ {64,128,256}`, `L ∈ {8,16}`, seeds 0–9; MC truth with N chosen so the MC
   floor is ≥ 10× below the effect measured (state the floor in every table); metrics: final-layer
   MSE, error projected on eigenmodes of Cov(h_L), per-term ablation deltas with bootstrap CIs
   over networks.
2. Official shape: 20-MLP dev subset for iteration, 100-MLP mini split for reporting via
   `whest run`; paired bootstrap over MLPs; report raw MSE, C/B, adjusted, residual seconds, peak
   memory, and the minimum detectable effect (MDE) of every comparison.
3. Cost: utilisation from the FLOP counter; wall time single-threaded; check the 0.4 s residual
   and 5 s setup caps before every upload.
4. Heavy runs belong in the notebook: if the local machine has < 4 cores or < 16 GB, do not run
   official-shape evaluations locally; write `notebooks/*.py` entry points and ask Areeb to run
   them, then read the pushed `results/`. (The 2026-10-16 cloud sandbox: 4 cores, 15 GB, no
   HF/AIcrowd network access → official-shape *timing* on a few self-baked MLPs only.)

## 7. Experiment scripts

`experiments/predictability.py` (E1 → F4), `experiments/conditional.py` (E2 → F5),
`experiments/richardson.py` (E3 → F5/F6) are kept verbatim from the prompt; their outputs are in
`results/`.

## 8. Notebook bootstrap

`scripts/kaggle_bootstrap.sh` clones/pulls this repo in a Kaggle/Colab CPU notebook using a
GitHub deploy key stored as base64 secret `ARC_MLP_DEPLOY_KEY_B64`; `scripts/push_results.sh`
commits `results/` and pushes when the key has write access. See README for the notebook cells.

## 9. How to operate

- Work autonomously through the stages; commit small, push often; name the result file in each
  commit message.
- Stop and ask only for: credentials (GitHub push, AIcrowd submit), a missing V29 link, a spend of
  more than ~6 CPU-hours on one experiment, or any irreversible decision (submitting, posting).
- Before every experiment write one sentence stating what would falsify the idea; record the
  outcome either way.
- Keep the physics explicit: when a term is dropped, state its order in 1/n and whether it is
  coherent or random-sign.
- End every session by updating `results/STATUS.md`.
