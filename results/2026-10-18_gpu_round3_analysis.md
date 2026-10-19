# GPU round 3 (2026-10-18) — scoping the factored-κ4 program: reading of pass 1

> LLM-generated record (Claude Code) for team Arieb-007. Raw report: `results/2026-10-18_gpu_round3_pass1_as_received.md`
> (A40, `scripts/gpu_run.sh --profile round3`, commit 8270746). Device MC truth N = 1e7, depth 16, 3 seeds.

## What worked, what did not

- **n = 256 K=4 rows errored** in ~5 s for every configuration containing the single random-sign
  blocks (the terms that read the full n⁴ pre-activation tensor); configurations without them ran
  (K3c2 7.17e-6; K4c2 = 0.35 × K3c2; fam:−rs 0.70; fam:+k3e+c3 0.62). Diagnosis [inference]: the
  einsum path for the n⁴-operand terms plus pool fragmentation exceeded 46 GB. Fixed: those terms
  now use plain broadcasting on row chunks, the pool is released per term, chunk rows 16 at n = 256
  (verified bit-identical to the previous path at n = 48).
- **gen:A (κ4 born only at layers ≤ A) diverged at every A, even A = 6** (MSE 1e8–1e10 at n = 192), and
  a redefinition that keeps the coherent births diverged at n = 48 too. Finding [measured]: the
  transmitted κ4 alone is unstable; the newborn random-sign κ4 regulates it (the top Lyapunov
  directions of the transport grow, the births cancel them). Generations cannot be ablated by term
  removal; the question "how many generations must be carried" needs a different design (e.g. a
  linear-response tagging of sources) and is left open. Dropped from the profile.

## Diagram families at n = 192, L = 16 (3 seeds; ratio to K4c3, recovery of the K3c2→K4c3 gap)

| config | / K4c3 | recovery | reading |
|---|---|---|---|
| fam:−rs (no single random-sign blocks) | 10.2 | −0.12 | essential |
| fam:−k3e (no κ3 block + edge) | 6.6 | 0.09 | essential |
| fam:−p4e (no paired κ4 + edge) | 1.12 | 0.94 | droppable |
| fam:−c3 (no three-edge chains) | 0.99 | 1.00 | droppable |
| cheap + rs | 4.8 | 0.24 | |
| cheap + rs + k3e | 2.3 | 0.59 | |
| cheap + rs + k3e + c3 (= −p4e) | 1.12 | 0.94 | |
| cheap + k3e + c3 (no rs) | 9.6 | −0.10 | |

Reading [measured, n = 192]: the κ4 gain needs the random-sign blocks **and** the κ3-block + edge
family; the three-edge chains and the paired κ4 + edge terms substitute for each other (either one
suffices, together with rs and k3e, to reach 94–100 % of K4c3). For a factored chain that is good
news: c3 chains are tensor-train objects, while p4e terms are built from the κ4 diagonal/(2,2) memory
and the covariance (O(n²) state). Still open and decisive: within k3e, whether the κ4 *births* (output
order 4, 15 terms) or only the order ≤ 3 outputs (9 terms) are needed — the births are the expensive
tensor-train family. At n = 48 (CPU, one seed) dropping the k3e births cost 7×; pass 2 measures it at
192 and 256 (`fam:±<family>_4/_3`).

## Status of the kill rules

- K4c3/K3c2 at n = 256: not yet measured (pass 2).
- Generations: unmeasurable by ablation (see above); rule void.
- Families: two of four droppable → rule passed.

---

# Pass 2 (commit 09e789a, received 2026-10-19) — families split by output order; n = 256 still blocked

> Raw report: `results/2026-10-18_gpu_round3_pass2_as_received.md`. n = 192, L = 16, 3 seeds, device MC
> truth N = 1e7; ratios are geometric means over seeds, CI = bootstrap over seeds.

## Families by output order at n = 192 (ratio to K4c3; "_4" = terms writing κ4, "_3" = terms writing orders ≤ 3)

| removed | / K4c3 [95 % CI] | recovery | reading |
|---|---|---|---|
| rs_3 (random-sign κ4 block → mean/cov/κ3) | 4728 [1450, 19127] | −3.1 | the read-out of the κ4 memory; without it the chain is worse than K3c2 by 484× (unstable) |
| k3e_3 (κ3 block + edge → orders ≤ 3) | 64.8 [58, 69] | −1.0 | essential |
| k3e_4 (κ3 block + edge → κ4, the births) | 26.0 [19, 32] | −0.6 | essential |
| rs_4 (transmission of the random-sign slices) | 13.0 [11, 16] | −0.2 | essential |
| c3_3 (three edges → orders ≤ 3) | 1.74 [1.67, 1.81] | 0.73 | minor |
| p4e_3 (paired κ4 + edge → orders ≤ 3) | 1.50 [1.40, 1.63] | 0.80 | minor |
| c3_4 (three edges → κ4) | 1.03 [0.83, 1.22] | 0.99 | droppable |
| p4e_4 (paired κ4 + edge → κ4) | 0.96 [0.77, 1.17] | 1.02 | droppable |

Constructive sets (cost ≤ 2 terms always on):

| kept | / K4c3 | recovery |
|---|---|---|
| rs + k3e_3 + p4e_3 (no κ4 births beyond cost 2) | 42.1 | −0.81 |
| rs + k3e_3 + p4e_3 + c3_3 | 44.0 | −0.83 |
| rs + k3e_3 + p4e_4 + p4e_3 | 30.8 | −0.66 |
| rs + k3e_4 + k3e_3 + p4e_3 | **2.11** [1.70, 2.96] | 0.64 |
| rs + k3e + p4e | 0.99 | 1.00 |
| rs + k3e + c3 | 1.12 | 0.94 |

Reading [measured, n = 192, L = 16; 3 seeds]:

1. **The κ4 sources that matter are the κ3-block + edge births (k3e_4)**: without them, nothing else
   recovers the gap (42×); with them and the cheap read-outs, 64 % of the log-gap is recovered. The
   three-edge births (c3_4, the Gaussian-input κ4 sources) and the paired-κ4 + edge births (p4e_4)
   add nothing at this width. For a factored chain this fixes the design: the random-sign κ4(z) must be
   born from products of the factored κ3(h) slices with the covariance, not from covariance chains.
2. **The read-out terms (rs_3) are the most important terms of the whole cap-3 set**: a factored κ4
   memory is worthless unless its contraction into the mean/covariance/κ3 is computed (and it is what
   stabilises the chain — removing it diverges, consistent with the gen:A finding of pass 1).
3. The remaining 2.11× (recovery 0.64 → 1.0) comes from c3_3 and the p4e family jointly; each alone is
   worth 1.5–1.7×, and both are O(n²)-state contractions in a factored chain (cheap). So the full
   factored target is: rs (transmission + read-out) + k3e (births + read-out) + p4e_3 + c3_3; droppable:
   c3_4, p4e_4 (the two κ4 births that would have needed tensor-train state built from three
   covariance edges).
4. [hypothesis] Whether these ratios hold at n = 1024 is untested; the per-family ratios at n = 192 are
   the only measurement, and the width exponent of each family is unknown (pass 1 measured n = 192 only).

## n = 256: all rows with the random-sign transmission terms failed again (OutOfMemoryError, 2^34 bytes, ~5 s)

Diagnosed on the CPU (2026-10-19, `tracemalloc` peak at n = 72, float32): with the rs_4 terms on, the peak
is **3.09 n⁴** instead of 2.50 n⁴. Cause: in `power_cumulants` the operand list `ops` of the *last*
order-4 term survives the term loop; for configurations with rs_4 the last (1,1,1,1) term is the
transmission term whose operand is the n⁴ pre-activation tensor itself, so `del Z[4]` did not free it
and the symmetrizer's buffer was the third n⁴ tensor: 3 × 16 GiB = 48 GiB > the 46 GiB available (the
2^34-byte allocation is exactly one n⁴ float32 tensor at n = 256). Pass 1's "fix" (broadcasting path,
pool release) addressed a different, non-existent cause [the pass-1 diagnosis was wrong and is
superseded]. Fix: `del ops, subs` after every term (commit of 2026-10-19); peak back to 2.50 n⁴ with
rs_4 on, results unchanged (`tests/test_dense.py`). The report now prints the failing frame of the
traceback in error rows. Rerun: `bash scripts/gpu_run.sh --profile round3` (only errored rows recompute).

Rows that did run at n = 256 (3 seeds, ratio to K3c2 = 7.165e-6): K4c2 0.35 (0.42 at n = 192, 0.64 at 160,
0.84 at 128 — falling, ≈ n^-1.3 over the last doubling); fam:−rs 0.70; fam:−rs_4 0.72; fam:+k3e+c3 0.62.

## Status of the kill rules after pass 2

- K4c3/K3c2 at n = 256 > 0.09 → still unmeasured (memory bug fixed, rerun pending).
- Generations: void (unmeasurable by ablation, pass 1).
- Families: c3_4 and p4e_4 droppable (≤ 1.03×) → passed; the expensive κ4 births from κ3 + edge are required.

---

# Pass 3 (commit 45eb12f, received 2026-10-19) — n = 256 measured: GO

> Raw report: `results/2026-10-19_gpu_round3_pass3_as_received.md`. n = 256, L = 16, seeds 0–2, device MC
> truth N = 1e7 (floor 1.1e-8, 2 % of K4c3). The memory fix held: every requested n = 256 row ran (K4c3
> 95 s/run, peak within the 2.5 n⁴ budget). Rows still showing OutOfMemoryError are stale pass-2 entries
> of configurations no longer in the round-3 list (whole families, gen:A); the report now labels such
> rows "stale" and `k4_gonogo` records the requested list and the commit per row.

## The kill-rule number

| n (L = 16) | 128 | 160 | 192 | **256** |
|---|---|---|---|---|
| K4c3 / K3c2 | 0.34 | 0.22 | 0.13 | **0.06 [0.05, 0.08]** |
| K4c3 (abs.) | 1.02e-5 | 3.97e-6 | 2.19e-6 | 4.84e-7 |
| K3c2 (abs.) | 2.60e-5 | 1.94e-5 | 1.70e-5 | 7.17e-6 |
| seeds | 5 | 5 | 5 | 3 |

[measured] 0.06 < 0.09 (pre-registered kill line, `docs/notes/w7_prereg_n256_L8.md`): **the factored-κ4
program is GO**; the pre-registered point prediction 0.10 [0.05, 0.20] is confirmed (inside the interval).
Exponents over 128 → 256 at depth 16: ratio n^-2.5, K4c3 n^-4.4, K3c2 n^-1.9 (the K3 exponent agrees
with the n^-1.99 measured 256 → 1024 on the dataset). Caveat: n = 256 uses 3 seeds and the K3c2 drop
192 → 256 (2.4×) is steeper than 160 → 192 (1.1×), so the ratio's exponent is uncertain by ~±0.5.

Extrapolation to n = 1024 [hypothesis; the n ≤ 256 fits of 2026-10-17 over-predicted the 1024 gains by
1.5–2.5×]: with K3c2(1024) = 4.29e-7 measured, a dense cap-3 K=4 chain would reach

| assumed ratio law from 256 | ratio at 1024 | K4c3 at 1024 | vs V29 raw 2.13e-8 |
|---|---|---|---|
| n^-1 (asymptotic order counting: K=4 truncation O(n^-3) vs K=3 O(n^-2)) | 0.015 | 6.4e-9 | 3.3× better |
| n^-2.5 (the measured 128 → 256 slope, pre-asymptotic) | 0.0019 | 8.1e-10 | 26× better |

Either bound beats V29's raw error, and at a C/B of 0.1–0.25 either would beat the leader's 1.42e-9
score (raw 1.4e-8 suffices at the 0.1 cost floor). The dense chain itself is unusable at 1024 (n⁴
float32 = 4 TB); the gain has to be realised by the factored program, whose losses are not yet measured.

## Families at n = 256 (3 seeds; ratio to K4c3 when removed; n = 192 in brackets)

| removed | / K4c3 [95 % CI] | n = 192 | trend |
|---|---|---|---|
| rs_3 (κ4 read-out) | 4504 [1477, 11493] | 4728 | unchanged: unstable without |
| k3e_3 | 83.7 [58, 158] | 64.8 | growing |
| k3e_4 (κ4 births from κ3 + edge) | 46.6 [35, 80] | 26.0 | **growing** |
| rs_4 (random-sign transmission) | 11.3 [9.7, 13.2] | 13.0 | flat |
| p4e_4 | 1.74 [1.01, 3.13] | 0.96 | no longer clearly droppable |
| c3_4 | 1.73 [0.87, 3.18] | 1.03 | no longer clearly droppable |
| p4e_3 | 1.64 [1.58, 1.76] | 1.50 | flat |
| c3_3 | 1.63 [1.54, 1.68] | 1.74 | flat |

Constructive: rs + k3e + p4e (no c3) 1.39 [0.85, 2.31] (192: 0.99); rs + k3e_4 + k3e_3 + p4e_3 **4.07**
[2.5, 7.6], recovery 0.49 (192: 2.11, 0.64); without the k3e_4 births 74.8 (192: 42).

Reading [measured, 3 seeds, wide CIs]: the ranking is stable (read-outs ≫ κ3-block births ≫ transmission
≫ the rest) and the weight of the κ3-block + edge births *increases* with width, as does the importance
of everything beyond the minimal set: the minimal set's recovery fell from 0.64 to 0.49, and the two
births previously called droppable (c3_4, p4e_4) now cost ~1.7× each with CIs touching 1. Conclusion for
the factored design: carry **all** cost-3 families; the cheap approximation "rs + k3e only" loses a
factor that grows with n (2.1× → 4.1× from 192 to 256) and would eat most of the extrapolated gain at
1024. The expensive objects are therefore the random-sign κ4(h) memory itself (transmission on four
modes) and its births from κ3(h) ⊗ C — their compressibility (tensor-train / matricisation ranks as a
function of n) is the one quantity the program now depends on and has not been measured.

## Kill rules after pass 3

- K4c3/K3c2 at n = 256 > 0.09 → **0.06, passed (GO)**.
- Generations: void (unmeasurable by ablation).
- Families: at n = 192 two were droppable; at n = 256 none clearly is → the rule's literal reading
  ("no family can be dropped") is now borderline, but it measured substitutability, and the chain with
  rs + k3e + p4e (no c3) is still within 1.4× [0.85, 2.31]; recorded as "carry all, drop none" rather
  than as a kill.
