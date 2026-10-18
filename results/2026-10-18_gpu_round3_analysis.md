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
