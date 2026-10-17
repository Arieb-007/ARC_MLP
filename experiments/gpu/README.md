# GPU pipeline (`scripts/gpu_run.sh`)

> Written by an LLM agent (Claude Code) for team Arieb-007. Research tooling; nothing here is part
> of a submission bundle.

One command on a CUDA box (tested design target: one A40, 48 GB) runs every algorithm variant we
want measured at the official shape and writes `results/<date>_gpu_run.md` (+ `.json`):

```bash
git clone https://github.com/Arieb-007/ARC_MLP && cd ARC_MLP
bash scripts/gpu_run.sh                      # profile fast (~3 h); --profile quick (~1.5 h) | full (~8 h) | smoke (~15 min)
bash scripts/gpu_run.sh --push               # also commits the report to a branch gpu-results-<date> (needs GH_TOKEN or push rights)
```

Stages (each resumable; a failure is recorded in the report, not fatal):

| stage | what | device | output |
|---|---|---|---|
| setup | venv `venv_gpu`, requirements, `cupy-cuda1Nx` matching `nvidia-smi` | — | `data/gpu/logs/pip.log` |
| smoke | the device port must reproduce the CPU chains and MC at tiny sizes | GPU | `data/gpu/logs/smoke.log` |
| data | materialise MLPs + the organizers' MC reference from `hf://aicrowd/arc-whestbench-public-2026@v2-phase2` split `mini` | CPU | `data/gpu/mlps/mlp_*.npz` |
| cpu (background) | V29 (+leaf 8) baseline with `V17_DEBUG` features; knob sweep (Algorithm D); confirmation of the best knob on held-out MLPs | CPU, parallel | `data/gpu/v29/<tag>/` |
| gpu: mc_check | our device MC (N = 1e8) vs the dataset reference | GPU | `data/gpu/mc_check/` |
| gpu: dense | dense K=2/K=3 and K=3 + coherent κ4 references at 1024×16 (pre-registered P1/P2) | GPU | `data/gpu/dense/` |
| gpu: k4 | dense K=4 cap 3 at n = 128/160/192, depth 16, κ4 slices switched one at a time (Algorithm A go/no-go) | GPU | `data/gpu/k4/` |
| corrector | per-neuron residual corrector on V29 (Algorithm C), grouped CV + held-out test | CPU | `data/gpu/corrector/` |
| report | everything → `results/<date>_gpu_run.{md,json}` | — | |

Profiles set the MLP index ranges, variant lists and sample counts (see the `case` block in the
script); every value can be overridden from the environment, e.g. `DENSE_IDX=0-19 K4_N="128 192"`.
`--stages` selects stages (`setup,smoke,data,cpu,gpu,corrector,report`). Weight orientation: datasets
store `w` with `z = w.T @ h`; `common.ours_orientation` transposes for our chains and the `K2`
variant checks the result against the kit's covariance closure (max |Δ| is in the report).

Local dry run on a CPU box (no CUDA; tiny sizes, a local bake as the dataset):

```bash
PYTHON=venv/bin/python DATASET=data/bake_1024x16_s0-3_n2e4 SPLIT=dev PROFILE=smoke DENSE_VARIANTS=K2 \
  K4_N=32 K4_NMC=2e5 MC_N=2e5 ALLOW_CPU_GPU_STAGES=1 GPU_DATA_DIR=$PWD/data/gpu_dry \
  bash scripts/gpu_run.sh --device cpu --stages smoke,data,cpu,gpu,corrector,report
```

## Round 2 (`--profile round2`, ~2–3 h): conditional-coherent proxies of the random-sign κ4 slices

Question: can the (2,1,1) slice's effect (12× of the κ4 gain at depth 16, round 1) be bought at O(n²)
by its regression on the covariance, κ4(z)_{aabc} ≈ λ_a C_bc (`arcmlp/estimators/cc_proxy.py`)?
Stages: `k4` with the `round2` config set (oracle fit vs cheap formula, (2,1,1) and (3,1), with and
without the (1,1,1,1) transmission and the full cap-3 set; n = 128/160/192, L = 16, 5 seeds), `k4L8`
(depth-8 ladder), `dense` with the light chain + proxies at 1024×16 on 20 mini MLPs paired against
the round-1 V29 baseline. Kill rule: oracle (2,1,1) recovers < 30 % of the paired→K4c3 log-gap.

```bash
bash scripts/gpu_run.sh --profile round2      # reuses round-1 data/gpu (MLPs, V29 baseline, MC truths)
```

Round 2 also carries **Algorithm E** (`arcmlp/estimators/spiked.py`, from Areeb's RMT suggestions): the
light chain with the covariance projected to diag + rank-r spike after every ReLU (`coh2_spike0/64/256`
in `dense`) and the `structure` stage (off-diagonal spectral fractions, pivoted-Cholesky anchor R², the
same for the κ3 (2,1) matrix and the κ4 memory B, at 1024×16). Pre-registered: top-64 eigenvectors hold
< 15 % of the off-diagonal energy and 64 anchors explain < 10 % of a neuron's variance; the spike-64
chain is > 10× worse than the full chain. (3-point moment-matched quadrature for the ReLU mean was
rejected by arithmetic: it gives 0.289 instead of 0.399 for a standard Gaussian.)
