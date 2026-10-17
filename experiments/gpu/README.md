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
