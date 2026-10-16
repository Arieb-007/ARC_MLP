# CLAUDE.md — conventions for ARC_MLP

Research repo for team Arieb-007's entry to the ARC White-Box Estimation Challenge 2026
(AIcrowd, Phase 2). The standing task brief is **`docs/BRIEF.md`** — read it first.
Current state: **`results/STATUS.md`**. Findings with status labels: `docs/FINDINGS.md`.

## Ground rules

1. **Status labels on every claim**: **proven** (derivation in `docs/`), **measured** (a file in
   `results/` with seed, package versions, commit hash), **hypothesis**. Never round, extrapolate
   or relabel a number into a stronger claim. Negative results are recorded like positive ones.
2. **Numerics**: numpy, float64 for references (`arcmlp/reference/`), float32 for estimators,
   behind `arcmlp.xp` (`arcmlp/flops/backend.py` switches numpy <-> flopscope).
3. **`submission/` is flopscope-only**: it must not import numpy, scipy, threading,
   multiprocessing, subprocess, ctypes/cffi, or touch the flopscope client internals. Only
   `flopscope`, `flopscope.numpy`, `whestbench` contract types and the pure-Python stdlib.
4. **Guard shape-indexed constants** (per-layer tables, width-specific ranks): the grader's smoke
   test uses shapes other than 16 x 1024; every such constant needs a fallback.
5. **Keep third-party licences** (`third_party/*/LICENSE`, and the MIT header in any file derived
   from them).
6. **Never tune on the MLPs you report on**: dev subset = 20 MLPs for iteration; reporting on the
   100-MLP mini split (or a disjoint self-baked set). State which set every number comes from.
7. **Timing** only under `threadpoolctl.threadpool_limits(1)` / `whest run --max-threads 1`; check
   the 0.4 s residual cap, 120 s wall cap and 5 s setup cap before every upload.
8. **Before every experiment** write one sentence stating what result would falsify the idea;
   record the outcome either way. When a term is dropped, state its order in 1/n and whether it is
   coherent or random-sign.
9. **Heavy runs** (official shape, >1e8 MC samples, > ~1 CPU-hour) go to the notebook
   (`notebooks/`, `scripts/kaggle_bootstrap.sh`) unless the machine has >= 4 cores and >= 16 GB.
10. **LLM disclosure**: this repository is written by an LLM agent (Claude Code) directed by Areeb;
    every document says so.
11. **Irreversible actions** (submitting, posting the write-up, spending > ~6 CPU-hours on one
    experiment): ask Areeb first.

## Layout

```
docs/          BRIEF.md (task), FINDINGS.md (status-labelled results), REFERENCES.md, writeup/
arcmlp/        reference/ (MC truth, dense cumulants) · closure/ (ReLU moments) · estimators/ ·
               compress/ (Lyapunov ranks, Gramians, sparsity) · flops/ (xp backend, FLOP counter,
               Strassen) · eval/ (grader-faithful local eval, bootstrap, report)
experiments/   one script per experiment (E1-E3 are the evidence behind FINDINGS F4-F6)
notebooks/     Kaggle/Colab entry points for heavy runs
scripts/       kaggle_bootstrap.sh, push_results.sh
results/       YYYY-MM-DD_<name>.md + .json per experiment; STATUS.md
submission/    flopscope-only bundle (what gets packaged)
third_party/   vendored code with licences (v29 = 504aldo, MIT)
data/          git-ignored (baked datasets, MC truth, dumps)
```

## Commands

```bash
python -m venv venv && . venv/bin/activate && pip install -r requirements.txt
whest validate --estimator submission/estimator.py
whest dataset bake --n-mlps 4 --n-samples 20000 --width 1024 --depth 16 \
    --mlp-seeds data/seeds_0-3.json --split dev --output data/bake_1024x16_s0-3_n2e4
whest run --estimator submission/estimator.py --dataset data/<bake> --runner subprocess \
    --max-threads 1 --detail full --json > data/runs/<name>.json
whest package --estimator submission/estimator.py     # -> submission tarball (<= 50 MiB, <= 50 files)
python experiments/predictability.py                  # E1 (F4)
```

## Session protocol

1. `date -u`; check deadlines in `docs/BRIEF.md` §0.
2. Read `results/STATUS.md`; pick the next acceptance criterion.
3. Work in small commits; name the `results/` file in each commit message.
4. End the session by updating `results/STATUS.md`: what is proven, measured, open, and the next
   acceptance criterion.
