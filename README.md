# ARC_MLP — team Arieb-007, ARC White-Box Estimation Challenge 2026

Estimating per-neuron post-ReLU means of random 1024×16 ReLU MLPs under a FLOP budget
(AIcrowd / Alignment Research Center, Phase 2). The task brief is [`docs/BRIEF.md`](docs/BRIEF.md);
the current state is [`results/STATUS.md`](results/STATUS.md); status-labelled findings are in
[`docs/FINDINGS.md`](docs/FINDINGS.md).

> **LLM disclosure.** This repository — code, documents and experiment logs — is written by an LLM
> agent (Claude Code, Anthropic) directed by Areeb (GitHub `Arieb-007`). Every number in `results/`
> comes from a run recorded there with seed, package versions and commit hash. Claims are labelled
> **proven** / **measured** / **hypothesis**.

> **Third-party code.** `third_party/v29/` is 504aldo's V29 estimator (MIT, unmodified;
> provenance in `third_party/v29/PROVENANCE.md`). `submission/estimator.py` is derived from it
> and carries the MIT notice.

## Layout

| path | content |
|---|---|
| `docs/` | `BRIEF.md`, `FINDINGS.md`, `REFERENCES.md`, `writeup/` (Stage-1 write-up) |
| `arcmlp/` | library: `reference/` (MC truth, dense cumulants), `closure/` (ReLU moments), `estimators/`, `compress/`, `flops/` (numpy↔flopscope `xp`, FLOP counter, Strassen), `eval/` |
| `experiments/` | E1 `predictability.py`, E2 `conditional.py`, E3 `richardson.py` |
| `notebooks/` | entry points for heavy runs on Kaggle/Colab CPU |
| `scripts/` | `kaggle_bootstrap.sh`, `push_results.sh` |
| `results/` | one `YYYY-MM-DD_<name>.{md,json}` per experiment; `STATUS.md`; `stage0_submission.md` |
| `submission/` | flopscope-only bundle that gets packaged and submitted |
| `third_party/` | vendored code with licences |
| `data/` | git-ignored: baked datasets, MC truth, run logs |

## Quick start

```bash
python -m venv venv && . venv/bin/activate
pip install -r requirements.txt
whest validate --estimator submission/estimator.py
python experiments/predictability.py          # ~2 min, 4 cores
```

Local grader-faithful run on a self-baked shard (HF not needed):

```bash
echo '[0, 1, 2, 3]' > data/seeds_0-3.json
whest dataset bake --n-mlps 4 --n-samples 20000 --width 1024 --depth 16 \
    --mlp-seeds data/seeds_0-3.json --split dev --output data/bake_1024x16_s0-3_n2e4
whest run --estimator submission/estimator.py --dataset data/bake_1024x16_s0-3_n2e4 \
    --runner subprocess --max-threads 1 --detail full --json > data/runs/check.json
```

The subprocess runner enforces the grader's 8 GiB address-space limit (`RLIMIT_AS`), the 0.4 s
residual cap and the 120 s wall cap. Residual time on a slow VM over-states the grader's (see
`results/stage0_submission.md`).

## Submitting (Areeb)

```bash
whest login                      # once; AIcrowd API key from https://www.aicrowd.com/participants/me/edit
whest package --estimator submission/estimator.py     # inspect: size, file count
whest submit  --estimator submission/estimator.py --watch
```

## Running heavy jobs in a Kaggle / Colab CPU notebook

One-time GitHub step: create an SSH key pair (`ssh-keygen -t ed25519 -f arc_mlp -N ""`), add
`arc_mlp.pub` under **GitHub → Arieb-007/ARC_MLP → Settings → Deploy keys → Add deploy key**, tick
**Allow write access** if the notebook should push `results/` back, and store
`base64 -w0 arc_mlp` as a notebook secret named `ARC_MLP_DEPLOY_KEY_B64`.

Kaggle (cell 1 — load the secret):

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["ARC_MLP_DEPLOY_KEY_B64"] = UserSecretsClient().get_secret("ARC_MLP_DEPLOY_KEY_B64")
```

Colab (cell 1 — load the secret):

```python
import os
from google.colab import userdata
os.environ["ARC_MLP_DEPLOY_KEY_B64"] = userdata.get("ARC_MLP_DEPLOY_KEY_B64")
os.environ["ARC_MLP_DIR"] = "/content/ARC_MLP"
```

Cell 2 (both): the repo is cloned *by* the bootstrap, so paste the body of
[`scripts/kaggle_bootstrap.sh`](scripts/kaggle_bootstrap.sh) under a `%%bash` line and run it
(env vars set in cell 1 are inherited). It decodes the key to `~/.ssh/arc_mlp`, clones or
fast-forwards the repo into `$ARC_MLP_DIR` (default `/kaggle/working/ARC_MLP`) and installs
`requirements.txt`.

Then run an entry point and push the results:

```bash
cd /kaggle/working/ARC_MLP
python notebooks/run_official_eval.py --estimator submission/estimator.py --split mini --n-mlps 20
python notebooks/run_mc_truth.py --width 256 --depth 16 --seeds 0-9 --n-samples 1e8
scripts/push_results.sh "results: <file name>"
```
