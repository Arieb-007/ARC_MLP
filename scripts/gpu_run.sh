#!/usr/bin/env bash
# One-shot GPU pipeline for ARC_MLP: environment, data, every algorithm variant, report.
# Written by an LLM agent (Claude Code) for team Arieb-007. See experiments/gpu/README.md.
#
#   git clone https://github.com/Arieb-007/ARC_MLP && cd ARC_MLP && bash scripts/gpu_run.sh
#
# Options (flags or environment variables):
#   --profile smoke|quick|fast|full   (default fast)    --stages all|setup,smoke,data,cpu,gpu,corrector,report
#   --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2   --split mini   --device auto|cuda|cpu   --push
# Everything is resumable: finished (variant, MLP) results are skipped on a rerun. Logs: data/gpu/logs/.
# Output: results/<date>_gpu_run.md (+ .json); with --push also committed to branch gpu-results-<date>.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT=$(pwd)
export GPU_RUN_ARGS="$*"
PROFILE=${PROFILE:-fast}; STAGES=${STAGES:-all}; PUSH=${PUSH:-0}; DEVICE=${DEVICE:-auto}
DATASET=${DATASET:-hf://aicrowd/arc-whestbench-public-2026@v2-phase2}; SPLIT=${SPLIT:-mini}
ALLOW_CPU_GPU_STAGES=${ALLOW_CPU_GPU_STAGES:-0}
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE=$2; shift 2;; --stages) STAGES=$2; shift 2;; --push) PUSH=1; shift;;
    --dataset) DATASET=$2; shift 2;; --split) SPLIT=$2; shift 2;; --device) DEVICE=$2; shift 2;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
export PROFILE
GD=${GPU_DATA_DIR:-$ROOT/data/gpu}; export GPU_DATA_DIR="$GD"
LOG="$GD/logs"; MLPS="$GD/mlps"; mkdir -p "$LOG" "$MLPS"
PY=${PYTHON:-python3}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUPY_TF32=0 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1

# ------------------------------------------------------------------ profiles
case "$PROFILE" in
  smoke) FETCH_IDX=${FETCH_IDX:-0-3}; BASE_IDX=${BASE_IDX:-0-3}; SWEEP_IDX=${SWEEP_IDX:-0-1}; CONFIRM_IDX=${CONFIRM_IDX:-2-3}; DENSE_IDX=${DENSE_IDX:-0}; DENSE_VARIANTS=${DENSE_VARIANTS:-K2,K3c2,coh2}
         K4_N=${K4_N:-"128"}; K4_SEEDS=${K4_SEEDS:-0}; K4_NMC=${K4_NMC:-1e6}; K4_CONFIGS=${K4_CONFIGS:-K3c2,K4c3,paired_c3+31,coh2}; MC_IDX=${MC_IDX:-0}; MC_N=${MC_N:-1e7}
         TRAIN_IDX=${TRAIN_IDX:-2-3}; TEST_IDX=${TEST_IDX:-0-1}; SWEEP_VARIANTS=${SWEEP_VARIANTS:-"s6:V26_STRASSEN=6|lam105:V17_LAM_SCALE=1.05"};;
  quick) FETCH_IDX=${FETCH_IDX:-0-39}; BASE_IDX=${BASE_IDX:-0-39}; SWEEP_IDX=${SWEEP_IDX:-0-9}; CONFIRM_IDX=${CONFIRM_IDX:-20-29}; DENSE_IDX=${DENSE_IDX:-0-4}; DENSE_VARIANTS=${DENSE_VARIANTS:-K2,K3c2,coh2,ex4c2}
         K4_N=${K4_N:-"128 160"}; K4_SEEDS=${K4_SEEDS:-0-2}; K4_NMC=${K4_NMC:-1e7}; K4_CONFIGS=${K4_CONFIGS:-all}; MC_IDX=${MC_IDX:-0}; MC_N=${MC_N:-3e7}
         TRAIN_IDX=${TRAIN_IDX:-20-39}; TEST_IDX=${TEST_IDX:-0-19}
         SWEEP_VARIANTS=${SWEEP_VARIANTS:-"s6:V26_STRASSEN=6|rold256:V21_R_OLD=256|age3:V21_AGE_OLD=3|lam085:V17_LAM_SCALE=0.85|lam105:V17_LAM_SCALE=1.05|nowk431:V17_NO_WK431=1"};;
  fast)  FETCH_IDX=${FETCH_IDX:-0-39}; BASE_IDX=${BASE_IDX:-0-39}; SWEEP_IDX=${SWEEP_IDX:-0-19}; CONFIRM_IDX=${CONFIRM_IDX:-20-39}; DENSE_IDX=${DENSE_IDX:-0-9}; DENSE_VARIANTS=${DENSE_VARIANTS:-K2,K3c2,coh2,ex4c2,K3c3}
         K4_N=${K4_N:-"128 160 192"}; K4_SEEDS=${K4_SEEDS:-0-2}; K4_NMC=${K4_NMC:-1e7}; K4_CONFIGS=${K4_CONFIGS:-all}; MC_IDX=${MC_IDX:-0-1}; MC_N=${MC_N:-1e8}
         TRAIN_IDX=${TRAIN_IDX:-20-39}; TEST_IDX=${TEST_IDX:-0-19}
         SWEEP_VARIANTS=${SWEEP_VARIANTS:-"s6:V26_STRASSEN=6|rold256:V21_R_OLD=256|rold2_128:V24_R_OLD2=128|age3:V21_AGE_OLD=3|age2off:V24_AGE_OLD2=0|lam085:V17_LAM_SCALE=0.85|lam105:V17_LAM_SCALE=1.05|nowk431:V17_NO_WK431=1"};;
  full)  FETCH_IDX=${FETCH_IDX:-0-99}; BASE_IDX=${BASE_IDX:-0-99}; SWEEP_IDX=${SWEEP_IDX:-0-19}; CONFIRM_IDX=${CONFIRM_IDX:-20-39}; DENSE_IDX=${DENSE_IDX:-0-19}; DENSE_VARIANTS=${DENSE_VARIANTS:-K2,K3c2,K3c3,coh2,coh3,ex4c2,exallc2}
         K4_N=${K4_N:-"128 160 192"}; K4_SEEDS=${K4_SEEDS:-0-4}; K4_NMC=${K4_NMC:-1e7}; K4_CONFIGS=${K4_CONFIGS:-all}; MC_IDX=${MC_IDX:-0-1}; MC_N=${MC_N:-1e8}
         TRAIN_IDX=${TRAIN_IDX:-20-99}; TEST_IDX=${TEST_IDX:-0-19}
         SWEEP_VARIANTS=${SWEEP_VARIANTS:-"s4:V26_STRASSEN=4|s6:V26_STRASSEN=6|min16:V26_STRASSEN_MIN=16|rold256:V21_R_OLD=256|rold2_128:V24_R_OLD2=128|age3:V21_AGE_OLD=3|age5:V21_AGE_OLD=5|age2off:V24_AGE_OLD2=0|lam085:V17_LAM_SCALE=0.85|lam105:V17_LAM_SCALE=1.05|beta0:V25_BETA=0|nowk431:V17_NO_WK431=1|nofb:V18_NO_FB=1|noregen:V17_NO_REGEN=1"};;
  *) echo "unknown profile $PROFILE" >&2; exit 2;;
esac
[ "$K4_CONFIGS" = all ] && K4_CONFIGS="K3c2,K3c3,K4c2,K4c3,K4c3-no31,K4c3-no211,K4c3-no1111,paired_c3,paired_c3+31,paired_c3+211,coh2,ex4c2"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
want() { [ "$STAGES" = all ] || [[ ",$STAGES," == *",$1,"* ]]; }
status() { "$PY" experiments/gpu/status.py set "$@" 2>/dev/null || true; }
# stage NAME [timeout-seconds] -- command...: runs, times, records; a failure never aborts the pipeline
stage() {
  local name=$1 to=$2; shift 2
  local t0=$(date +%s)
  log "stage $name: $*"
  if timeout "$to" "$@" > "$LOG/$name.log" 2>&1; then
    status "$name" ok $(( $(date +%s) - t0 )) ""; log "stage $name ok ($(( $(date +%s) - t0 )) s)"; return 0
  else
    local rc=$?
    status "$name" fail $(( $(date +%s) - t0 )) "exit $rc; see data/gpu/logs/$name.log"
    log "stage $name FAILED (exit $rc) — tail of log:"; tail -n 15 "$LOG/$name.log"; return $rc
  fi
}

# ------------------------------------------------------------------ setup
if want setup; then
  t0=$(date +%s)
  if [ ! -x venv_gpu/bin/python ]; then "$PY" -m venv venv_gpu || { echo "venv failed" >&2; exit 1; }; fi
  # shellcheck disable=SC1091
  . venv_gpu/bin/activate; PY=python
  python -m pip install -q -U pip wheel > "$LOG/pip.log" 2>&1
  python -m pip install -q -r requirements.txt mpmath matplotlib >> "$LOG/pip.log" 2>&1 || { tail -5 "$LOG/pip.log"; }
  if command -v nvidia-smi > /dev/null 2>&1 && [ "$DEVICE" != cpu ]; then
    CUDA_MAJOR=$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9]*' | grep -o '[0-9]*$' || true)
    for pkg in "cupy-cuda${CUDA_MAJOR:-12}x" cupy-cuda12x cupy-cuda11x; do
      python -c "import cupy, cupy.cuda.runtime as r; r.getDeviceCount()" > /dev/null 2>&1 && break
      log "pip install $pkg"; python -m pip install -q "$pkg" >> "$LOG/pip.log" 2>&1 || true
    done
  fi
  status setup ok $(( $(date +%s) - t0 )) "$(python -c 'import cupy;print("cupy",cupy.__version__)' 2>/dev/null || echo no-cupy)"
else
  [ -x venv_gpu/bin/python ] && { . venv_gpu/bin/activate; PY=python; }
fi
if [ "$DEVICE" = auto ]; then
  if "$PY" -c "import cupy, cupy.cuda.runtime as r; assert r.getDeviceCount() > 0" > /dev/null 2>&1; then DEVICE=cuda; else DEVICE=cpu; fi
fi
export ARCMLP_DEVICE=$DEVICE
"$PY" experiments/gpu/status.py env > /dev/null 2>&1 || true
log "profile=$PROFILE device=$DEVICE dataset=$DATASET split=$SPLIT stages=$STAGES root=$ROOT"

# ------------------------------------------------------------------ smoke
if want smoke; then
  if ! stage smoke 900 "$PY" experiments/gpu/smoke.py --device "$DEVICE"; then
    if [ "$DEVICE" = cuda ]; then log "cuda smoke test failed: falling back to cpu for the GPU stages (they will be skipped)"; DEVICE=cpu; export ARCMLP_DEVICE=cpu; fi
  fi
fi

# ------------------------------------------------------------------ data
if want data; then
  stage data 7200 "$PY" experiments/gpu/fetch_dataset.py --dataset "$DATASET" --split "$SPLIT" --idx "$FETCH_IDX" --out "$MLPS"
fi
NMLP=$(ls "$MLPS"/mlp_*.npz 2>/dev/null | wc -l)
if [ "$NMLP" -eq 0 ]; then log "no MLPs under $MLPS — only the K4 go/no-go (self-generated weights) can run"; fi

# ------------------------------------------------------------------ CPU block (V29 under flopscope): background
v29_parallel() {   # tag idx features(0|1) env...
  local tag=$1 idx=$2 feats=$3; shift 3
  local P chunks
  P=$("$PY" -c "import os; mem=os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')/2**30; print(max(1, min((os.cpu_count() or 2)-1, int((mem-6)//7))))")
  chunks=$("$PY" -c "import sys; sys.path.insert(0,'experiments/gpu'); from common import parse_idx; ids=parse_idx('$idx'); P=$P; print(' '.join(','.join(map(str,ids[k::P])) for k in range(P) if ids[k::P]))")
  local f=(); [ "$feats" = 1 ] && f=(--features)
  for c in $chunks; do
    "$PY" experiments/gpu/run_v29.py --estimator submission/estimator.py --mlps "$MLPS" --idx "$c" --tag "$tag" --out "$GD/v29" ${f[@]+"${f[@]}"} --env "$@" > "$LOG/v29_${tag}_${c%%,*}.log" 2>&1 &
  done
  wait
}
cpu_block() {
  local t0=$(date +%s)
  v29_parallel baseline "$BASE_IDX" 1
  status cpu_baseline ok $(( $(date +%s) - t0 )) "idx $BASE_IDX"
  IFS='|' read -ra VARS <<< "$SWEEP_VARIANTS"
  for spec in "${VARS[@]}"; do
    local name=${spec%%:*} envs=${spec#*:}; IFS=';' read -ra EV <<< "$envs"
    t0=$(date +%s); v29_parallel "$name" "$SWEEP_IDX" 0 "${EV[@]}"; status "sweep_$name" ok $(( $(date +%s) - t0 )) "${EV[*]}"
  done
  # confirm the best sweep variant (by mean score ratio to the baseline) on held-out MLPs
  local best
  best=$("$PY" - <<EOF
import glob, json, os, numpy as np
D="$GD/v29"; base={}
for p in glob.glob(f"{D}/baseline/mlp_*.json"):
    r=json.load(open(p)); base[r["idx"]]=r
best=None
for d in glob.glob(f"{D}/*"):
    t=os.path.basename(d)
    if t in ("baseline",) or t.startswith("confirm_"): continue
    rows=[json.load(open(p)) for p in glob.glob(f"{d}/mlp_*.json")]
    c=[r for r in rows if r["idx"] in base and not r.get("error") and not base[r["idx"]].get("error")]
    if len(c) < 2: continue
    ratio=float(np.mean([np.log(r["score"]/base[r["idx"]]["score"]) for r in c]))
    if best is None or ratio < best[0]: best=(ratio, t, rows[0]["env"])
print("" if best is None or best[0] >= 0 else best[1] + " " + " ".join(best[2]))
EOF
)
  if [ -n "$best" ]; then
    local bname=${best%% *} benv=${best#* }
    t0=$(date +%s); v29_parallel "confirm_$bname" "$CONFIRM_IDX" 0 $benv; v29_parallel baseline "$CONFIRM_IDX" 1
    status "confirm_$bname" ok $(( $(date +%s) - t0 )) "idx $CONFIRM_IDX env $benv"
  fi
}
CPU_PID=""
if want cpu && [ "$NMLP" -gt 0 ]; then
  cpu_block > "$LOG/cpu_block.log" 2>&1 &
  CPU_PID=$!
  log "CPU block started in the background (pid $CPU_PID; log $LOG/cpu_block.log)"
fi

# ------------------------------------------------------------------ GPU block: foreground
if want gpu; then
  if [ "$DEVICE" = cuda ] || [ "$ALLOW_CPU_GPU_STAGES" = 1 ]; then
    if [ "$NMLP" -gt 0 ]; then
      stage mc_check 7200 "$PY" experiments/gpu/mc_check.py --mlps "$MLPS" --idx "$MC_IDX" --N "$MC_N" --device "$DEVICE" --out "$GD/mc_check"
      stage dense 14400 "$PY" experiments/gpu/dense_refs.py --mlps "$MLPS" --idx "$DENSE_IDX" --variants "$DENSE_VARIANTS" --device "$DEVICE" --out "$GD/dense"
    fi
    # shellcheck disable=SC2086
    stage k4 14400 "$PY" experiments/gpu/k4_gonogo.py --n $K4_N --L 16 --seeds "$K4_SEEDS" --N "$K4_NMC" --device "$DEVICE" --configs "$K4_CONFIGS" --out "$GD/k4" --mc-cache "$GD/mc"
  else
    status gpu skip 0 "no CUDA device (set ALLOW_CPU_GPU_STAGES=1 to run the GPU stages on the CPU at small sizes)"
    log "GPU stages skipped: no CUDA device"
  fi
fi

# ------------------------------------------------------------------ join, corrector, report
if [ -n "$CPU_PID" ]; then log "waiting for the CPU block (pid $CPU_PID)"; wait "$CPU_PID" || true; tail -n 5 "$LOG/cpu_block.log"; fi
if want corrector && [ "$NMLP" -gt 0 ]; then
  stage corrector 3600 "$PY" experiments/gpu/corrector.py --v29 "$GD/v29/baseline" --mlps "$MLPS" --train-idx "$TRAIN_IDX" --test-idx "$TEST_IDX" --out "$GD/corrector"
fi
if want report; then
  stage report 600 "$PY" experiments/gpu/report.py
  ls -1 results/*_gpu_run.md 2>/dev/null | tail -1
fi
if [ "$PUSH" = 1 ]; then bash scripts/gpu_push.sh || true; fi
log "done. Report: $(ls -1t results/*_gpu_run.md 2>/dev/null | head -1)"
