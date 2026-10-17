#!/usr/bin/env bash
# Commit the GPU-run report (results/*_gpu_run.*, results/*corrector*) and push to a results branch.
# Uses GH_TOKEN (https) if set, else whatever credentials the clone has. Never touches other files.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
BR="gpu-results-$(date -u +%Y%m%d-%H%M)"
git config user.name > /dev/null 2>&1 || git config user.name "ARC_MLP gpu runner"
git config user.email > /dev/null 2>&1 || git config user.email "gpu-runner@arc-mlp.invalid"
git checkout -q -b "$BR"
git add results/*_gpu_run.md results/*_gpu_run.json 2>/dev/null || true
if git diff --cached --quiet; then echo "nothing to commit"; exit 0; fi
git commit -q -m "GPU run report $(date -u +%Y-%m-%dT%H:%MZ) (results/$(ls -1t results/*_gpu_run.md | head -1 | xargs basename))

Produced by scripts/gpu_run.sh (LLM-written pipeline, Claude Code) on $(hostname)."
URL=$(git remote get-url origin)
if [ -n "${GH_TOKEN:-}" ]; then
  URL=$(echo "$URL" | sed -E "s#https://(.*@)?github.com/#https://x-access-token:${GH_TOKEN}@github.com/#")
fi
for i in 1 2 3 4; do git push -q "$URL" "HEAD:$BR" && { echo "pushed branch $BR"; exit 0; }; sleep $((2 ** i)); done
echo "push failed; the report is committed locally on branch $BR" >&2; exit 1
