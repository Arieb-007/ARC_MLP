#!/usr/bin/env bash
# Commit results/ (and only results/) and push, from a notebook that ran kaggle_bootstrap.sh.
# Usage: scripts/push_results.sh "message naming the results file"
# Needs the deploy key with "Allow write access"; otherwise the push fails and nothing else changes.
set -euo pipefail
DIR="${ARC_MLP_DIR:-/kaggle/working/ARC_MLP}"; KEY="$HOME/.ssh/arc_mlp"
BRANCH="${ARC_MLP_BRANCH:-$(git -C "$DIR" rev-parse --abbrev-ref HEAD)}"
MSG="${1:-results: notebook run $(date -u +%Y-%m-%dT%H:%MZ)}"
export GIT_SSH_COMMAND="ssh -i $KEY -o IdentitiesOnly=yes"
git -C "$DIR" config user.name  >/dev/null 2>&1 || git -C "$DIR" config user.name  "ARC_MLP notebook"
git -C "$DIR" config user.email >/dev/null 2>&1 || git -C "$DIR" config user.email "notebook@arc-mlp.invalid"
git -C "$DIR" add results/
if git -C "$DIR" diff --cached --quiet; then echo "nothing to commit in results/"; exit 0; fi
git -C "$DIR" commit -q -m "$MSG"
for i in 1 2 3 4; do
  if git -C "$DIR" pull --rebase -q origin "$BRANCH" && git -C "$DIR" push -q origin "HEAD:$BRANCH"; then
    echo "pushed $(git -C "$DIR" rev-parse --short HEAD) to $BRANCH"; exit 0; fi
  sleep $((2**i))
done
echo "push failed (does the deploy key have write access?)" >&2; exit 1
