#!/usr/bin/env bash
set -euo pipefail
REPO_SSH="${ARC_MLP_REPO:-git@github.com:Arieb-007/ARC_MLP.git}"
DIR="${ARC_MLP_DIR:-/kaggle/working/ARC_MLP}"; KEY="$HOME/.ssh/arc_mlp"
mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
[ -n "${ARC_MLP_DEPLOY_KEY_B64:-}" ] && { echo "$ARC_MLP_DEPLOY_KEY_B64" | base64 -d > "$KEY"; chmod 600 "$KEY"; }
[ -f "$KEY" ] || { echo "deploy key missing" >&2; exit 1; }
ssh-keyscan -t ed25519 github.com >> "$HOME/.ssh/known_hosts" 2>/dev/null
export GIT_SSH_COMMAND="ssh -i $KEY -o IdentitiesOnly=yes"
if [ -d "$DIR/.git" ]; then git -C "$DIR" pull --ff-only; else git clone "$REPO_SSH" "$DIR"; fi
python -m pip install -q -r "$DIR/requirements.txt"; mkdir -p "$DIR/data" "$DIR/results"
echo "ARC_MLP ready at $DIR ($(git -C "$DIR" rev-parse --short HEAD))"
