#!/usr/bin/env bash
set -euo pipefail


REMOTE_HOST="${REMOTE_HOST:-coma}"
REMOTE_REPO="${REMOTE_REPO:-fully-inductive-cardinality-estimation}"
BRANCH="${BRANCH:-main}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-scripts/train_tim.sbatch}"
SBATCH_NODE="${SBATCH_NODE:-gpu-nvidia-h200-2}"

ssh "$REMOTE_HOST" bash -s -- \
  "$REMOTE_REPO" "$BRANCH" "$SBATCH_SCRIPT" "$SBATCH_NODE" \
  <<'EOF'
set -euo pipefail

repo="$1"
branch="$2"
sbatch_script="$3"
node="$4"

cd "$repo"
git pull --ff-only

sbatch --parsable \
  -w "$node" \
  "$sbatch_script"
EOF