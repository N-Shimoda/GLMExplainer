#!/usr/bin/env bash
set -euo pipefail

# Move to repository root (assuming this script is directly under scripts/)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_gt_all.log"

# Remove existing log file
rm -f "$LOG_FILE"

log() {
  # Output to both stdout and log file
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[$ts] $*" | tee -a "$LOG_FILE"
}

subsets=(
  node_count
  edge_count
  cycle_check
  triangle_counting
  maximum_flow
)

for subset in "${subsets[@]}"; do
  cmd=(
    torchrun --nproc_per_node=2 train.py --subset "${subset}"
    --base_model "Qwen/Qwen3-4B-Base"
    --node_feat_dim 8
    --gnn_hidden_dim 256 --gnn_out_dim 512 --num_gnn_layers 4
    --epochs 3 --lr 0.01
    --do_eval --wandb
  )
  log "[START] ${cmd[*]}"
  start_ts=$(date +%s)

  # Run torchrun and get exit code (avoid set -e effect)
  set +e
  "${cmd[@]}"
  rc=$?
  set -e

  end_ts=$(date +%s)
  dur=$(( end_ts - start_ts ))

  if [[ $rc -eq 0 ]]; then
    log "[COMPLETED] subset=${subset} duration=${dur}s"
  else
    log "[ERROR] subset=${subset} rc=${rc} duration=${dur}s"
  fi
  echo
done

log "[INFO] ALL subsets finished successfully."