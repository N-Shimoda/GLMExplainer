#!/usr/bin/env bash
set -euo pipefail

# Move to repository root (assuming this script is directly under scripts/)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_gt_custom.log"

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
	# cycle_check
	# triangle_counting
)

for subset in "${subsets[@]}"; do
	cmd=(
		torchrun --nproc_per_node=2 train.py --subset "${subset}"
		--use-custom-dataset
		--base-model "Qwen/Qwen3-4B-Base"
		--gnn-type "GCN"
		--node-feat-dim 8 --pos-emb-dim 8
		--gnn-hidden-dim 256 --gnn-out-dim 512 --num-gnn-layers 4
		--num-proj-layers 2
		--num-graph-tokens 4
		--epochs 3
		--optim "lion" --lr 0.01
		--lr-scheduler-type "linear" --warmup-ratio 0.05
		--do-eval --wandb
	)
	log "[START] ${cmd[*]}"
	start_ts=$(date +%s)

	# Run torchrun and get exit code (avoid set -e effect)
	set +e
	"${cmd[@]}"
	rc=$?
	set -e

	end_ts=$(date +%s)
	dur=$((end_ts - start_ts))

	if [[ $rc -eq 0 ]]; then
		log "[COMPLETED] subset=${subset} duration=${dur}s"
	else
		log "[ERROR] subset=${subset} rc=${rc} duration=${dur}s"
	fi
	echo
done

log "[INFO] ALL subsets finished successfully."
