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
	# node_count
	# edge_count
	# cycle_check
	# triangle_counting
	house_check
)

gnns=("GCN" "GAT" "GIN" "GraphSAGE")

for subset in "${subsets[@]}"; do
	for gnn in "${gnns[@]}"; do
		log "[INFO] Starting training for subset='${subset}' with GNN='${gnn}'"
		cmd=(
			torchrun --nproc_per_node=2 train.py --subset "${subset}"
			--num-graph-tokens 4 --node-feat-dim 8 --pos-emb-dim 8
			--gnn-hidden-dim 256 --gnn-out-dim 512 --num-gnn-layers 4
			--epochs 12
			--gnn-type "${gnn}"
			# --optim "lion" --lr 0.01
			# --lr-scheduler-type "linear" --warmup-ratio 0.05
			--optim "adamw" --lr 0.0075 --weight-decay 0.01
			--lr-scheduler-type "cosine" --warmup-ratio 0.05
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
done
log "[INFO] ALL subsets finished successfully."
