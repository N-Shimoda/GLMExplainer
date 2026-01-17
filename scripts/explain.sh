#!/bin/bash

LOG_FILE="logs/explain.log"
rm -f "$LOG_FILE"
mkdir -p logs

log() {
	# Output to both stdout and log file
	local ts
	ts="$(date '+%Y-%m-%d %H:%M:%S')"
	echo "[$ts] $*" | tee -a "$LOG_FILE"
}

format_duration() {
	local total=$1
	local hours=$((total / 3600))
	local minutes=$(((total % 3600) / 60))
	local seconds=$((total % 60))
	printf "%dh %02dmin %02dsec" "$hours" "$minutes" "$seconds"
}

run_explain_loop() {
	local subset="$1"
	local epochs="$2"
	local lr="$3"
	local edge_size="$4"
	local edge_ent="$5"

	cmd=(
		torchrun --nproc_per_node=2 explain.py
		--dataset MotifQA --subset "${subset}"
		--model-path outputs/"${subset}"
		--target-pos-samples
		--num-trials 5
		--epochs $epochs --lr $lr
		--edge-size $edge_size --edge-ent $edge_ent
		--wandb
	)
	log "[INFO] Starting explanation with edge_size: $edge_size, edge_ent: $edge_ent"
	log "[START] ${cmd[*]}"
	start_ts=$(date +%s)

	# Run torchrun and get exit code (avoid set -e effect)
	set +e
	"${cmd[@]}"
	rc=$?
	set -e

	end_ts=$(date +%s)
	dur=$((end_ts - start_ts))

	# Report status
	if [[ $rc -eq 0 ]]; then
		log "[COMPLETED] edge_size=${edge_size} edge_ent=${edge_ent} duration=$(format_duration "$dur")"
	else
		log "[ERROR] edge_size=${edge_size} edge_ent=${edge_ent} rc=${rc} duration=$(format_duration "$dur")"
	fi
}

subsets=(
	# ba_shapes
	# tree_cycle
	tree_grid
	# ba_two_motifs
)

for subset in "${subsets[@]}"; do
	log "[INFO] Starting explanations for subset='${subset}'"
	for epochs in 200; do
		for lr in 0.1; do
			for edge_size in 0.0005 0.005 0.05 0.5 5.0; do
				for edge_ent in 1.0; do
					run_explain_loop "$subset" "$epochs" "$lr" "$edge_size" "$edge_ent"
				done
			done
		done
	done
done
