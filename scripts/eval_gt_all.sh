#!/usr/bin/env bash
set -euo pipefail

# Move to repository root (assuming this script is directly under scripts/)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/eval_gt_all.log"

# Remove existing log file
rm -f "$LOG_FILE"

log() {
	# Output to both stdout and log file
	local ts
	ts="$(date '+%Y-%m-%d %H:%M:%S')"
	echo "[$ts] $*" | tee -a "$LOG_FILE"
}

graphqa_subsets=(
	# node_count
	# edge_count
	# cycle_check
	# triangle_counting
	# reachability
	# node_degree
	# edge_existence
)

motifqa_subsets=(
	ba_shapes
	tree_cycle
	tree_grid
	ba_two_motifs
	shortest_path
)

ckpt_indices=(-1)

run_eval_loop() {
	local dataset="$1"
	local subset="$2"
	local ckpt_index="$3"

	cmd=(
		torchrun --standalone --nproc_per_node=2 eval.py
		--dataset "${dataset}" --subset "${subset}"
		--model-path "masters/${subset}"
		--num-trials 5
		--model-index -1
		--ckpt-index "${ckpt_index}"
	)
	log "[START] ${cmd[*]}"
	start_ts=$(date +%s)

	tmp_output_file="$(mktemp)"
	set +e
	"${cmd[@]}" 2>&1 | tee "$tmp_output_file"
	rc=${PIPESTATUS[0]}
	set -e

	end_ts=$(date +%s)
	dur=$((end_ts - start_ts))

	summary_line="$(grep -E '\[SUMMARY\]' "$tmp_output_file" | tail -n 1 || true)"
	rm -f "$tmp_output_file"

	if [[ $rc -eq 0 ]]; then
		acc_note=""
		if [[ -n "$summary_line" ]]; then
			acc_value="$(sed -n 's/.*accuracy=\([0-9.][0-9.]*\).*/\1/p' <<<"$summary_line")"
			if [[ -n "$acc_value" ]]; then
				acc_percent="$(awk -v acc="$acc_value" 'BEGIN { printf "%.3f", acc * 100 }')"
				acc_note=" accuracy=${acc_percent}% (raw=${acc_value})"
				log "[INFO] dataset=${dataset} subset=${subset} ckpt_index=${ckpt_index}${acc_note}"
			else
				log "[WARNING] dataset=${dataset} subset=${subset} ckpt_index=${ckpt_index} accuracy value not found in summary output."
			fi
		else
			log "[WARNING] dataset=${dataset} subset=${subset} ckpt_index=${ckpt_index} summary line not found in eval output."
		fi
		log "[COMPLETED] dataset=${dataset} subset=${subset} ckpt_index=${ckpt_index} duration=${dur}s${acc_note}"
	else
		log "[ERROR] dataset=${dataset} subset=${subset} ckpt_index=${ckpt_index} rc=${rc} duration=${dur}s"
		exit $rc
	fi
	echo
}

for subset in "${graphqa_subsets[@]}"; do
	for ckpt_index in "${ckpt_indices[@]}"; do
		run_eval_loop "GraphQA" "$subset" "$ckpt_index"
	done
done

for subset in "${motifqa_subsets[@]}"; do
	for ckpt_index in "${ckpt_indices[@]}"; do
		run_eval_loop "MotifQA" "$subset" "$ckpt_index"
	done
done

log "[INFO] ALL subsets finished successfully."
