LOG_FILE="logs/explain_ba_shapes.log"
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

for epochs in 200; do
	for lr in 0.01; do
		for edge_size in 0.005 0.05 0.5; do
			for edge_ent in 1.0; do
				cmd=(
					torchrun --nproc_per_node=2 explain.py
					--model-path masters/house_check
					--explain-pos-samples
					--num-trials 10
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
			done
		done
	done
done
