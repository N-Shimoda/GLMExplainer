LOG_FILE="logs/explain_house_check.log"
rm -f "$LOG_FILE"
mkdir -p logs

log() {
	# Output to both stdout and log file
	local ts
	ts="$(date '+%Y-%m-%d %H:%M:%S')"
	echo "[$ts] $*" | tee -a "$LOG_FILE"
}

# for edge_size in 0.005 0.01 0.1 1 6 12 24 48; do
for edge_size in 12 24; do
	for edge_ent in 1.0 2.0; do
		cmd=(
			torchrun --nproc_per_node=2 explain.py
			--model-path masters/house_check
			--explain-pos-samples
			--edge-size $edge_size --edge-ent $edge_ent
			--epochs 200 --lr 0.01
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
			log "[COMPLETED] edge_size=${edge_size} edge_ent=${edge_ent} duration=${dur}s"
		else
			log "[ERROR] edge_size=${edge_size} edge_ent=${edge_ent} rc=${rc} duration=${dur}s"
		fi
	done
done
