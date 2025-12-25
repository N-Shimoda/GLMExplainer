LOG_FILE="logs/explain_dev.log"
rm -f "$LOG_FILE"
mkdir -p logs

log() {
	local ts
	ts="$(date '+%Y-%m-%d %H:%M:%S')"
	echo "[$ts] $*" | tee -a "$LOG_FILE"
}

motifqa_subsets=(
	ba_shapes
	tree_cycle
	tree_grid
	ba_two_motifs
)

run_explain_loop() {
	local subset="$1"

	log "[INFO] Starting explanation for subset='${subset}'"
	cmd=(
		torchrun --nproc_per_node=2 explain.py
		--dataset "MotifQA" --subset "${subset}"
		--model-path "outputs/${subset}"
		--explain-pos-samples
		--num-samples 8 --num-trials 5
		--epochs 200 --lr 0.1
		--edge-size 96 --edge-ent 1.0
		# --wandb
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
	log "[END] ${cmd[*]} (rc=$rc, duration=${dur}s)"
}

for subset in "${motifqa_subsets[@]}"; do
	run_explain_loop "$subset"
done
