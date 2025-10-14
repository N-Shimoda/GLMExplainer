#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")
LOG_DIR="$ROOT_DIR/logs"
FT_DIR="$ROOT_DIR/fine-tuning"

mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/eval_zs_all.log"
: >"$LOGFILE" # Create empty log file

cd "$FT_DIR"

subsets=(
	node_count
	edge_count
	cycle_check
	triangle_counting
)

for subset in "${subsets[@]}"; do
	echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') ${subset} started" >>"$LOGFILE"
	torchrun --standalone --nproc_per_node=2 eval_ft.py --subset "$subset" \
		--use-pretrained \
		--base-model Qwen/Qwen3-4B-Base \
		--num-trials 10 |
		tee -a "$LOGFILE"
	echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') ${subset} finished" >>"$LOGFILE"
done
