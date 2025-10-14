#!/bin/bash

LOGFILE="../logs/eval_zs_all.log"
: >"$LOGFILE" # Create empty log file

for subset in node_count edge_count cycle_check triangle_counting; do
	echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') ${subset} started" >>"$LOGFILE"
	torchrun --standalone --nproc_per_node=2 eval_ft.py --subset "$subset" \
		--use-pretrained \
		--base-model Qwen/Qwen3-4B-Base \
		--num-trials 10 |
		tee -a "$LOGFILE"
	echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') ${subset} finished" >>"$LOGFILE"
done
