#!/bin/bash

LOGFILE="../logs/eval_ft_all.log"
: >"$LOGFILE" # Create empty log file

for subset in node_count edge_count cycle_check triangle_counting; do
	echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') ${subset} started" >>"$LOGFILE"
	python eval.py --subset "$subset" --model-path "models/$subset" --num-trials 10 | tee -a "$LOGFILE"
	echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') ${subset} finished" >>"$LOGFILE"
done
