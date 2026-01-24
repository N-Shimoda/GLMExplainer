#!/bin/bash

set -euo pipefail
cd /home/naoki/github/GraphToken

subsets=(ba_shapes tree_cycle tree_grid ba_two_motifs shortest_path)
edge_sizes=(10 30 300 30 30)
llr_thresholds=(1.0 1.0 1.0 1.0 0.1)

for i in "${!subsets[@]}"; do
	subset="${subsets[$i]}"
	edge_size="${edge_sizes[$i]}"
	thresh="${llr_thresholds[$i]}"

	/home/naoki/anaconda3/condabin/conda run -n graphtoken --no-capture-output \
		torchrun --nproc_per_node=2 explain.py \
		--dataset MotifQA --subset "$subset" \
		--model-path "masters/$subset" \
		--target-pos-samples --num-trials 5 \
		--llr-threshold "$thresh" \
		--epochs 200 --lr 0.3 \
		--edge-size "$edge_size" --edge-ent 1.0 \
		--wandb --tags master full
done
