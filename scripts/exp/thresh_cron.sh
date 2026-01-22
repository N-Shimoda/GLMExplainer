#!/bin/bash
set -euo pipefail

# Move to project root
cd /home/naoki/github/GraphToken

edge_sizes=(3e3 1e4 3e4 1e5)
subset=tree_grid

for edge_size in "${edge_sizes[@]}"; do
	for thresh in 0.0 1.0; do
		/home/naoki/anaconda3/condabin/conda run -n graphtoken --no-capture-output \
			torchrun --nproc_per_node=2 explain.py \
			--dataset MotifQA --subset "$subset" \
			--model-path "masters/$subset" \
			--target-pos-samples \
			--num-samples 50 --num-trials 5 \
			--llr-threshold "$thresh" \
			--epochs 200 --lr 0.3 \
			--edge-size "$edge_size" --edge-ent 1.0 \
			--wandb --tags master small
	done
done
