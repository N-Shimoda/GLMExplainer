#!/bin/bash

set -euo pipefail

for subset in ba_shapes tree_cycle tree_grid ba_two_motifs shortest_path; do
	for edge_size in 1e-2 1e-1 1 10 100; do
		for thresh in 0.0 1.0; do
			torchrun --nproc_per_node=2 explain.py \
				--dataset MotifQA --subset "$subset" \
				--model-path masters/"$subset" \
				--target-pos-samples --num-samples 50 \
				--num-trials 5 \
				--epochs 200 --lr 3.0 \
				--edge-size "$edge_size" --edge-ent 1.0 \
				--llr-threshold "$thresh" \
				--baseline-graph "complete" \
				--wandb \
				--tags thresh_search small master
		done
	done
done
