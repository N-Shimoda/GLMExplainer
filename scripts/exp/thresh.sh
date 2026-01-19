#!/bin/bash

set -euo pipefail

edge_sizes=(1.0 10 100)

# Explore thresh = 0.0 vs 1.0
for subset in tree_cycle tree_grid ba_two_motifs; do
	for edge_size in "${edge_sizes[@]}"; do
		for thresh in 0.0 1.0; do
			torchrun --nproc_per_node=2 explain.py \
				--dataset MotifQA --subset "$subset" \
				--model-path masters/"$subset" \
				--target-pos-samples \
				--num-samples 50 --num-trials 5 \
				--llr-threshold "$thresh" \
				--epochs 200 --lr 0.3 \
				--edge-size "$edge_size" --edge-ent 1.0 \
				--wandb --tags master small
		done
	done
done

# Shortest path
subset=shortest_path
for edge_size in "${edge_sizes[@]}"; do
	for thresh in 0.0 0.1; do
		torchrun --nproc_per_node=2 explain.py \
			--dataset MotifQA --subset "$subset" \
			--model-path masters/"$subset" \
			--target-pos-samples \
			--num-samples 50 --num-trials 5 \
			--llr-threshold "$thresh" \
			--epochs 200 --lr 0.3 \
			--edge-size "$edge_size" --edge-ent 1.0 \
			--wandb --tags master small
	done
done
