#!/bin/bash

set -euo pipefail

edge_sizes=(0.1 0.3 3 30)

subset=tree_grid
edge_size=30
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

# Explore thresh = 0.0 vs 1.0
subset=ba_two_motifs
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
