#!/usr/bin/env bash

set -euo pipefail

subsets=(ba_shapes tree_cycle tree_grid_v2 ba_two_motifs)
graph_types=(empty empty empty empty)
edge_sizes=(30 10 10 30)
jaccard_k_values=(6 6 7 5)

# Epochs
for i in "${!subsets[@]}"; do
	for epochs in 100 300 400; do
		torchrun --nproc_per_node=2 explain.py \
			--dataset MotifQA \
			--subset "${subsets[$i]}" --split validation \
			--model-path masters/multitask \
			--target-pos-samples \
			--num-samples 50 --num-trials 5 \
			--num-gen-trials 10 --min-correct-answers 5 \
			--baseline-graph "${graph_types[$i]}" \
			--llr-threshold 1.0 \
			--epochs "${epochs}" --lr 0.3 \
			--edge-size "${edge_sizes[$i]}" --edge-ent 1.0 \
			--jaccard-k "${jaccard_k_values[$i]}" \
			--wandb --tags epochs small
	done
done

# Learning Rate
for i in "${!subsets[@]}"; do
	for lr in 1e-2 3e-2 1e-1 1e0; do
		torchrun --nproc_per_node=2 explain.py \
			--dataset MotifQA \
			--subset "${subsets[$i]}" --split validation \
			--model-path masters/multitask \
			--target-pos-samples \
			--num-samples 50 --num-trials 5 \
			--num-gen-trials 10 --min-correct-answers 5 \
			--baseline-graph "${graph_types[$i]}" \
			--llr-threshold 1.0 \
			--epochs 200 --lr "${lr}" \
			--edge-size "${edge_sizes[$i]}" --edge-ent 1.0 \
			--jaccard-k "${jaccard_k_values[$i]}" \
			--wandb --tags lr small
	done
done
