#!/usr/bin/env bash

set -euo pipefail

subsets=(ba_shapes tree_cycle tree_grid_v2 ba_two_motifs)
jaccard_k_values=(6 6 7 5)
edge_sizes=(3e-1 3e0 3e1 3e2)

for graph_type in complete empty; do
	for i in "${!subsets[@]}"; do
		for edge_size in "${edge_sizes[@]}"; do
			torchrun --nproc_per_node=2 explain.py \
				--dataset MotifQA \
				--subset "${subsets[$i]}" --split validation \
				--model-path masters/multitask \
				--target-pos-samples \
				--num-samples 50 --num-trials 5 \
				--num-gen-trials 10 --min-correct-answers 5 \
				--baseline-graph "${graph_type}" \
				--llr-threshold 1.0 \
				--epoch 200 --lr 0.3 \
				--edge-size "${edge_size}" --edge-ent 1.0 \
				--jaccard-k "${jaccard_k_values[$i]}" \
				--wandb --tags jsai edge_size small
		done
	done
done
