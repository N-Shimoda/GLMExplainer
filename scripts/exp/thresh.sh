#!/bin/bash

set -euo pipefail

for subset in ba_shapes tree_cycle tree_grid ba_two_motifs shortest_path; do
	for edge_size in 1e-2 1e-1 1 10 100; do
		# Without thresholding
		cmd_v1=(
			torchrun --nproc_per_node=2 explain.py
			--dataset MotifQA --subset "$subset"
			--model-path masters/"$subset"
			--target-pos-samples --num-samples 50
			--num-trials 5
			--epochs 200 --lr 3.0
			--edge-size "$edge_size" --edge-ent 1.0
			--wandb
			--tags thresh_search master small
		)
		"${cmd_v1[@]}"

		# With thresholding
		cmd_v2=(
			"${cmd_v1[@]}"
			--llr-threshold 1.0
			--baseline-graph "complete"
		)
		"${cmd_v2[@]}"
	done
done
