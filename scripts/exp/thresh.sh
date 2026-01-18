#!/bin/bash

for subset in ba_shapes tree_cycle tree_grid ba_two_motifs shortest_path; do
	for lr in 1.0 3.0; do
		for edge_size in 1e-3 1e-2 1e-1 1 10 100; do
			# Without thresholding
			wo_cmd=(
				torchrun --nproc_per_node=2 explain.py
				--dataset MotifQA --subset "$subset"
				--model-path masters/"$subset"
				--target-pos-samples --num-samples 50
				--num-trials 5
				--epochs 200 --lr "$lr"
				--edge-size "$edge_size" --edge-ent 1.0
				--wandb --tags thresh_search small
			)
			"${wo_cmd[@]}"

			# With thresholding
			w_cmd=(
				"${wo_cmd[@]}"
				--llr-threshold 1.0
				--baseline-graph "complete"
			)
			"${w_cmd[@]}"
		done
	done
done
