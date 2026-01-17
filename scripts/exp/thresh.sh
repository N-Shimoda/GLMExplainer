#!/bin/bash

for subset in tree_grid; do
	for lr in 0.1 1.0 3.0; do
		for edge_size in 1e-3 1e-2 1e-1 1 10 100; do
			for thresh in 0.0 1.0; do
				torchrun --nproc_per_node=2 explain.py \
					--dataset MotifQA --subset $subset \
					--model-path masters/"$subset" \
					--target-pos-samples --num-samples 50 \
					--num-trials 5 \
					--epochs 200 --lr $lr \
					--edge-size $edge_size --edge-ent 1.0 \
					--llr-threshold $thresh --baseline-graph "complete" \
					--wandb --tags thresh_search small
			done
		done
	done
done
