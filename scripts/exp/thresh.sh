#!/bin/bash

set -euo pipefail

subset=ba_shapes

for edge_size in 1 10 100; do
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
