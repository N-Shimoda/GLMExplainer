#!/bin/bash

set -euo pipefail

subset=ba_shapes

for thresh in None 1.0; do
	torchrun --nproc_per_node=2 explain.py \
		--dataset MotifQA --subset "$subset" \
		--model-path masters/"$subset" \
		--target-pos-samples \
		--num-samples 50 --num-trials 5 \
		--llr-threshold "$thresh" \
		--epochs 200 --lr 0.3 \
		--edge-size 100 --edge-ent 1.0 \
		--wandb --tags masters small
done
