#!/bin/bash

set -euo pipefail

subsets=(ba_shapes tree_cycle tree_grid ba_two_motifs)
gnn_types=(GCN GAT GIN)

for gnn in "${gnn_types[@]}"; do
	for subset in "${subsets[@]}"; do
		for num_layer in 3 4 5; do
			torchrun --nproc_per_node=2 train.py \
				--dataset "MotifQA" --subset "${subset}" \
				--num-graph-tokens 4 \
				--lpe-dim 8 --pos-emb-dim 8 \
				--gnn-type "${gnn}" \
				--gnn-hidden-dim 256 --gnn-out-dim 512 \
				--num-gnn-layers "${num_layer}" \
				--epochs 12 \
				--optim "adamw" --lr 0.0075 --weight-decay 0.01 \
				--lr-scheduler-type "cosine" --warmup-ratio 0.05 \
				--do-eval --no-save \
				--wandb --tags gnn_layers
		done
	done
done
