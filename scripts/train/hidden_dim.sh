#!/bin/bash

set -euo pipefail

subsets=(ba_shapes tree_cycle tree_grid ba_two_motifs shortest_path)
graph_pooling_types=(sum mean)

for subset in "${subsets[@]}"; do
	for hidden_dim in 32 64 128 256; do
		for graph_pooling in "${graph_pooling_types[@]}"; do
			torchrun --nproc_per_node=2 train.py \
				--dataset "MotifQA" --subset "${subset}" \
				--lpe-dim 8 --pos-emb-dim 8 \
				--gnn-type "GAT" \
				--gnn-hidden-dim "${hidden_dim}" --gnn-out-dim "${hidden_dim}" \
				--num-gnn-layers 5 \
				--graph-pooling "${graph_pooling}" \
				--num-graph-tokens 4 \
				--epochs 24 \
				--optim "adamw" --lr 0.0075 --weight-decay 0.01 \
				--lr-scheduler-type "cosine" --warmup-ratio 0.05 \
				--do-eval --no-save \
				--wandb --tags hidden_dim
		done
	done
done
