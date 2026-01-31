#!/bin/bash

set -euo pipefail

subsets=(ba_shapes tree_cycle tree_grid ba_two_motifs shortest_path)
gnn_types=(GAT GIN GCN)
graph_poolings=("mean sum" "mean sum max")

for gnn in "${gnn_types[@]}"; do
	for subset in "${subsets[@]}"; do
		for num_layer in 3 4 5; do
			for graph_pooling in "${graph_poolings[@]}"; do
				read -r -a pooling_args <<<"${graph_pooling}"
				torchrun --nproc_per_node=2 train.py \
					--dataset "MotifQA" --subset "${subset}" \
					--gnn-type "${gnn}" \
					--lpe-dim 8 --pos-emb-dim 8 \
					--gnn-hidden-dim 256 --gnn-out-dim 512 \
					--num-gnn-layers "${num_layer}" \
					--graph-pooling "${pooling_args[@]}" \
					--num-graph-tokens 4 \
					--epochs 24 \
					--optim "adamw" --lr 0.0075 --weight-decay 0.01 \
					--lr-scheduler-type "cosine" --warmup-ratio 0.05 \
					--do-eval --no-save \
					--wandb --tags gnn_layers
			done
		done
	done
done
