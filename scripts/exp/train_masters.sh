#!/bin/bash

set -euo pipefail

subsets=(ba_shapes tree_cycle tree_grid ba_two_motifs shortest_path)
hidden_dims=(16 32 8 256 32)

for i in "${!subsets[@]}"; do
	subset="${subsets[$i]}"
	gnn_dim="${hidden_dims[$i]}"
	torchrun --nproc_per_node=2 train.py \
		--dataset MotifQA --subset "$subset" \
		--lpe-dim 8 --use-degree-emb \
		--pos-emb-dim 8 \
		--gnn-type GCN \
		--gnn-hidden-dim "$gnn_dim" --gnn-out-dim $((gnn_dim * 2)) \
		--num-gnn-layers 3 \
		--num-graph-tokens 4 \
		--epochs 24 \
		--optim adamw \
		--lr 0.0075 --weight-decay 0.01 \
		--lr-scheduler-type cosine --warmup-ratio 0.05 \
		--do-eval \
		--output-dir masters \
		--wandb --tags masters
done
