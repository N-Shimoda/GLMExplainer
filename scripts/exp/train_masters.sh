#!/usr/bin/env bash

set -euo pipefail

subsets=(ba_shapes tree_cycle tree_grid ba_two_motifs shortest_path)
hidden_dims=(16 32 16 256 32)
use_degree_embs=(false true false true true)

for i in "${!subsets[@]}"; do
	subset="${subsets[$i]}"
	gnn_dim="${hidden_dims[$i]}"
	use_degree_emb="${use_degree_embs[$i]}"
	torchrun --nproc_per_node=2 train.py \
		--dataset MotifQA --subset "$subset" \
		--gnn-type GCN \
		--lpe-dim 8 --use-degree-emb "$use_degree_emb" \
		--pos-emb-dim 8 \
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
