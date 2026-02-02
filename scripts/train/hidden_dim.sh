#!/usr/bin/env bash

set -euo pipefail

subsets=(tree_cycle tree_grid_v2 ba_two_motifs)

for subset in "${subsets[@]}"; do
	for hidden_dim in 32 64 128 256; do
		torchrun --nproc_per_node=2 train.py \
			--dataset "MotifQA" --subset "${subset}" \
			--lpe-dim 8 --pos-emb-dim 8 \
			--gnn-type "GAT" \
			--gnn-hidden-dim "${hidden_dim}" --gnn-out-dim "${hidden_dim}" \
			--num-gnn-layers 5 \
			--graph-pooling "mean" "sum" \
			--num-graph-tokens 4 \
			--epochs 24 \
			--optim "adamw" --lr 0.0075 --weight-decay 0.01 \
			--lr-scheduler-type "cosine" --warmup-ratio 0.05 \
			--do-eval --no-save \
			--wandb --tags "hidden_dim"
	done
done
