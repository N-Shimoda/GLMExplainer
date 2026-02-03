#!/usr/bin/env bash

set -euo pipefail

# GIN
hidden_dim=64
for num_gnn_layers in 3 5; do
	for graph_pooling in "sum" "mean"; do
		torchrun --nproc_per_node=2 train.py \
			--dataset MotifQA \
			--subset ba_shapes ba_two_motifs tree_cycle tree_grid_v2 \
			--lpe-dim 8 --pos-emb-dim 8 \
			--gnn-type "GIN" \
			--gnn-hidden-dim $hidden_dim --gnn-out-dim $hidden_dim \
			--num-gnn-layers $num_gnn_layers \
			--graph-pooling "$graph_pooling" \
			--num-proj-layers 2 \
			--num-graph-tokens 4 \
			--epochs 32 \
			--optim adamw \
			--lr 5e-3 --weight-decay 1e-2 \
			--lr-scheduler-type cosine --warmup-ratio 0.05 \
			--do-eval --num-eval-trials 5 \
			--wandb --tags multitask
	done
done

# GAT
hidden_dim=64
for num_gnn_layers in 2 3; do
	for graph_pooling in "sum" "mean"; do
		torchrun --nproc_per_node=2 train.py \
			--dataset MotifQA \
			--subset ba_shapes ba_two_motifs tree_cycle tree_grid_v2 \
			--lpe-dim 8 --pos-emb-dim 8 \
			--gnn-type "GAT" \
			--gnn-hidden-dim $hidden_dim --gnn-out-dim $hidden_dim \
			--num-gnn-layers $num_gnn_layers \
			--graph-pooling "$graph_pooling" \
			--num-proj-layers 2 \
			--num-graph-tokens 4 \
			--epochs 32 \
			--optim adamw \
			--lr 5e-3 --weight-decay 1e-2 \
			--lr-scheduler-type cosine --warmup-ratio 0.05 \
			--do-eval --num-eval-trials 5 \
			--wandb --tags multitask
	done
done

# GraphTransformer
for hidden_dim in 32 64; do
	for graph_pooling in "sum" "mean"; do
		torchrun --nproc_per_node=2 train.py \
			--dataset MotifQA \
			--subset ba_shapes tree_cycle tree_grid_v2 ba_two_motifs \
			--lpe-dim 8 --pos-emb-dim 8 \
			--gnn-type "GraphTransformer" \
			--gnn-hidden-dim $hidden_dim --gnn-out-dim $hidden_dim \
			--num-gnn-layers 2 \
			--graph-pooling "$graph_pooling" \
			--num-proj-layers 2 \
			--num-graph-tokens 4 \
			--epochs 32 \
			--optim adamw \
			--lr 5e-3 --weight-decay 1e-2 \
			--lr-scheduler-type cosine --warmup-ratio 0.05 \
			--do-eval --num-eval-trials 5 \
			--wandb --tags multitask
	done
done
