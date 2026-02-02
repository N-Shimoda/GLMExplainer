#!/usr/bin/env bash

for subset in ba_shapes tree_cycle tree_grid ba_two_motifs shortest_path; do
	for epochs in 12 24; do
		for gnn_dim in 8 16 32 64 128 256 512; do
			torchrun --nproc_per_node=2 train.py \
				--dataset MotifQA --subset $subset \
				--gnn-type GCN \
				--lpe-dim 8 --use-degree-emb \
				--pos-emb-dim 8 \
				--gnn-hidden-dim $gnn_dim --gnn-out-dim $((gnn_dim * 2)) \
				--num-gnn-layers 3 \
				--num-graph-tokens 4 \
				--epochs $epochs \
				--optim adamw \
				--lr 0.0075 --weight-decay 0.01 \
				--lr-scheduler-type cosine --warmup-ratio 0.05 \
				--do-eval --no-save \
				--wandb --tags hidden_dim
		done
	done
done
