#!/usr/bin/env bash

for gnn in GAT GIN GraphTransformer; do
	for hidden_dim in 32 64 128; do
		torchrun --nproc_per_node=2 train.py \
			--dataset MotifQA \
			--subset ba_shapes ba_two_motifs tree_cycle tree_grid_v2 \
			--lpe-dim 8 --pos-emb-dim 8 \
			--gnn-type $gnn \
			--gnn-hidden-dim $hidden_dim --gnn-out-dim $hidden_dim \
			--num-gnn-layers 5 \
			--num-proj-layers 2 \
			--num-graph-tokens 4 \
			--epochs 24 \
			--optim adamw \
			--lr 0.0075 --weight-decay 0.01 \
			--lr-scheduler-type cosine --warmup-ratio 0.05 \
			--do-eval \
			--wandb --tags multitask hidden_dim
	done
done
