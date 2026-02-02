#!/usr/bin/env bash

torchrun --nproc_per_node=2 train.py \
	--dataset MotifQA \
	--subset ba_shapes tree_cycle tree_grid tree_grid_v2 ba_two_motifs \
	--lpe-dim 8 \
	--pos-emb-dim 8 \
	--gnn-type GAT \
	--gnn-hidden-dim 32 --gnn-out-dim 32 \
	--num-gnn-layers 5 \
	--num-graph-tokens 4 \
	--epochs 24 \
	--optim adamw \
	--lr 0.0075 --weight-decay 0.01 \
	--lr-scheduler-type cosine --warmup-ratio 0.05 \
	--do-eval \
	--wandb --tags dev multitask
