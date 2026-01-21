#!/bin/bash

torchrun --nproc_per_node=2 train.py \
	--dataset MotifQA --subset tree_grid \
	--lpe-dim 8 \
	--pos-emb-dim 8 \
	--gnn-type GCN \
	--gnn-hidden-dim 16 --gnn-out-dim 32 \
	--num-gnn-layers 3 \
	--num-graph-tokens 4 \
	--epochs 24 \
	--optim adamw \
	--lr 0.0075 --weight-decay 0.01 \
	--lr-scheduler-type cosine --warmup-ratio 0.05 \
	--do-eval \
	--wandb --tags masters
