#!/bin/bash

torchrun --nproc_per_node=2 train.py \
	--dataset MotifQA --subset ba_shapes \
	--lpe-dim 8 --use-degree-emb \
	--pos-emb-dim 8 \
	--gnn-type GCN \
	--gnn-hidden-dim 16 --gnn-out-dim 32 \
	--num-gnn-layers 3 \
	--num-graph-tokens 4 \
	--epochs 2 \
	--optim adamw \
	--lr 0.0075 --weight-decay 0.01 \
	--lr-scheduler-type cosine --warmup-ratio 0.05 \
	--do-eval \
	--no-save \
	--wandb --tags dev
