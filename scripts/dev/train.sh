#!/bin/bash

torchrun --nproc_per_node=2 train.py \
	--dataset MotifQA --subset ba_two_motifs \
	--gnn-type GAT \
	--lpe-dim 8 \
	--pos-emb-dim 8 \
	--gnn-hidden-dim 128 --gnn-out-dim 256 \
	--num-gnn-layers 3 \
	--num-graph-tokens 4 \
	--graph-pooling mean sum add \
	--epochs 2 \
	--optim adamw \
	--lr 0.0075 --weight-decay 0.01 \
	--lr-scheduler-type cosine --warmup-ratio 0.05 \
	--do-eval \
	--wandb --tags dev
