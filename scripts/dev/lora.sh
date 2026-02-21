#!/usr/bin/env bash

torchrun --nproc_per_node=2 train.py \
	--dataset MotifQA --subset ba_shapes \
	--use-lora \
	--lora-r 16 \
	--lora-alpha 16 \
	--lora-dropout 0.05 \
	--lora-target-modules q_proj,k_proj,v_proj,o_proj \
	--epochs 3 \
	--optim adamw --lr 1e-4 --weight-decay 0.01 \
	--lr-scheduler-type cosine --warmup-ratio 0.05 \
	--do-eval --wandb
