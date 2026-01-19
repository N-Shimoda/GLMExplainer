#!/bin/bash

torchrun --nproc_per_node=2 explain.py \
	--dataset MotifQA --subset ba_shapes \
	--model-path masters/ba_shapes \
	--target-pos-samples \
	--num-samples 50 --num-trials 5 \
	--llr-threshold 3.0 \
	--epochs 200 --lr 3.0 \
	--edge-size 96 --edge-ent 1.0 \
	--wandb --tags dev
