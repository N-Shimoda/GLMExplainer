#!/usr/bin/env bash

torchrun --nproc_per_node=2 explain.py \
	--dataset MotifQA --subset tree_cycle \
	--model-path naos-ku/GraphTokenLM \
	--target-pos-samples \
	--num-samples 4 --num-trials 2 \
	--baseline-graph empty --llr-threshold 1.0 \
	--epochs 200 --lr 3.0 \
	--edge-size 96 --edge-ent 1.0 \
	--f1-threshold 0.3 \
	--wandb --tags dev
