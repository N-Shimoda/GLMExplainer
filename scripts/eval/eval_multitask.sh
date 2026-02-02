#!/usr/bin/env bash

torchrun --standalone --nproc_per_node=2 eval_multitask.py \
	--model-path outputs/multitask --num-trials 10
