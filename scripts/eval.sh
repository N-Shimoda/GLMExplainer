#!/usr/bin/env bash

torchrun --nproc_per_node=2 eval.py \
	--dataset MotifQA --subset ba_shapes tree_cycle tree_grid_v2 ba_two_motifs \
	--model-path naos-ku/GraphTokenLM \
	--num-trials 5 --per-device-batch-size 5
