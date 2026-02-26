#!/usr/bin/env bash

set -euo pipefail

subsets=(ba_shapes tree_cycle ba_two_motifs)
graph_types=(complete empty complete)
edge_sizes=(3 10 10 30)
lr=(500 50 500)

# Learning Rate
for i in "${!subsets[@]}"; do
	torchrun --nproc_per_node=2 explain.py \
		--dataset MotifQA \
		--subset "${subsets[$i]}" --split validation \
		--model-path naos-ku/GraphTokenLM \
		--target-pos-samples \
		--num-samples 50 --num-trials 5 \
		--num-gen-trials 10 --min-correct-answers 5 \
		--baseline-graph "${graph_types[$i]}" \
		--llr-threshold 1.0 \
		--epochs 200 --lr "${lr[$i]}" \
		--edge-size "${edge_sizes[$i]}" --edge-ent 1.0 \
		--wandb --tags fpai lr small
done
