#!/usr/bin/env bash

set -euo pipefail

subsets=(ba_shapes tree_cycle ba_two_motifs)
graph_types=(complete empty complete)
epochs=(500 50 500)
lrs=(0.03 0.1 0.1)
edge_sizes=(3 10 30)

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
		--epochs "${epochs[$i]}" --lr "${lrs[$i]}" \
		--edge-size "${edge_sizes[$i]}" --edge-ent 1.0 \
		--wandb --tags fpai epochs small
done
