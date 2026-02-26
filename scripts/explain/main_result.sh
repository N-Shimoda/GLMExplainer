#!/usr/bin/env bash

set -euo pipefail

subsets=(ba_shapes tree_cycle tree_grid_v2 ba_two_motifs)
graph_types=(complete empty empty complete)
epochs=(400 100 200 500)
lrs=(0.03 0.1 0.03 0.1)
edge_sizes=(3 10 10 30)

for i in "${!subsets[@]}"; do
	torchrun --nproc_per_node=2 explain.py \
		--dataset MotifQA --subset "${subsets[$i]}" \
		--model-path naos-ku/GraphTokenLM \
		--target-pos-samples --num-trials 5 \
		--num-gen-trials 10 --min-correct-answers 5 \
		--baseline-graph "${graph_types[$i]}" \
		--llr-threshold 1.0 \
		--epochs "${epochs[$i]}" --lr "${lrs[$i]}" \
		--edge-size "${edge_sizes[$i]}" --edge-ent 1.0 \
		--wandb --tags fpai full
done
