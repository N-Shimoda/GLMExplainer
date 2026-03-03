#!/usr/bin/env bash

set -euo pipefail

for subset in ba_shapes tree_cycle tree_grid_v2 ba_two_motifs; do
	for baseline_type in complete empty random; do
		echo -e "\nsubset: $subset, baseline: $baseline_type"
		python tools/llr_study.py \
			--subset $subset \
			--model-path "naos-ku/GraphTokenLM" \
			--num-samples 20 \
			--baseline-graph $baseline_type \
			--llr-threshold 1.0 \
			--verbose --output-format svg
	done
done
