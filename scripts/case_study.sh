#!/bin/bash

for subset in ba_shapes ba_two_motifs tree_cycle tree_grid; do
	for baseline_type in complete empty random; do
		echo -e "\nsubset: $subset, baseline: $baseline_type"
		python tools/case_study.py \
			--subset $subset \
			--model-path outputs/$subset/ --ckpt-index -2 \
			--num-samples 12 \
			--baseline-graph $baseline_type
	done
done
