#!/usr/bin/env bash

set -euo pipefail

for baseline_type in complete empty random; do
	python tools/llr_study.py \
		--subset ba_shapes \
		--model-path masters/multitask \
		--num-samples 20 \
		--baseline-graph $baseline_type \
		--verbose --output-format svg
done
