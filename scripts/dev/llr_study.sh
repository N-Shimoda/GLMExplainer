#!/usr/bin/env bash

set -euo pipefail

python tools/llr_study.py \
	--subset ba_shapes \
	--model-path naos-ku/GraphTokenLM \
	--num-samples 20 \
	--baseline-graph complete \
	--llr-threshold 1.0 \
	--verbose --output-format svg
