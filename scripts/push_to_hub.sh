#!/usr/bin/env bash

set -euo pipefail

export_dir="GraphTokenLM-HF"

python tools/export_to_hf.py \
	--ckpt-path masters/multitask \
	--export-dir ${export_dir}
cp src/glm.py ${export_dir}/glm.py
