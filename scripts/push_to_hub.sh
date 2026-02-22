#!/usr/bin/env bash

set -euo pipefail

# Prepare export directory
git clone https://huggingface.co/naos-ku/GraphTokenLM
export_dir="GraphTokenLM"
mkdir -p ${export_dir}

# Export model and config
python tools/export_to_hf.py \
	--ckpt-path masters/multitask \
	--export-dir ${export_dir}
cp src/glm.py ${export_dir}/glm.py

# Push to Hugging Face Hub
cd ${export_dir}
git lfs install
git lfs track "*.bin" "*.safetensors" "tokenizer.json"
git add .
git commit -m "Export GraphTokenLM to Hugging Face Hub"
git push origin main
