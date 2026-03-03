#!/usr/bin/env bash

set -euo pipefail

# Prepare export directory
export_dir="GraphTokenLM"
if [ -d "${export_dir}" ]; then
	echo "[INFO] Export directory ${export_dir} already exists. Pulling latest changes..."
	cd ${export_dir}
	git switch main
	git pull
	cd ..
else
	git clone https://huggingface.co/naos-ku/GraphTokenLM
fi

date_str=$(date +"%Y-%m-%d %H:%M:%S")

# Export model and config
echo
echo "[INFO] Step 1/2: Exporting model and config to ./${export_dir} ..."
export CUDA_VISIBLE_DEVICES=""
python tools/hf/export_to_hf.py \
	--ckpt-path masters/multitask \
	--export-dir ${export_dir}
cp src/glm.py ${export_dir}/glm.py

# Push to Hugging Face Hub
echo
echo "[INFO] Step 2/2: Pushing to Hugging Face Hub..."
cd ${export_dir}
git lfs install
git lfs track "*.bin" "*.safetensors" "tokenizer.json"

if [ "$(git rev-list --count '@{u}..HEAD' 2>/dev/null)" -eq 0 ]; then
	echo "[INFO] Current git status: $(git status --short)"
	exit 0
else
	echo "[INFO] Changes detected. Committing and pushing to Hugging Face Hub..."
	git add -A
	git commit -m "Update GraphTokenLM export on ${date_str}"
	git push
fi
