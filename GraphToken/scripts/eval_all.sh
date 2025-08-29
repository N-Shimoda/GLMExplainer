#!/bin/bash

for subset in cycle_check node_count edge_count; do
  python eval.py \
    --subset "$subset" \
    # --model_path "./qwen3-4b-$subset/checkpoint-final"
  echo
done
