#!/bin/bash

# --local フラグの有無で挙動を分岐
USE_LOCAL=false
for arg in "$@"; do
  if [ "$arg" = "--local" ]; then
    USE_LOCAL=true
    break
  fi
done

for subset in cycle_check node_count edge_count; do
  if [ "$USE_LOCAL" = true ]; then
    python eval.py \
      --subset "$subset" \
      --model_path "./qwen3-4b-$subset/checkpoint-final"
  else
    python eval.py \
      --subset "$subset"
  fi
  echo
done
