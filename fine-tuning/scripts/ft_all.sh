#!/bin/bash

for subset in cycle_check node_count edge_count; do
  python ft_qwen3_4b.py \
    --do_train \
    --subset "$subset" \
    --output_dir "qwen3-4b-$subset" \
    --wandb
done
