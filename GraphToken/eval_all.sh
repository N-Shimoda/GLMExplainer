#!/bin/bash

# 各 subset に対して ft_qwen3_4b.py を実行
for subset in cycle_check node_count edge_count; do
  python ft_qwen3_4b.py \
    --do_eval \
    --subset "$subset" \
    # --output_dir "qwern3_4b-$subset"
done
