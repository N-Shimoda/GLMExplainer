#!/bin/bash

mkdir -p ../logs
> ../logs/ft_all.log

for subset in node_count edge_count cycle_check triangle_counting; do
  echo
  start_time=$(date +%s)
  python ft_qwen3_4b.py \
    --base-model "Qwen/Qwen3-4B-Base" \
    --subset "$subset" \
    --epochs 3 \
    --wandb --do-eval
  status=$?
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  min=$((duration / 60))
  sec=$((duration % 60))
  if [ $status -eq 0 ]; then
    echo "[`date "+%Y-%m-%d %H:%M:%S"`] [SUCCESS] Fine-tuning completed for $subset in ${min} min ${sec} seconds" >> ../logs/ft_all.log
  else
    echo "[`date "+%Y-%m-%d %H:%M:%S"`] [ERROR] Fine-tuning failed for $subset (exit code $status)" >> ../logs/ft_all.log
  fi
done
