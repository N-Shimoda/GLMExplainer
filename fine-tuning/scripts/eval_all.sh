#!/bin/bash

USE_LOCAL=false
USE_QUICK=false
for arg in "$@"; do
  if [ "$arg" = "--local" ]; then
    USE_LOCAL=true
  fi
  if [ "$arg" = "--quick" ]; then
    USE_QUICK=true
  fi
done

for subset in node_count edge_count cycle_check triangle_counting; do
  cmd=(python eval.py --subset "$subset")
  if [ "$USE_LOCAL" = true ]; then
    # Get the latest model directory
    MODEL_DIR=$(ls -d ./models/$subset/*/ 2>/dev/null | sort | tail -n 1)
    if [ -z "$MODEL_DIR" ]; then
      echo "[WARN] Model directory not found: ./models/$subset/"
      continue
    fi
    cmd+=(--model-path "${MODEL_DIR}checkpoint-final")
  fi
  if [ "$USE_QUICK" = true ]; then
    cmd+=(--quick)
  fi
  # Execute the command
  eval "${cmd[@]}"
  echo
done
