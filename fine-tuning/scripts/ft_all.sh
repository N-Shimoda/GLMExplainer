#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FT_DIR=$(dirname "$SCRIPT_DIR")
cd "$FT_DIR"

LOG_DIR="../logs"
LOG_FILE="$LOG_DIR/ft_all.log"
mkdir -p "$LOG_DIR"
> "$LOG_FILE"

ACCELERATE_BIN=${ACCELERATE_BIN:-accelerate}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-accelerate_config.yaml}

if [ ! -f "$ACCELERATE_CONFIG" ]; then
	echo "Accelerate config '$ACCELERATE_CONFIG' not found. Generate one via 'accelerate config' or use the provided template." >&2
	exit 1
fi

BASE_ARGS=(
	--base-model "Qwen/Qwen3-4B-Instruct-2507"
	--epochs 3
	--do-eval
	--wandb
)

for subset in node_count edge_count cycle_check triangle_counting; do
	echo
	start_time=$(date +%s)
	if "$ACCELERATE_BIN" launch --config_file "$ACCELERATE_CONFIG" ft_qwen3_4b.py "${BASE_ARGS[@]}" --subset "$subset"; then
		status=0
	else
		status=$?
	fi

	end_time=$(date +%s)
	duration=$((end_time - start_time))
	min=$((duration / 60))
	sec=$((duration % 60))
	timestamp=$(date "+%Y-%m-%d %H:%M:%S")

	if [ $status -eq 0 ]; then
		echo "[$timestamp] [SUCCESS] Fine-tuning completed for $subset in ${min} min ${sec} sec" >>"$LOG_FILE"
	else
		echo "[$timestamp] [ERROR] Fine-tuning failed for $subset (exit code $status)" >>"$LOG_FILE"
	fi
done
