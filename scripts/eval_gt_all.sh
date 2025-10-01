#!/usr/bin/env bash
set -euo pipefail

# Move to repository root (assuming this script is directly under scripts/)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/eval_gt_all.log"

# Remove existing log file
rm -f "$LOG_FILE"

log() {
  # Output to both stdout and log file
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[$ts] $*" | tee -a "$LOG_FILE"
}

# -----------------------------------------
# Argument parsing
# Accepts --split {test|train|validation} and passes to eval.py
# Default is test
# -----------------------------------------
SPLIT="test"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --split)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --split に値が必要です (test|train|validation)" >&2
        exit 2
      fi
      SPLIT="$2"
      shift 2
      ;;
    -h|--help)
      cat <<USAGE
Usage: $(basename "$0") [--split {test|train|validation}]

Options:
  --split   Which dataset split to evaluate (default: test)
USAGE
      exit 0
      ;;
    *)
      echo "ERROR: 未知の引数: $1" >&2
      exit 2
      ;;
  esac
done

# Validate split value
case "$SPLIT" in
  test|train|validation) ;;
  *)
    echo "ERROR: --split should be chosen from test|train|validation (got: $SPLIT)" >&2
    exit 2
    ;;
esac

subsets=(
  node_count
  edge_count
  cycle_check
  triangle_counting
  maximum_flow
)

for subset in "${subsets[@]}"; do
  cmd=(
    python eval.py --subset "${subset}"
    --model-path "outputs/${subset}"
    --split "${SPLIT}"
    --num-trials 5
  )
  log "[START] ${cmd[*]}"
  start_ts=$(date +%s)

  tmp_output_file="$(mktemp)"
  set +e
  "${cmd[@]}" 2>&1 | tee "$tmp_output_file"
  rc=${PIPESTATUS[0]}
  set -e

  end_ts=$(date +%s)
  dur=$(( end_ts - start_ts ))

  summary_line="$(grep -E '\[SUMMARY\]' "$tmp_output_file" | tail -n 1 || true)"
  rm -f "$tmp_output_file"

  if [[ $rc -eq 0 ]]; then
    acc_note=""
    if [[ -n "$summary_line" ]]; then
      acc_value="$(sed -n 's/.*accuracy=\([0-9.][0-9.]*\).*/\1/p' <<<"$summary_line")"
      if [[ -n "$acc_value" ]]; then
        acc_percent="$(awk -v acc="$acc_value" 'BEGIN { printf "%.2f", acc * 100 }')"
        acc_note=" accuracy=${acc_percent}% (raw=${acc_value})"
        log "[INFO] subset=${subset}${acc_note}"
      else
        log "[WARNING] subset=${subset} accuracy value not found in summary output."
      fi
    else
      log "[WARNING] subset=${subset} summary line not found in eval output."
    fi
    log "[COMPLETED] subset=${subset} duration=${dur}s${acc_note}"
  else
    log "[ERROR] subset=${subset} rc=${rc} duration=${dur}s"
    exit $rc
  fi
  echo

done

log "[INFO] ALL subsets finished successfully."
