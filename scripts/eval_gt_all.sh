#!/usr/bin/env bash
set -euo pipefail

# リポジトリルートへ移動（このスクリプトの場所が scripts/ の直下である前提）
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_gt_all.log"

# 既存のログを削除
rm -f "$LOG_FILE"

log() {
  # 両方へ出力（標準出力 + ログファイル）
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[$ts] $*" | tee -a "$LOG_FILE"
}

# -----------------------------------------
# 引数処理
# --split {test|train|validation} を受け取り、eval.py に渡す
# デフォルトは test
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

# 値検証
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
  cmd=(python eval.py --subset "${subset}" --model_path "outputs/${subset}" --split "${SPLIT}")
  log "[START] ${cmd[*]}"
  start_ts=$(date +%s)

  set +e
  "${cmd[@]}"
  rc=$?
  set -e

  end_ts=$(date +%s)
  dur=$(( end_ts - start_ts ))

  if [[ $rc -eq 0 ]]; then
    log "[COMPLETED] subset=${subset} duration=${dur}s"
  else
    log "[ERROR] subset=${subset} rc=${rc} duration=${dur}s"
    exit $rc
  fi
  echo

done

log "[INFO] ALL subsets finished successfully."