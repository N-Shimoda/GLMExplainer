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

subsets=(
  node_count
  edge_count
  cycle_check
  triangle_counting
  maximum_flow
)

for subset in "${subsets[@]}"; do
  cmd=(torchrun --nproc_per_node=2 train.py --subset "${subset}" --do_eval --wandb)
  log "[START] subset=${subset} cmd: ${cmd[*]}"
  start_ts=$(date +%s)

  # torchrun を実行して終了コードを取得（set -e の影響を避ける）
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