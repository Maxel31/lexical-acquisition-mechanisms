#!/usr/bin/env bash
# 課題1の選択肢の確率の保存（save_sense_choice_probabilities.py）を、GPU 0〜4 で並べて流す。
# GPU 0〜3: 語義の追加の11の run の、学習の 25・50・75・100% の重み
# GPU 4: 学習前の 7B と 1B で、空欄にターゲット語・非語・donor を入れた課題1
# 使い方（リポジトリの最上位のディレクトリで）:
#   setsid nohup bash experiments/trial-1b-base-sequences-5632-k-1000/run_sense_choice_probabilities.sh &
set -u
SCRIPT=experiments/trial-1b-base-sequences-5632-k-1000/save_sense_choice_probabilities.py
LOGS=outputs/trial-1b-base-sequences-5632-k-1000/logs/sense-choice-probabilities
mkdir -p "$LOGS"
run() {
  local gpu=$1 name=$2
  shift 2
  echo "=== $(date '+%F %T') 開始: $name（GPU $gpu）" >> "$LOGS/progress.log"
  if CUDA_VISIBLE_DEVICES=$gpu uv run python "$SCRIPT" "$@" > "$LOGS/$name.log" 2>&1; then
    echo "=== $(date '+%F %T') 終了（0）: $name" >> "$LOGS/progress.log"
  else
    echo "=== $(date '+%F %T') 止まりました: $name（$LOGS/$name.log）" >> "$LOGS/progress.log"
  fi
}
run 0 checkpoint-25 checkpoints OLMo-2-0425-1B checkpoint-25 &
run 1 checkpoint-50 checkpoints OLMo-2-0425-1B checkpoint-50 &
run 2 checkpoint-75 checkpoints OLMo-2-0425-1B checkpoint-75 &
run 3 final checkpoints OLMo-2-0425-1B final &
(
  run 4 context-7B context OLMo-2-1124-7B allenai/OLMo-2-1124-7B c0371f4281bf2376207646c6b62ddc6c442c7577
  run 4 context-1B context OLMo-2-0425-1B allenai/OLMo-2-0425-1B 9d3e43659f00c17e6da23cf32333afd1fc39fa1a
) &
wait
echo "=== $(date '+%F %T') 全て終わりました" >> "$LOGS/progress.log"
