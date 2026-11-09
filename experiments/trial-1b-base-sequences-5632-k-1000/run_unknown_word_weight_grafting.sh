#!/usr/bin/env bash
# 34|: 未知語が既知語になるフェーズで変わった重みを部品ごとに移し、未知語の 3 つの指標を測る（GPU）。
# GPU ごとに、その GPU が空く（使用中のメモリが 2 GB 未満になる）のを待ってから、run を順に測る。
# 測り終えた run は飛ばす。全て終わったら表を書く。
# 使い方（リポジトリの最上位のディレクトリで。引数は使う GPU の番号）:
#   setsid nohup bash experiments/trial-1b-base-sequences-5632-k-1000/run_unknown_word_weight_grafting.sh "3 4" &
# 環境変数 FIRST_PHASE_EXPERIMENT を渡すと、その学習の重みを読む（measure_unknown_word_weight_grafting.py）。
set -u
read -r -a GPUS <<< "${1:-0 1 2 3 4}"
EXPERIMENT=${FIRST_PHASE_EXPERIMENT:-unknown-word-acquisition}
E=experiments/trial-1b-base-sequences-5632-k-1000
L=outputs/trial-1b-base-sequences-5632-k-1000/logs/unknown-word-weight-grafting/$EXPERIMENT
R=outputs/partial-results/unknown-word-weight-grafting
[ "$EXPERIMENT" = unknown-word-acquisition ] || R=$R/$EXPERIMENT
export PYTHONUNBUFFERED=1
mkdir -p "$L"
note() { echo "=== $(date '+%F %T') $*" >> "$L/grafting.log"; }

wait_for_gpu() {
  while [ "$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits)" -ge 2000 ]; do
    sleep 60
  done
}

measure() {  # GPU run
  local gpu=$1 run=$2
  [ -e "$R/$run.parquet" ] && return
  wait_for_gpu "$gpu"
  note "開始: $run（GPU $gpu）"
  if CUDA_VISIBLE_DEVICES=$gpu uv run python "$E/measure_unknown_word_weight_grafting.py" measure "$run" \
    >> "$L/$run.log" 2>&1; then
    note "終了: $run"
  else
    note "止まりました: $run"
  fi
}

runs=(run-1 run-2 run-3 run-4 run-5 run-6 run-7 run-8)
for slot in "${!GPUS[@]}"; do
  (
    for ((index = slot; index < ${#runs[@]}; index += ${#GPUS[@]})); do
      measure "${GPUS[$slot]}" "${runs[$index]}"
    done
  ) &
done
wait
uv run python "$E/measure_unknown_word_weight_grafting.py" summarize >> "$L/summarize.log" 2>&1
note "全て終わりました: $EXPERIMENT"
