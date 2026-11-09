#!/usr/bin/env bash
# seed を変えて学習し直した 1 の解析を、GPU 3・4 で流す（54|）。
# 3 つの指標と獲得の順番（16|・17|）、1 で変わった重みを部品ごとに移す分析（33|・34|）。
# 2 の学習し直し（run_second_sense_replication.sh）は GPU 3・4 で run-4・run-5 を学習し、1 枚で約 75 GB
# 使う。重みを移す分析（約 21 GB）を重ねるとメモリが足りなくなるおそれがあるので、その 2 run の学習が
# 終わってから始める（2 の解析とは、GPU を分け合ってよい）。
# 使い方（リポジトリの最上位のディレクトリで）:
#   setsid nohup bash experiments/trial-1b-base-sequences-5632-k-1000/run_unknown_word_replication_analysis.sh &
set -u
E=experiments/trial-1b-base-sequences-5632-k-1000
L=outputs/trial-1b-base-sequences-5632-k-1000/logs/unknown-word-acquisition-replication
SECOND=outputs/checkpoints/second-sense-addition-both-phases-replication/OLMo-2-0425-1B
RUNS=(run-1 run-2 run-3 run-4 run-5 run-6 run-7 run-8)
export PYTHONUNBUFFERED=1
export FIRST_PHASE_EXPERIMENT=unknown-word-acquisition-replication
export SECOND_SENSE_EXPERIMENT=second-sense-addition-both-phases-replication
mkdir -p "$L"
note() { echo "=== $(date '+%F %T') $*" >> "$L/analysis.log"; }

# 引数: GPU の番号、ログの名前、コマンド
job() {
  local gpu=$1 name=$2
  shift 2
  note "開始: $name（GPU $gpu）"
  if CUDA_VISIBLE_DEVICES=$gpu uv run python "$@" >> "$L/$name.log" 2>&1; then
    note "終了: $name"
  else
    note "止まりました: $name"
  fi
}

# 引数: GPU の番号、その GPU で学習する 2 の run
wait_for_training() {
  while [ ! -e "$SECOND/$2/completed.json" ]; do sleep 60; done
  while [ "$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits)" -ge 50000 ]; do
    sleep 60
  done
}

(
  wait_for_training 3 run-4
  for run in "${RUNS[@]:0:4}"; do
    job 3 "grafting_first_$run" "$E/measure_second_sense_weight_grafting.py" measure first "$run"
  done
  for run in "${RUNS[@]:0:4}"; do
    job 3 "grafting_indicators_$run" "$E/measure_unknown_word_weight_grafting.py" measure "$run"
  done
) &
(
  wait_for_training 4 run-5
  job 4 indicators "$E/measure_unknown_word_replication.py" indicators
  job 4 order "$E/measure_unknown_word_replication.py" order
  for run in "${RUNS[@]:4:4}"; do
    job 4 "grafting_first_$run" "$E/measure_second_sense_weight_grafting.py" measure first "$run"
  done
  for run in "${RUNS[@]:4:4}"; do
    job 4 "grafting_indicators_$run" "$E/measure_unknown_word_weight_grafting.py" measure "$run"
  done
) &
wait
uv run python "$E/measure_second_sense_weight_grafting.py" summarize first >> "$L/summarize.log" 2>&1
uv run python "$E/measure_unknown_word_weight_grafting.py" summarize >> "$L/summarize.log" 2>&1
note "全て終わりました: 1 の解析"
