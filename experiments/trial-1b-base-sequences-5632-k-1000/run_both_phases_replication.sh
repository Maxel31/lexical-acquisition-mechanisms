#!/usr/bin/env bash
# 1 を seed を変えて学習し直した後の作業を、まとめて流す（43| の続き。GPU 0〜4）。
# 1. 2 の学習の設定を作り（build_both_phases_replication.py）、2 の学習と解析をやり直す
#    （run_second_sense_replication.sh。GPU 0〜4）
# 2. 1 と同時に、1 の解析をやり直す（GPU 3・4。その GPU から 2 の学習が抜けるのを待つ）:
#    3 つの指標と獲得の順番（16|・17|）、1 で変わった重みを部品ごとに移す分析（33|・34|）
# 先に run_unknown_word_replication.sh で 1 を学習し直しておく。
# 使い方（リポジトリの最上位のディレクトリで）:
#   setsid nohup bash experiments/trial-1b-base-sequences-5632-k-1000/run_both_phases_replication.sh \
#     [先に終わるのを待つプロセスの番号] &
set -u
WAIT_FOR=${1:-}
E=experiments/trial-1b-base-sequences-5632-k-1000
L=outputs/trial-1b-base-sequences-5632-k-1000/logs/unknown-word-acquisition-replication
RUNS=(run-1 run-2 run-3 run-4 run-5 run-6 run-7 run-8)
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FIRST_PHASE_EXPERIMENT=unknown-word-acquisition-replication
export SECOND_SENSE_EXPERIMENT=second-sense-addition-both-phases-replication
mkdir -p "$L"
if [ -n "$WAIT_FOR" ]; then
  while kill -0 "$WAIT_FOR" 2> /dev/null; do sleep 30; done
fi
note() { echo "=== $(date '+%F %T') $*" >> "$L/after_training.log"; }

# 学習は 1 枚の GPU で約 75 GB 使うので、解析は、学習が抜ける（使用中のメモリが 50 GB 未満になる）
# のを待ってから始める（解析どうしは、GPU を分け合ってよい）
wait_for_gpu() {
  while [ "$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits)" -ge 50000 ]; do
    sleep 60
  done
}

# 引数: GPU の番号、ログの名前、コマンド
job() {
  local gpu=$1 name=$2
  shift 2
  wait_for_gpu "$gpu"
  if CUDA_VISIBLE_DEVICES=$gpu uv run python "$@" >> "$L/$name.log" 2>&1; then
    note "終了: $name"
  else
    note "止まりました: $name"
  fi
}

if ! uv run python "$E/build_both_phases_replication.py" >> "$L/after_training.log" 2>&1; then
  note "止まりました: 2 の学習の設定"
  exit 1
fi
bash "$E/run_second_sense_replication.sh" &
(
  for run in "${RUNS[@]:0:4}"; do
    job 3 "grafting_first_$run" "$E/measure_second_sense_weight_grafting.py" measure first "$run"
  done
  for run in "${RUNS[@]:0:4}"; do
    job 3 "grafting_indicators_$run" "$E/measure_unknown_word_weight_grafting.py" measure "$run"
  done
) &
(
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
note "全て終わりました: 1 と 2 の学習し直しと解析"
