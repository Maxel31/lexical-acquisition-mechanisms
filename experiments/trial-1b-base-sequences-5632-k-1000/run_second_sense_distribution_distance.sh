#!/usr/bin/env bash
# 2つ目の語義の追加で、手がかりで語義を選べるかを、次のトークンの分布全体で測る（55|。GPU 0〜3）。
# 元の学習と、2 つの学習し直し（用例を差し込む順番だけを変えたもの・1 から学習し直したもの）の、
# 全ての run で測り、まとめる。1 run は約 25 秒。
# 使い方（リポジトリの最上位のディレクトリで）:
#   setsid nohup bash experiments/trial-1b-base-sequences-5632-k-1000/run_second_sense_distribution_distance.sh &
set -u
L=outputs/trial-1b-base-sequences-5632-k-1000/logs/second-sense-addition
S=experiments/trial-1b-base-sequences-5632-k-1000/measure_second_sense_distribution_distance.py
export PYTHONUNBUFFERED=1
mkdir -p "$L"
note() { echo "=== $(date '+%F %T') $*" >> "$L/evaluation.log"; }

# 引数: 2 の学習の名前、1 の学習の名前
measure_all() {
  export SECOND_SENSE_EXPERIMENT=$1 FIRST_PHASE_EXPERIMENT=$2
  for g in 0 1 2 3; do
    (
      for n in $((g + 1)) $((g + 5)); do
        if CUDA_VISIBLE_DEVICES=$g uv run python $S measure "run-$n" \
          >> "$L/distribution_${1}_run-$n.log" 2>&1; then
          note "終了: distribution $1 run-$n"
        else
          note "止まりました: distribution $1 run-$n"
        fi
      done
    ) &
  done
  wait
  uv run python $S summarize >> "$L/distribution_${1}_summarize.log" 2>&1
  note "全て終わりました: distribution $1"
}

measure_all second-sense-addition unknown-word-acquisition
measure_all second-sense-addition-replication unknown-word-acquisition
measure_all second-sense-addition-both-phases-replication unknown-word-acquisition-replication
