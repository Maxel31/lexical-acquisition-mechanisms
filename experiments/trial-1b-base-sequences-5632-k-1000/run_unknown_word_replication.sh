#!/usr/bin/env bash
# 未知語の獲得の 8 run を、学習の乱数の seed だけ変えて学習し直す（43| の続き。GPU 0〜4）。
# 学習データと学習の設定は、先に build_unknown_word_replication.py で作っておく。
# 学習を終えた run は飛ばす。
# 使い方（リポジトリの最上位のディレクトリで）:
#   setsid nohup bash experiments/trial-1b-base-sequences-5632-k-1000/run_unknown_word_replication.sh \
#     [先に終わるのを待つプロセスの番号] &
set -u
WAIT_FOR=${1:-}
EXPERIMENT=unknown-word-acquisition-replication
MODEL=OLMo-2-0425-1B
L=outputs/trial-1b-base-sequences-5632-k-1000/logs/$EXPERIMENT
GPUS=(0 1 2 3 4)
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$L"
if [ -n "$WAIT_FOR" ]; then
  while kill -0 "$WAIT_FOR" 2> /dev/null; do sleep 30; done
fi
note() { echo "=== $(date '+%F %T') $*" >> "$L/training.log"; }

train() {  # GPU run
  local gpu=$1 run=$2
  [ -e "outputs/checkpoints/$EXPERIMENT/$MODEL/$run/completed.json" ] && return
  note "開始: 学習 $run（GPU $gpu）"
  if CUDA_VISIBLE_DEVICES=$gpu uv run torchrun --nproc-per-node 1 --master-port $((29500 + gpu)) \
    -m lexical_acquisition_mechanisms training run \
    --run-settings "outputs/runs/$EXPERIMENT/$MODEL/$run.yaml" >> "$L/train_$run.log" 2>&1; then
    note "終了: 学習 $run"
  else
    note "止まりました: 学習 $run"
  fi
}

runs=(run-1 run-2 run-3 run-4 run-5 run-6 run-7 run-8)
for slot in "${!GPUS[@]}"; do
  (
    for ((index = slot; index < ${#runs[@]}; index += ${#GPUS[@]})); do
      train "${GPUS[$slot]}" "${runs[$index]}"
    done
  ) &
done
wait
note "全て終わりました: $EXPERIMENT"
