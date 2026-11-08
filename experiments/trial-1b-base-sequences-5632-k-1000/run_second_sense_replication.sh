#!/usr/bin/env bash
# 2つ目の語義の追加を、用例を差し込む順番だけ変えて学習し直し、解析をやり直す（37| の別の seed。
# GPU 0〜4）。学習データと学習の設定は、先に build_second_sense_replication.py で作っておく。
# 1. 8 run の学習（GPU ごとに順に。学習を終えた run は飛ばす）
# 2. run ごとの解析: 課題1の文脈の効果（27|）、表現（38|）、損失（25|）、重みを部品ごとに移す
#    分析（40|）、head を 1 つずつ止める走査（41|）
# 3. head の組をまとめて止める測定（41|）と、表の書き出し
# 結果は outputs/partial-results/second-sense-evaluation/second-sense-addition-replication/ に書く。
# 使い方（リポジトリの最上位のディレクトリで）:
#   setsid nohup bash experiments/trial-1b-base-sequences-5632-k-1000/run_second_sense_replication.sh \
#     [先に終わるのを待つプロセスの番号] &
set -u
WAIT_FOR=${1:-}
EXPERIMENT=second-sense-addition-replication
MODEL=OLMo-2-0425-1B
L=outputs/trial-1b-base-sequences-5632-k-1000/logs/$EXPERIMENT
E=experiments/trial-1b-base-sequences-5632-k-1000
GPUS=(0 1 2 3 4)
export SECOND_SENSE_EXPERIMENT=$EXPERIMENT
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$L"
if [ -n "$WAIT_FOR" ]; then
  while kill -0 "$WAIT_FOR" 2> /dev/null; do sleep 30; done
fi
note() { echo "=== $(date '+%F %T') $*" >> "$L/evaluation.log"; }

# 引数の仕事（1 つずつ、GPU の番号を受け取る関数の名前と引数）を、GPU ごとに順に流す
run_jobs() {
  local jobs=("$@")
  for slot in "${!GPUS[@]}"; do
    (
      for ((index = slot; index < ${#jobs[@]}; index += ${#GPUS[@]})); do
        read -r -a job <<< "${jobs[$index]}"
        "${job[0]}" "${GPUS[$slot]}" "${job[@]:1}"
      done
    ) &
  done
  wait
}

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

analyze() {  # GPU run
  local gpu=$1 run=$2 name script arguments
  for name in context meaning losses grafting heads_scan; do
    case $name in
      context) script=measure_second_sense_context_effects.py arguments="measure $run" ;;
      meaning) script=measure_second_sense_meaning.py arguments="measure $run" ;;
      losses) script=measure_second_sense_losses.py arguments="measure $run" ;;
      grafting) script=measure_second_sense_weight_grafting.py arguments="measure second $run" ;;
      heads_scan) script=measure_sense_selection_heads.py arguments="scan $run" ;;
    esac
    # shellcheck disable=SC2086
    if CUDA_VISIBLE_DEVICES=$gpu uv run python "$E/$script" $arguments \
      >> "$L/${name}_$run.log" 2>&1; then
      note "終了: $name $run"
    else
      note "止まりました: $name $run"
    fi
  done
}

head_sets() {  # GPU run
  local gpu=$1 run=$2
  if CUDA_VISIBLE_DEVICES=$gpu uv run python "$E/measure_sense_selection_heads.py" sets "$run" \
    >> "$L/heads_sets_$run.log" 2>&1; then
    note "終了: heads_sets $run"
  else
    note "止まりました: heads_sets $run"
  fi
}

runs=(run-1 run-2 run-3 run-4 run-5 run-6 run-7 run-8)
run_jobs "${runs[@]/#/train }"
run_jobs "${runs[@]/#/analyze }"
run_jobs "${runs[@]/#/head_sets }"
uv run python "$E/measure_second_sense_context_effects.py" summarize >> "$L/summarize.log" 2>&1
uv run python "$E/measure_second_sense_meaning.py" summarize >> "$L/summarize.log" 2>&1
uv run python "$E/measure_second_sense_losses.py" summarize >> "$L/summarize.log" 2>&1
uv run python "$E/measure_second_sense_weight_grafting.py" summarize second >> "$L/summarize.log" 2>&1
uv run python "$E/measure_sense_selection_heads.py" summarize >> "$L/summarize.log" 2>&1
note "全て終わりました: $EXPERIMENT"
