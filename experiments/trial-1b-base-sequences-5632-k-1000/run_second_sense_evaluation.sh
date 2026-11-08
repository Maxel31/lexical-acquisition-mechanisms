#!/usr/bin/env bash
# 2つ目の語義の追加の試行の評価を、8 run の学習が終わってから順に流す（GPU は GPUS。既定は 1〜4）。
# 1. LLM のサーバ（2つを並べる）を立て、名詞の組ごとに最小対を作って判定し、サーバを止める
# 2. 学習前の 1B で最小対を確かめ、課題1の問題を作る
# 3. 8 run の、2つ目の語義を足す前と後の重みで、文脈の効果を測り、表にまとめる
# 使い方（リポジトリの最上位のディレクトリで）:
#   setsid nohup bash experiments/trial-1b-base-sequences-5632-k-1000/run_second_sense_evaluation.sh &
set -u
MODEL=OLMo-2-0425-1B
LOGS=outputs/trial-1b-base-sequences-5632-k-1000/logs/second-sense-addition
SCRIPTS=experiments/trial-1b-base-sequences-5632-k-1000
GPUS=${GPUS:-1,2,3,4}
IFS=, read -r -a GPU_LIST <<< "$GPUS"
export PYTHONUNBUFFERED=1
log() { echo "=== $(date '+%F %T') $*" >> "$LOGS/evaluation.log"; }
step() {  # ログの名前、GPU（空なら CPU）、スクリプト、引数
  local name=$1 gpu=$2
  shift 2
  log "開始: $name（GPU ${gpu:-なし}）"
  if CUDA_VISIBLE_DEVICES=$gpu uv run python "$@" >> "$LOGS/evaluation_$name.log" 2>&1; then
    log "終了: $name"
  else
    log "止まりました: $name（$LOGS/evaluation_$name.log）"
    return 1
  fi
}
for number in $(seq 1 8); do
  until [ -e "outputs/checkpoints/second-sense-addition/$MODEL/run-$number/completed.json" ]; do
    sleep 60
  done
done
log "8 run の学習が終わりました"

CUDA_VISIBLE_DEVICES=$GPUS setsid nohup uvx --from vllm==0.32.0 vllm serve \
  Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 --tensor-parallel-size 2 --data-parallel-size 2 \
  --port 8000 > "$LOGS/vllm.log" 2>&1 < /dev/null &
SERVER=$!
until curl -s localhost:8000/v1/models > /dev/null; do
  if ! kill -0 "$SERVER" 2> /dev/null; then
    log "止まりました: LLM のサーバ（$LOGS/vllm.log）"
    exit 1
  fi
  sleep 20
done
log "LLM のサーバが立ちました"
step generate "" "$SCRIPTS/build_second_sense_evaluation_data.py" generate
GENERATED=$?
kill -TERM -- -"$SERVER" 2> /dev/null || kill -TERM "$SERVER"
while pgrep -u "$USER" -f "[v]llm serve" > /dev/null; do sleep 10; done
log "LLM のサーバを止めました"
[ "$GENERATED" -eq 0 ] || exit 1

step check "${GPU_LIST[0]}" "$SCRIPTS/build_second_sense_evaluation_data.py" check || exit 1
step questions "" "$SCRIPTS/build_second_sense_evaluation_data.py" questions || exit 1

pids=()
for index in "${!GPU_LIST[@]}"; do
  (
    for number in $(seq $((index + 1)) "${#GPU_LIST[@]}" 8); do
      step "measure_run-$number" "${GPU_LIST[$index]}" \
        "$SCRIPTS/measure_second_sense_context_effects.py" measure "run-$number"
    done
  ) &
  pids+=($!)
done
wait "${pids[@]}"
step summarize "" "$SCRIPTS/measure_second_sense_context_effects.py" summarize
log "全て終わりました"
