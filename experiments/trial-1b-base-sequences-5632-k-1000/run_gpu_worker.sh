#!/bin/bash
# 試行（1B、基データ 5,632 系列、K = 1000）の学習と解析を、GPU 1 枚ぶん受け持つ作業者。
# リポジトリの最上位のディレクトリで、使う GPU ごとに1つずつ起動する。仕事は共有の列から取る。
# - 学習の列（<ログのディレクトリ>/training_queue.txt）: 1行に1つ「train <run の設定ファイル>」
# - 解析の列（<ログのディレクトリ>/analysis_queue.txt）: 1行に1つ実験の名前。その実験の run が全て学習を
#   終え（completed.json）、評価データが確定していたら、学習より先に解析を行う
# - 学習の列が空で、解析の条件がそろっていなければ、ほかの GPU の学習が終わるのを待つ
# 使い方: run_gpu_worker.sh <GPU の番号> <ログのディレクトリ> [先に終わるのを待つプロセスの番号]
set -u
GPU=$1
LOGS=$2
WAIT_FOR=${3:-}
MODEL=OLMo-2-0425-1B
A=unknown-word-acquisition
B=sense-addition
QUEUE=$LOGS/training_queue.txt
ANALYSES=$LOGS/analysis_queue.txt
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -n "$WAIT_FOR" ]; then
  while kill -0 "$WAIT_FOR" 2> /dev/null; do sleep 30; done
fi
take_train() { flock "$QUEUE.lock" bash -c "head -1 '$QUEUE'; sed -i 1d '$QUEUE'"; }
ready() {  # 実験の run が全て学習を終え、評価データが確定したか
  local experiment=$1 settings
  [ -e data/evaluation/sense_addition_questions.parquet.record.json ] || return 1
  for settings in outputs/runs/"$experiment"/"$MODEL"/*.yaml; do
    [ -e "outputs/checkpoints/$experiment/$MODEL/$(basename "$settings" .yaml)/completed.json" ] || return 1
  done
}
take_ready_analysis() {  # 条件がそろった解析を1つ取る（なければ空）
  (
    flock 9
    while read -r experiment; do
      if ready "$experiment"; then
        grep -v -x "$experiment" "$ANALYSES" > "$ANALYSES.new"
        mv "$ANALYSES.new" "$ANALYSES"
        echo "$experiment"
        break
      fi
    done < "$ANALYSES"
  ) 9> "$ANALYSES.lock"
}
command_on() {  # GPU（空なら CPU）、ログの名前、コマンド
  local gpu=$1 name=$2
  shift 2
  echo "=== $(date '+%F %T') 開始: $*（GPU ${gpu:-なし}）"
  CUDA_VISIBLE_DEVICES=$gpu uv run lexical-acquisition-mechanisms "$@" >> "$LOGS/$name.log" 2>&1
  local status=$?
  echo "=== $(date '+%F %T') 終了（$status）: $*（GPU ${gpu:-なし}）"
  return $status
}
train() {
  local settings=$1 experiment name
  experiment=$(basename "$(dirname "$(dirname "$settings")")")
  name=$(basename "$settings" .yaml)
  echo "=== $(date '+%F %T') 開始: $experiment $name（GPU $GPU）"
  CUDA_VISIBLE_DEVICES=$GPU uv run torchrun --nproc-per-node 1 --master-port $((29500 + GPU)) \
    -m lexical_acquisition_mechanisms training run --run-settings "$settings" \
    >> "$LOGS/train_${experiment}_$name.log" 2>&1
  local status=$?
  echo "=== $(date '+%F %T') 終了（$status）: $experiment $name（GPU $GPU）"
  return $status
}
analysis() {
  local experiment=$1
  if [ "$experiment" = $A ]; then
    command_on "$GPU" analysis_$A $A measure-indicators --model $MODEL &&
      command_on "" analysis_$A $A analyze-acquisition-order --model $MODEL &&
      command_on "" analysis_$A $A make-report
  else
    command_on "$GPU" analysis_$B $B measure-sense-accuracy --model $MODEL &&
      command_on "$GPU" analysis_$B $B measure-head-and-mlp-effects --model $MODEL &&
      command_on "" analysis_$B $B select-significant-heads-and-mlps --model $MODEL &&
      command_on "$GPU" analysis_$B $B train-attribute-probes --model $MODEL &&
      command_on "$GPU" analysis_$B $B test-attribute-directions --model $MODEL &&
      command_on "" analysis_$B $B make-report
  fi
}
echo "=== $(date '+%F %T') GPU $GPU の作業者を始めます"
while true; do
  experiment=$(take_ready_analysis)
  if [ -n "$experiment" ]; then
    analysis "$experiment" || echo "=== 止まりました: 解析 $experiment（GPU $GPU。学習は続けます）"
    continue
  fi
  job=$(take_train)
  if [ -n "$job" ]; then
    train "${job#train }" || echo "=== 止まりました: ${job#train }（GPU $GPU）"
    continue
  fi
  [ -s "$ANALYSES" ] || break
  if pgrep -u "$USER" -f "[t]raining run --run-settings" > /dev/null; then
    sleep 60
    continue
  fi
  echo "=== $(date '+%F %T') 止まりました: 解析の条件がそろいません（$(tr '\n' ' ' < "$ANALYSES")。GPU $GPU）"
  break
done
echo "=== $(date '+%F %T') GPU $GPU の作業者を終えました"
