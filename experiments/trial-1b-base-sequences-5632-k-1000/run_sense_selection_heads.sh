#!/usr/bin/env bash
# 手がかりで語義を選ぶ head を探し、自然な多義語と、2つ目の語義を足した未知語で比べる（41|。GPU 0〜4）。
# 1. natural と、8 run の scan（GPU 0〜4 に順に割り当てる）
# 2. 8 run の sets（natural と全ての scan の後）
# 3. summarize（CPU）
# 結果がすでにある作業は飛ばす。
# 使い方（リポジトリの最上位のディレクトリで）:
#   setsid nohup bash experiments/trial-1b-base-sequences-5632-k-1000/run_sense_selection_heads.sh &
L=outputs/trial-1b-base-sequences-5632-k-1000/logs/second-sense-addition
S=experiments/trial-1b-base-sequences-5632-k-1000/measure_sense_selection_heads.py
O=outputs/partial-results/second-sense-evaluation/heads
GPUS=(0 1 2 3 4)

# 作業の一覧（"コマンド 引数 結果のファイル"。引数がなければ -）を、GPU ごとに順に流す
run_jobs() {
  local jobs=("$@")
  for slot in "${!GPUS[@]}"; do
    (
      for ((index = slot; index < ${#jobs[@]}; index += ${#GPUS[@]})); do
        read -r command argument output <<< "${jobs[$index]}"
        [ -f "$output" ] && continue
        [ "$argument" = "-" ] && argument=""
        name="heads_${command}${argument:+_$argument}"
        if CUDA_VISIBLE_DEVICES=${GPUS[$slot]} uv run python $S $command $argument \
          >> "$L/$name.log" 2>&1; then
          echo "=== $(date '+%F %T') 終了: $name" >> "$L/evaluation.log"
        else
          echo "=== $(date '+%F %T') 止まりました: $name" >> "$L/evaluation.log"
        fi
      done
    ) &
  done
  wait
}

first=("natural - $O/natural_sets.parquet")
for n in 1 2 3 4 5 6 7 8; do first+=("scan run-$n $O/scan/run-$n.parquet"); done
run_jobs "${first[@]}"
second=()
for n in 1 2 3 4 5 6 7 8; do second+=("sets run-$n $O/sets/run-$n.parquet"); done
run_jobs "${second[@]}"
uv run python $S summarize >> "$L/heads_summarize.log" 2>&1
echo "=== $(date '+%F %T') 全て終わりました: heads" >> "$L/evaluation.log"
