#!/usr/bin/env bash
# 2つ目の語義の追加の試行で、変わった重みを部品ごとに移し、使い分けを担う部品を探す（40|）。
# GPU 1〜4 を使う。損失の測定（run_second_sense_losses.sh）が終わるのを待ってから始める。
# 使い方（リポジトリの最上位のディレクトリで）:
#   setsid nohup bash experiments/trial-1b-base-sequences-5632-k-1000/run_second_sense_weight_grafting.sh &
L=outputs/trial-1b-base-sequences-5632-k-1000/logs/second-sense-addition
S=experiments/trial-1b-base-sequences-5632-k-1000/measure_second_sense_weight_grafting.py
until grep -q "全て終わりました: losses" "$L/evaluation.log"; do sleep 30; done
for g in 1 2 3 4; do
  (
    for n in $g $((g + 4)); do
      if CUDA_VISIBLE_DEVICES=$g uv run python $S measure "run-$n" >> "$L/grafting_run-$n.log" 2>&1; then
        echo "=== $(date '+%F %T') 終了: grafting run-$n" >> "$L/evaluation.log"
      else
        echo "=== $(date '+%F %T') 止まりました: grafting run-$n" >> "$L/evaluation.log"
      fi
    done
  ) &
done
wait
uv run python $S summarize >> "$L/grafting_summarize.log" 2>&1
echo "=== $(date '+%F %T') 全て終わりました: grafting" >> "$L/evaluation.log"
