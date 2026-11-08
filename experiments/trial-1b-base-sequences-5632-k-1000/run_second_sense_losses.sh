#!/usr/bin/env bash
# 2つ目の語義の追加の試行で、1つ目と2つ目の語義の用例での未知語の損失（25|）を、GPU 1〜4 で測る。
# 表現の測定（run_second_sense_meaning.sh）が終わるのを待ってから始める。
# 使い方（リポジトリの最上位のディレクトリで）:
#   setsid nohup bash experiments/trial-1b-base-sequences-5632-k-1000/run_second_sense_losses.sh &
L=outputs/trial-1b-base-sequences-5632-k-1000/logs/second-sense-addition
S=experiments/trial-1b-base-sequences-5632-k-1000/measure_second_sense_losses.py
until grep -q "全て終わりました: meaning" "$L/evaluation.log"; do sleep 30; done
for g in 1 2 3 4; do
  (
    for n in $g $((g + 4)); do
      if CUDA_VISIBLE_DEVICES=$g uv run python $S measure "run-$n" >> "$L/losses_run-$n.log" 2>&1; then
        echo "=== $(date '+%F %T') 終了: losses run-$n" >> "$L/evaluation.log"
      else
        echo "=== $(date '+%F %T') 止まりました: losses run-$n" >> "$L/evaluation.log"
      fi
    done
  ) &
done
wait
uv run python $S summarize >> "$L/losses_summarize.log" 2>&1
echo "=== $(date '+%F %T') 全て終わりました: losses" >> "$L/evaluation.log"
