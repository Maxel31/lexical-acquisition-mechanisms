#!/usr/bin/env bash
# 変わった重みを部品ごとに移し、役割を担う部品を探す（GPU 1〜4）。phase は次のどちらか。
# - second: 2つ目の語義の追加（40|。2つ目の語義を足す前 → 後）
# - first: 未知語が既知語になるフェーズ（39|。学習前 → 未知語の獲得の後）
# 使い方（リポジトリの最上位のディレクトリで）:
#   setsid nohup bash experiments/trial-1b-base-sequences-5632-k-1000/run_second_sense_weight_grafting.sh first &
PHASE=$1
L=outputs/trial-1b-base-sequences-5632-k-1000/logs/second-sense-addition
S=experiments/trial-1b-base-sequences-5632-k-1000/measure_second_sense_weight_grafting.py
for g in 1 2 3 4; do
  (
    for n in $g $((g + 4)); do
      if CUDA_VISIBLE_DEVICES=$g uv run python $S measure "$PHASE" "run-$n" \
        >> "$L/grafting-${PHASE}_run-$n.log" 2>&1; then
        echo "=== $(date '+%F %T') 終了: grafting-$PHASE run-$n" >> "$L/evaluation.log"
      else
        echo "=== $(date '+%F %T') 止まりました: grafting-$PHASE run-$n" >> "$L/evaluation.log"
      fi
    done
  ) &
done
wait
uv run python $S summarize "$PHASE" >> "$L/grafting-${PHASE}_summarize.log" 2>&1
echo "=== $(date '+%F %T') 全て終わりました: grafting-$PHASE" >> "$L/evaluation.log"
