#!/usr/bin/env bash
 の試行の測定を GPU 1〜4 で流す（リポジトリの最上位のディレクトリで）| の試行の測定を GPU 1〜4 で流す（リポジトリの最上位のディレクトリで）
L=outputs/trial-1b-base-sequences-5632-k-1000/logs/second-sense-addition
S=experiments/trial-1b-base-sequences-5632-k-1000/measure_second_sense_meaning.py
for g in 1 2 3 4; do
  (
    for n in $g $((g + 4)); do
      CUDA_VISIBLE_DEVICES=$g uv run python $S measure run-$n >> $L/meaning_run-$n.log 2>&1 \
        && echo "=== $(date '+%F %T') 終了: meaning run-$n" >> $L/evaluation.log \
        || echo "=== $(date '+%F %T') 止まりました: meaning run-$n" >> $L/evaluation.log
    done
  ) &
done
wait
uv run python $S summarize >> $L/meaning_summarize.log 2>&1
echo "=== $(date '+%F %T') 全て終わりました: meaning" >> $L/evaluation.log
