"""語義の追加の最初の結果: 足す語義が1つの3つの run と、対照の2つの run だけで、
課題1の正解率を測る。

measure-sense-accuracy は11の run が全て学習を終えるのを待つので、この試行だけ、run の一覧と、
解かせる問題（足す語義の数が1の条件だけ）を差し替えて実行する。結果は
outputs/partial-results/sense-addition-five-runs/<モデル>/ に書く（本来の解析の結果とは別の場所）。
使い方（リポジトリの最上位のディレクトリで）:
  CUDA_VISIBLE_DEVICES=0 uv run python \
    experiments/trial-1b-base-sequences-5632-k-1000/measure_partial_sense_accuracy.py \
    OLMo-2-0425-1B
"""

import sys
from pathlib import Path

from lexical_acquisition_mechanisms import sense_accuracy, sense_addition_assignments

RUNS = ["same-category-1", "other-category-1", "adjective-1", "noun-control", "adjective-control"]
sense_addition_assignments.run_names = lambda: list(RUNS)
sense_accuracy.RESULTS_DIRECTORY = Path("outputs/partial-results/sense-addition-five-runs")
evaluated_in = sense_accuracy.evaluated_in
sense_accuracy.evaluated_in = lambda run, question: (
    question["number_of_senses"] == 1 and evaluated_in(run, question)
)

from lexical_acquisition_mechanisms.command_line import main  # noqa: E402

sys.exit(main(["sense-addition", "measure-sense-accuracy", "--model", sys.argv[1]]))
