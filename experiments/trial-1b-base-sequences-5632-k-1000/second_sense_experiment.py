"""2つ目の語義の追加の試行の解析で、どの学習の重みを読み、結果をどこに書くか（試行の作業）。

環境変数 SECOND_SENSE_EXPERIMENT に second-sense-addition-replication を渡すと、用例を差し込む
順番を変えて学習し直した重み（build_second_sense_replication.py）を読み、結果を
outputs/partial-results/second-sense-evaluation/second-sense-addition-replication/ に書く。
課題1の問題と最小対は、どちらでも同じもの（DIRECTORY にあるもの）を読む。
"""

import os
from pathlib import Path

from lexical_acquisition_mechanisms import second_sense_assignments
from lexical_acquisition_mechanisms.run_settings import first_phase_weights

DIRECTORY = Path("outputs/partial-results/second-sense-evaluation")
MODEL = "OLMo-2-0425-1B"
EXPERIMENT = os.environ.get("SECOND_SENSE_EXPERIMENT", second_sense_assignments.EXPERIMENT)
RESULTS = DIRECTORY if EXPERIMENT == second_sense_assignments.EXPERIMENT else DIRECTORY / EXPERIMENT


def weights(run: str, checkpoint: str) -> Path:
    """2つ目の語義を足す前（before）と、足した後の学習の途中・最後の重み。"""
    if checkpoint == "before":
        return first_phase_weights(MODEL, run)
    return Path("outputs/checkpoints") / EXPERIMENT / MODEL / run / checkpoint
