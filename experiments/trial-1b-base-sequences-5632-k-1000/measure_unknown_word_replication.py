"""seed を変えて学習し直した未知語の獲得の 8 run で、3 つの指標を測り（16|）、獲得の順番を検定する
（17|）（試行の作業。43| の続き）。

測り方と検定は、元の学習と同じ（unknown-word-acquisition measure-indicators と
analyze-acquisition-order）。
正規化の基準は run ごとにとるので、対照の run は使わない。
- indicators（GPU 1 枚）: outputs/partial-results/unknown-word-acquisition-replication/ の
  indicators.parquet
- order（CPU）: 同じディレクトリの normalized_indicators.parquet・acquisition_curves.csv・
  acquisition_order.csv
使い方（リポジトリの最上位のディレクトリで）:
  E=experiments/trial-1b-base-sequences-5632-k-1000
  uv run python $E/measure_unknown_word_replication.py indicators
"""

import sys
from pathlib import Path

from lexical_acquisition_mechanisms import real_word_probes, unknown_word_assignments
from lexical_acquisition_mechanisms.acquisition_order import write_acquisition_order
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.settings import load_settings
from lexical_acquisition_mechanisms.unknown_word_indicators import BATCH_SIZE, write_indicators

EXPERIMENT = "unknown-word-acquisition-replication"
MODEL = "OLMo-2-0425-1B"
RESULTS = Path("outputs/partial-results") / EXPERIMENT


def indicators() -> None:
    checkpoints = {
        run: Path("outputs/checkpoints") / EXPERIMENT / MODEL / run / "final"
        for run in unknown_word_assignments.run_names()
    }
    write_indicators(
        checkpoints,
        lambda source: LanguageModel.load(source, batch_size=BATCH_SIZE),
        real_word_probes.probe_paths(MODEL)[0],
        RESULTS / "indicators.parquet",
    )


def order() -> None:
    write_acquisition_order(
        RESULTS / "indicators.parquet",
        (
            RESULTS / "normalized_indicators.parquet",
            RESULTS / "acquisition_curves.csv",
            RESULTS / "acquisition_order.csv",
        ),
        load_settings(Path("settings.yaml")).words.seed,
    )


if __name__ == "__main__":
    {"indicators": indicators, "order": order}[sys.argv[1]]()
