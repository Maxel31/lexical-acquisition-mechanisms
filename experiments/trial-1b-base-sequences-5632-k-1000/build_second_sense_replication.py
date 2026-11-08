"""2つ目の語義の追加を、用例を差し込む順番だけ変えて学習し直すための、学習データと学習の設定を作る
（試行の作業。37| の別の seed）。

学習の乱数は、用例を基データに差し込む順番だけ（学習は系列を順に読み、2 では新しい埋め込みを
初期化しない）。そこで、割り当て・用例・基データの系列は 2 と同じにし、差し込む順番の seed だけを
words.seed + SEED_OFFSET にする。
- data/training/second-sense-addition-replication/<run>/: 学習データ
- outputs/runs/second-sense-addition-replication/<モデル>/<run>.yaml: 学習の設定（2 の設定の、
  学習データと保存先と実験の名前だけを変えたもの）
使い方（リポジトリの最上位のディレクトリで。CPU）:
  uv run python experiments/trial-1b-base-sequences-5632-k-1000/build_second_sense_replication.py
"""

from pathlib import Path

import yaml

from lexical_acquisition_mechanisms import (
    base_data,
    passage_finalization,
    second_sense_assignments,
    word_finalization,
)
from lexical_acquisition_mechanisms.second_sense_training_data import (
    write_second_sense_training_data,
)
from lexical_acquisition_mechanisms.settings import load_settings

EXPERIMENT = "second-sense-addition-replication"
MODEL = "OLMo-2-0425-1B"
SEED_OFFSET = 1
TRAINING_DIRECTORY = Path("data/training") / EXPERIMENT
SOURCE_RUNS = Path("outputs/runs") / second_sense_assignments.EXPERIMENT / MODEL
RUNS = Path("outputs/runs") / EXPERIMENT / MODEL


def main() -> None:
    settings = load_settings(Path("settings.yaml"))
    words = settings.words.model_copy(update={"seed": settings.words.seed + SEED_OFFSET})
    write_second_sense_training_data(
        settings.model_copy(update={"words": words}),
        word_finalization.OUTPUT_PATH,
        passage_finalization.PASSAGES_PATH,
        second_sense_assignments.ASSIGNMENTS_PATH,
        base_data.output_paths()["base"],
        TRAINING_DIRECTORY,
    )
    RUNS.mkdir(parents=True, exist_ok=True)
    for source in sorted(SOURCE_RUNS.glob("*.yaml")):
        run = yaml.safe_load(source.read_text(encoding="utf-8"))
        run["experiment"] = EXPERIMENT
        run["training_tokens"] = str(TRAINING_DIRECTORY / run["run"] / "tokens.npy")
        run["passage_starts"] = str(TRAINING_DIRECTORY / run["run"] / "passage_starts.npy")
        run["output_directory"] = str(Path("outputs/checkpoints") / EXPERIMENT / MODEL / run["run"])
        (RUNS / source.name).write_text(
            yaml.safe_dump(run, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        print(f"{RUNS / source.name} を書きました。", flush=True)


if __name__ == "__main__":
    main()
