"""未知語の獲得を、学習の乱数の seed だけ変えて学習し直すための、学習データと学習の設定を作る
（試行の作業。43| の続きで、1 も学習し直す）。

1 の学習の乱数は、用例を基データに差し込む順番と、1 トークンの未知語の埋め込みの初期値の 2 つ
（学習は系列を順に読む）。割り当て・用例・基データの系列は 1 と同じにし、この 2 つの seed だけを
words.seed + SEED_OFFSET にする。対照の run は学習しない（3 つの指標の計算に使わないため）。
- data/training/unknown-word-acquisition-replication/<run>/: 学習データ
- outputs/runs/unknown-word-acquisition-replication/<モデル>/<run>.yaml: 学習の設定（1 の設定の、
  学習データと保存先と実験の名前と seed だけを変えたもの）
使い方（リポジトリの最上位のディレクトリで。CPU）:
  uv run python experiments/trial-1b-base-sequences-5632-k-1000/build_unknown_word_replication.py
"""

from pathlib import Path

import yaml

from lexical_acquisition_mechanisms import (
    base_data,
    passage_finalization,
    unknown_word_assignments,
    word_finalization,
)
from lexical_acquisition_mechanisms.settings import load_settings
from lexical_acquisition_mechanisms.unknown_word_training_data import (
    write_unknown_word_training_data,
)

EXPERIMENT = "unknown-word-acquisition-replication"
MODEL = "OLMo-2-0425-1B"
SEED_OFFSET = 1
TRAINING_DIRECTORY = Path("data/training") / EXPERIMENT
SOURCE_RUNS = Path("outputs/runs") / unknown_word_assignments.EXPERIMENT / MODEL
RUNS = Path("outputs/runs") / EXPERIMENT / MODEL


def main() -> None:
    settings = load_settings(Path("settings.yaml"))
    words = settings.words.model_copy(update={"seed": settings.words.seed + SEED_OFFSET})
    write_unknown_word_training_data(
        settings.model_copy(update={"words": words}),
        word_finalization.OUTPUT_PATH,
        passage_finalization.PASSAGES_PATH,
        unknown_word_assignments.ASSIGNMENTS_PATH,
        base_data.output_paths()["base"],
        TRAINING_DIRECTORY,
    )
    RUNS.mkdir(parents=True, exist_ok=True)
    for source in sorted(SOURCE_RUNS.glob("run-*.yaml")):
        run = yaml.safe_load(source.read_text(encoding="utf-8"))
        run["experiment"] = EXPERIMENT
        run["training_tokens"] = str(TRAINING_DIRECTORY / run["run"] / "tokens.npy")
        run["passage_starts"] = str(TRAINING_DIRECTORY / run["run"] / "passage_starts.npy")
        run["output_directory"] = str(Path("outputs/checkpoints") / EXPERIMENT / MODEL / run["run"])
        run["seed"] = run["seed"] + SEED_OFFSET
        (RUNS / source.name).write_text(
            yaml.safe_dump(run, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        print(f"{RUNS / source.name} を書きました。", flush=True)


if __name__ == "__main__":
    main()
