"""1 も 2 も seed を変えて学習し直すための、2 の学習の設定を作る（試行の作業。43| の続き）。

2 の学習データは、build_second_sense_replication.py で作ったもの（差し込む順番の seed が
words.seed + 1）をそのまま使う。学習を始める重みだけを、seed を変えて学習し直した 1 の最後の重み
（build_unknown_word_replication.py）にする。
- outputs/runs/second-sense-addition-both-phases-replication/<モデル>/<run>.yaml
使い方（リポジトリの最上位のディレクトリで。CPU）:
  uv run python experiments/trial-1b-base-sequences-5632-k-1000/build_both_phases_replication.py
"""

from pathlib import Path

import yaml

EXPERIMENT = "second-sense-addition-both-phases-replication"
FIRST_PHASE = "unknown-word-acquisition-replication"
MODEL = "OLMo-2-0425-1B"
SOURCE_RUNS = Path("outputs/runs/second-sense-addition-replication") / MODEL
RUNS = Path("outputs/runs") / EXPERIMENT / MODEL


def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    for source in sorted(SOURCE_RUNS.glob("run-*.yaml")):
        run = yaml.safe_load(source.read_text(encoding="utf-8"))
        first_phase = Path("outputs/checkpoints") / FIRST_PHASE / MODEL / run["run"] / "final"
        if not (first_phase / "config.json").exists():
            raise SystemExit(f"1 の重みがありません: {first_phase}")
        run["experiment"] = EXPERIMENT
        run["model_source"] = {"repository": str(first_phase), "revision": None}
        run["output_directory"] = str(Path("outputs/checkpoints") / EXPERIMENT / MODEL / run["run"])
        (RUNS / source.name).write_text(
            yaml.safe_dump(run, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        print(f"{RUNS / source.name} を書きました。", flush=True)


if __name__ == "__main__":
    main()
