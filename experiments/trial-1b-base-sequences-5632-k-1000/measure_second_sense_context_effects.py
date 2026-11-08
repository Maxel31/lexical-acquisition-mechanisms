"""2つ目の語義の追加の試行で、課題1の文脈の効果を測る（試行の作業。本来のコマンドは後で作る）。

- measure <run>（GPU 1枚）: その run の、2つ目の語義を足す前の重み（未知語の獲得の最後の
  重み）と、足した後の学習の 25・50・75・100% の重みで、run の全ての未知語に、その未知語の
  (X, Y) の組の課題1の問題（build_second_sense_evaluation_data.py questions が書いたもの。空欄は
  未知語）を解かせ、選択肢ごとの対数確率を
  outputs/partial-results/second-sense-evaluation/context/<run>.parquet に書く。
- summarize（CPU）: 最小対の組ごとに、Y の割合（X と Y の確率の和を1にしたときの Y の確率）の、
  Y の手がかりの文と X の手がかりの文の差（文脈の効果）を出し、重み × 2つ目の露出回数 × 種類ごとに
  平均して summary.txt に書く。2つ目の露出回数 0 の未知語（Y を足していない）が対照になる。
"""

import gc
import sys
from pathlib import Path

import numpy as np
import pandas
import pyarrow as arrow
import pyarrow.parquet as parquet
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from save_sense_choice_probabilities import choice_log_probabilities  # noqa: E402

from lexical_acquisition_mechanisms import second_sense_assignments, word_finalization  # noqa: E402
from lexical_acquisition_mechanisms.csv_tables import read_rows  # noqa: E402
from lexical_acquisition_mechanisms.language_model import LanguageModel  # noqa: E402
from lexical_acquisition_mechanisms.run_settings import first_phase_weights  # noqa: E402
from lexical_acquisition_mechanisms.second_sense_assignments import (  # noqa: E402
    SecondSenseAssignment,
)
from lexical_acquisition_mechanisms.settings import ModelSource  # noqa: E402
from lexical_acquisition_mechanisms.word_selection import SelectedWord  # noqa: E402

DIRECTORY = Path("outputs/partial-results/second-sense-evaluation")
MODEL = "OLMo-2-0425-1B"
CHECKPOINTS = ["before", "checkpoint-25", "checkpoint-50", "checkpoint-75", "final"]


def weights(run: str, checkpoint: str) -> Path:
    if checkpoint == "before":
        return first_phase_weights(MODEL, run)
    return (
        Path("outputs/checkpoints") / second_sense_assignments.EXPERIMENT / MODEL / run / checkpoint
    )


def measure(run: str) -> None:
    words = {word.word: word for word in read_rows(word_finalization.OUTPUT_PATH, SelectedWord)}
    questions = parquet.read_table(DIRECTORY / "questions.parquet").to_pylist()
    by_pair: dict[tuple[str, str], list[dict]] = {}
    for question in questions:
        by_pair.setdefault((question["first_noun"], question["second_noun"]), []).append(question)
    assignments = [
        row
        for row in read_rows(second_sense_assignments.ASSIGNMENTS_PATH, SecondSenseAssignment)
        if row.run == run
    ]
    chosen = [
        (assignment, question)
        for assignment in assignments
        for question in by_pair.get((assignment.first_noun, assignment.second_noun), [])
    ]
    ids = [
        [int(token) for token in words[assignment.unknown_word].token_ids.split()]
        for assignment, _ in chosen
    ]
    rows = []
    for checkpoint in CHECKPOINTS:
        model = LanguageModel.load(ModelSource(repository=str(weights(run, checkpoint))))
        values = choice_log_probabilities(model, [question for _, question in chosen], ids)
        rows += [
            {
                "run": run,
                "checkpoint": checkpoint,
                "unknown_word": assignment.unknown_word,
                "token_count": assignment.token_count,
                "first_exposure": assignment.first_exposure,
                "second_exposure": assignment.second_exposure,
                "kind": assignment.kind,
                "first_noun": question["first_noun"],
                "second_noun": question["second_noun"],
                "pair": question["pair"],
                "cue": question["cue"],
                "choices": question["choices"],
                "log_probabilities": value,
            }
            for (assignment, question), value in zip(chosen, values, strict=True)
        ]
        print(f"{run} {checkpoint}: {len(chosen)} 問", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    (DIRECTORY / "context").mkdir(parents=True, exist_ok=True)
    parquet.write_table(arrow.Table.from_pylist(rows), DIRECTORY / "context" / f"{run}.parquet")


def summarize() -> None:
    table = pandas.concat(
        parquet.read_table(path).to_pandas()
        for path in sorted((DIRECTORY / "context").glob("*.parquet"))
    )
    probabilities = [np.exp(np.array(value)) for value in table["log_probabilities"]]
    table["second_share"] = [
        row[list(choices).index(second)] / row.sum()
        for row, choices, second in zip(
            probabilities, table["choices"], table["second_noun"], strict=True
        )
    ]
    table["correct"] = [
        list(choices)[int(np.argmax(row))] == (second if cue == "second" else first)
        for row, choices, first, second, cue in zip(
            probabilities,
            table["choices"],
            table["first_noun"],
            table["second_noun"],
            table["cue"],
            strict=True,
        )
    ]
    keys = ["run", "checkpoint", "unknown_word", "second_exposure", "kind", "pair"]
    wide = table.pivot_table(index=keys, columns="cue", values=["second_share", "correct"])
    pairs = pandas.DataFrame(
        {
            "second_share_at_second_cue": wide[("second_share", "second")],
            "second_share_at_first_cue": wide[("second_share", "first")],
            "both_correct": (wide[("correct", "second")] > 0.5)
            & (wide[("correct", "first")] > 0.5),
        }
    )
    pairs["context_effect"] = (
        pairs["second_share_at_second_cue"] - pairs["second_share_at_first_cue"]
    )
    # 未知語ごとに平均してから、条件ごとに平均する（最小対の数が多い未知語に偏らないように）
    per_word = pairs.groupby(level=keys[:5]).mean().reset_index()
    per_word["checkpoint"] = pandas.Categorical(per_word["checkpoint"], CHECKPOINTS, ordered=True)
    grouped = per_word.groupby(["kind", "second_exposure", "checkpoint"], observed=True)
    summary = grouped[
        [
            "context_effect",
            "second_share_at_second_cue",
            "second_share_at_first_cue",
            "both_correct",
        ]
    ].mean()
    summary["words"] = grouped.size()
    text = summary.round(3).to_string()
    (DIRECTORY / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    if sys.argv[1] == "measure":
        measure(sys.argv[2])
    else:
        summarize()
