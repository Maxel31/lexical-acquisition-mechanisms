"""save_sense_choice_probabilities.py の結果を表にまとめる（CPU）。

最小対の2つの問題（手がかりが donor の文と、元の語義の文）を組にして、次を出す。
- donor の割合: 選択肢の確率の和を1にしたときの、その組の donor の確率
- 文脈の効果: donor の文での donor の割合 − 元の語義の文での donor の割合
  （選択肢の偏りは打ち消される）
- 両方正解: 組の2つの問題の両方で、最も確率の大きい選択肢が正解
- 選択肢の確率の和: 2つの選択肢以外に確率が流れていないか（答えの形の確認）
結果は outputs/partial-results/sense-choice-probabilities/summary.txt に書く。
使い方（リポジトリの最上位のディレクトリで）:
  uv run python \
    experiments/trial-1b-base-sequences-5632-k-1000/summarize_sense_choice_probabilities.py
"""

import math
from pathlib import Path

import pandas
import pyarrow.parquet as parquet

from lexical_acquisition_mechanisms import evaluation_finalization

DIRECTORY = Path("outputs/partial-results/sense-choice-probabilities")
PAIR = ["kind", "number_of_senses", "target", "donor", "donor_sentence"]
CHECKPOINTS = ["checkpoint-25", "checkpoint-50", "checkpoint-75", "final"]


def with_questions(rows: pandas.DataFrame) -> pandas.DataFrame:
    questions = pandas.DataFrame(
        parquet.read_table(evaluation_finalization.QUESTIONS_PATH).to_pylist()
    )
    table = rows.join(questions, on="question")
    probabilities = [[math.exp(value) for value in row] for row in table["log_probabilities"]]
    table["choice_mass"] = [sum(row) for row in probabilities]
    table["donor_share"] = [
        row[list(choices).index(donor)] / sum(row)
        for row, choices, donor in zip(probabilities, table["choices"], table["donor"], strict=True)
    ]
    table["correct"] = [
        choices[max(range(len(row)), key=row.__getitem__)] == answer
        for row, choices, answer in zip(
            probabilities, table["choices"], table["answer"], strict=True
        )
    ]
    return table


def pairs(table: pandas.DataFrame, keys: list[str]) -> pandas.DataFrame:
    """最小対の組ごとに、2つの手がかりの donor の割合と正解を横に並べる。"""
    wide = table.pivot_table(
        index=keys + PAIR, columns="cue", values=["donor_share", "correct"], aggfunc="first"
    )
    result = pandas.DataFrame(
        {
            "donor_share_donor_cue": wide[("donor_share", "donor")],
            "donor_share_original_cue": wide[("donor_share", "target")],
            "both_correct": wide[("correct", "donor")].astype(bool)
            & wide[("correct", "target")].astype(bool),
        }
    )
    result["context_effect"] = result["donor_share_donor_cue"] - result["donor_share_original_cue"]
    return result.reset_index()


def context_summary() -> str:
    lines = ["## 学習前のモデルで、空欄に入れる語を変えた課題1（足す語義が1つの問題）"]
    for path in sorted((DIRECTORY / "context").glob("*.parquet")):
        table = with_questions(parquet.read_table(path).to_pandas())
        table = table[table["number_of_senses"] == 1]
        grouped = pairs(table, ["model", "word"]).groupby(["model", "kind", "word"])
        summary = grouped[
            ["donor_share_donor_cue", "donor_share_original_cue", "context_effect", "both_correct"]
        ].mean()
        summary["pairs"] = grouped.size()
        mass = table.groupby(["model", "kind", "word"])["choice_mass"].mean()
        lines.append(summary.join(mass).round(2).to_string())
    return "\n\n".join(lines)


def checkpoint_summary() -> str:
    lines = ["## 語義の追加の run の、学習の途中の重みでの課題1（対照の run は同じ問題）"]
    tables = []
    for checkpoint in CHECKPOINTS:
        for path in sorted((DIRECTORY / "checkpoints").glob(f"*/{checkpoint}.parquet")):
            tables.append(with_questions(parquet.read_table(path).to_pandas()))
    if not tables:
        return lines[0] + "\n（まだ結果がありません）"
    table = pandas.concat(tables)
    table["condition"] = [
        "control" if run.endswith("control") else "trained" for run in table["run"]
    ]
    paired = pairs(table, ["checkpoint", "condition"])
    paired["checkpoint"] = pandas.Categorical(paired["checkpoint"], CHECKPOINTS, ordered=True)
    summary = paired.pivot_table(
        index=["kind", "number_of_senses", "checkpoint"],
        columns="condition",
        values=["context_effect", "both_correct", "donor_share_donor_cue"],
        aggfunc="mean",
        observed=True,
    )
    lines.append(summary.round(2).to_string())
    return "\n\n".join(lines)


if __name__ == "__main__":
    text = context_summary() + "\n\n" + checkpoint_summary() + "\n"
    (DIRECTORY / "summary.txt").write_text(text, encoding="utf-8")
    print(text)
