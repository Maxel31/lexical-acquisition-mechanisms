"""2つ目の語義の追加の割り当て（assignments build --experiment second-sense-addition）。

未知語の獲得（unknown-word-acquisition）の各 run で、統合まで済んだ未知語（1つ目の露出回数が
second_sense_addition.minimum_first_exposure 以上）に、2つ目の語義の元の名詞（Y）と、2つ目の
露出回数を割り当てる。2つ目の語義の追加の run は、同じ名前の未知語の獲得の run の最後の重みから
学習し、学習データには、Y の用例の Y を未知語に置き換えたものと、1つ目の語義の元の名詞（X）の
用例を、同じ数だけ差し込む（1つ目の語義が多義として残るかを見るため）。
- 種類: Y が X と同じカテゴリの名詞（same-category）か、別のカテゴリの名詞（other-category）
- 2つ目の露出回数: second_exposure_levels（0 は、Y を足さず X の用例も入れない対照）
run の中の未知語を、1つ目の露出回数・X のカテゴリ・未知語の順に並べ、k 番目の未知語に、
水準を (k + run の番号) を水準の数で割った余りの番目、種類を (k ÷ 水準の数 + run の番号) の
偶奇で割り当てる。
どの run でも各水準の未知語の数が同じになるので、学習データの総トークン数がそろう。
Y は、種類に合うカテゴリの名詞のうち、次を満たすものから選ぶ。
- X と違う
- 残りの用例の数が、2つ目の露出回数以上。残りの用例の数は、名詞の学習用の用例の数の
  (1 − SPARE_FRACTION) 倍（置き換えられずに飛ばす用例の分を残す）から、その run で、その名詞を
  1つ目の語義にした未知語の露出回数と、すでに Y にした未知語の2つ目の露出回数を引いたもの
  （Y の用例は後ろから取るので、未知語の獲得で前から使った用例と重ならない）
その中で、その run で Y にした回数が少ない名詞から、seed から決まる順に選ぶ。2つ目の露出回数の
大きい未知語から、同じカテゴリの未知語を先に選ぶ（選べる名詞が少ないので）。
未知語の獲得の割り当てでは、run の中で、カテゴリと露出回数の水準がまとまっている（例: run-1 の建物の
名詞は 5000 回と 9900 回）ので、同じ名詞を、複数の未知語の Y にすることがある。
別のカテゴリは、X のカテゴリから 1〜（カテゴリの数 − 1）ずらしたカテゴリを、別のカテゴリの
未知語ごとに順に回して選ぶ（そのカテゴリに選べる名詞がなければ、次のずらし方にする）。
data/assignments/second-sense-addition/assignments.csv に書く。
"""

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import pyarrow.compute as compute
import pyarrow.parquet as parquet

from lexical_acquisition_mechanisms import (
    passage_finalization,
    unknown_word_assignments,
    word_finalization,
)
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.settings import SecondSenseAdditionSettings, Settings
from lexical_acquisition_mechanisms.unknown_word_assignments import UnknownWordAssignment
from lexical_acquisition_mechanisms.word_selection import SelectedWord

EXPERIMENT = "second-sense-addition"
COMMAND = f"assignments build --experiment {EXPERIMENT}"
OUTPUT_DIRECTORY = Path("data/assignments") / EXPERIMENT
ASSIGNMENTS_PATH = OUTPUT_DIRECTORY / "assignments.csv"
KINDS = ("same-category", "other-category")
# 名詞の学習用の用例のうち、置き換えられずに飛ばす分として、Y に使わずに残す割合
SPARE_FRACTION = 0.05


@dataclass
class SecondSenseAssignment:
    run: str
    unknown_word: str
    token_count: int
    # 1つ目の語義の元の名詞（未知語の獲得で割り当てた名詞）と、そのカテゴリ・露出回数
    first_noun: str
    first_category: str
    first_exposure: int
    # 2つ目の語義の種類と、元の名詞・そのカテゴリ・露出回数
    kind: str
    second_noun: str
    second_category: str
    second_exposure: int


def run_names() -> list[str]:
    """2つ目の語義の追加の run の名前（学習を始める未知語の獲得の run と同じ。run-1〜run-8）。"""
    return unknown_word_assignments.run_names()


def build_second_sense_assignments(settings: Settings, arguments: argparse.Namespace) -> None:
    step = Step(
        COMMAND,
        inputs={
            word_finalization.OUTPUT_PATH: word_finalization.COMMAND,
            unknown_word_assignments.ASSIGNMENTS_PATH: unknown_word_assignments.COMMAND,
            passage_finalization.PASSAGES_PATH: passage_finalization.COMMAND,
        },
        outputs=[ASSIGNMENTS_PATH],
        settings={
            **settings.second_sense_addition.model_dump(),
            "seed": settings.words.seed,
            "spare_fraction": SPARE_FRACTION,
        },
    )
    run_unless_unchanged(
        step,
        lambda: write_rows(
            ASSIGNMENTS_PATH,
            SecondSenseAssignment,
            second_sense_assignments(
                settings.second_sense_addition,
                settings.words.seed,
                read_rows(word_finalization.OUTPUT_PATH, SelectedWord),
                read_rows(unknown_word_assignments.ASSIGNMENTS_PATH, UnknownWordAssignment),
                training_passage_counts(passage_finalization.PASSAGES_PATH),
            ),
        ),
        arguments.recompute,
    )


def second_sense_assignments(
    section: SecondSenseAdditionSettings,
    seed: int,
    words: list[SelectedWord],
    first_assignments: list[UnknownWordAssignment],
    passage_counts: dict[str, int],
) -> list[SecondSenseAssignment]:
    nouns: dict[str, list[str]] = {}
    for word in words:
        if word.role == "noun":
            nouns.setdefault(word.category, []).append(word.word)
    levels = section.second_exposure_levels
    assignments = []
    for run_index, run in enumerate(run_names()):
        rows = [row for row in first_assignments if row.run == run]
        chosen = sorted(
            (row for row in rows if row.exposure >= section.minimum_first_exposure),
            key=lambda row: (row.exposure, row.noun_category, row.unknown_word),
        )
        if not chosen or len(chosen) % len(levels):
            raise CommandStopped(
                f"{run} の、1つ目の露出回数が {section.minimum_first_exposure} 回以上の未知語の数"
                f"（{len(chosen)}）が、2つ目の露出回数の水準の数（{len(levels)}）で割り切れません。"
                "settings.yaml の second_sense_addition を直してください。",
                rerun_from=COMMAND,
            )
        plan = [
            (
                row,
                KINDS[(index // len(levels) + run_index) % len(KINDS)],
                levels[(index + run_index) % len(levels)],
            )
            for index, row in enumerate(chosen)
        ]
        remaining = {
            noun: int(count * (1 - SPARE_FRACTION)) for noun, count in passage_counts.items()
        }
        for row in rows:
            remaining[row.noun] -= row.exposure
        uses = dict.fromkeys(remaining, 0)
        # 同じカテゴリの未知語、2つ目の露出回数の大きい未知語から選ぶ
        order = sorted(
            range(len(plan)), key=lambda index: (plan[index][1] != KINDS[0], -plan[index][2])
        )
        seconds: dict[int, tuple[str, str]] = {}
        random_order = random.Random(seed + run_index)
        for count, index in enumerate(order):
            row, kind, level = plan[index]
            noun, category = _second_noun(
                row, kind, level, count, nouns, remaining, uses, random_order
            )
            seconds[index] = (noun, category)
            remaining[noun] -= level
            uses[noun] += 1
        assignments += [
            SecondSenseAssignment(
                run,
                row.unknown_word,
                row.token_count,
                row.noun,
                row.noun_category,
                row.exposure,
                kind,
                *seconds[index],
                level,
            )
            for index, (row, kind, level) in enumerate(plan)
        ]
    return assignments


def training_passage_counts(passages_path: Path) -> dict[str, int]:
    """名詞ごとの、学習用の用例の数。"""
    table = parquet.read_table(passages_path, columns=["word", "role", "split"])
    table = table.filter(
        compute.and_(
            compute.equal(table.column("role"), "noun"),
            compute.equal(table.column("split"), "training"),
        )
    )
    counts = table.group_by("word").aggregate([("word", "count")])
    return dict(
        zip(counts.column("word").to_pylist(), counts.column("word_count").to_pylist(), strict=True)
    )


def _second_noun(
    row: UnknownWordAssignment,
    kind: str,
    level: int,
    count: int,
    nouns: dict[str, list[str]],
    remaining: dict[str, int],
    uses: dict[str, int],
    random_order: random.Random,
) -> tuple[str, str]:
    """未知語の2つ目の語義の名詞とカテゴリ。別のカテゴリは、count で、ずらす量を順に回す。"""
    categories = sorted(nouns)
    shifts = len(categories) - 1
    offsets = (
        [0]
        if kind == "same-category"
        else [(count + number) % shifts + 1 for number in range(shifts)]
    )
    for offset in offsets:
        category = categories[(categories.index(row.noun_category) + offset) % len(categories)]
        candidates = [
            noun
            for noun in sorted(nouns[category])
            if noun != row.noun and remaining.get(noun, 0) >= level
        ]
        if candidates:
            fewest = min(uses[noun] for noun in candidates)
            return random_order.choice([n for n in candidates if uses[n] == fewest]), category
    raise CommandStopped(
        f"{row.run} の {row.unknown_word}（{row.noun}）に、{kind} の2つ目の語義の名詞が"
        "選べません（用例が残っている名詞がありません）。settings.yaml の second_sense_addition."
        "second_exposure_levels を小さくしてください。",
        rerun_from=COMMAND,
    )
