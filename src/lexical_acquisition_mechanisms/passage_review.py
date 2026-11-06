"""判定に合格した用例を抜き出し、人が確かめる data/passages/review.csv を書く
（passages sample-for-review）。

名詞・形容詞・ターゲット語ごとに、合格した用例から、鍵の小さい順に REVIEWED_PASSAGES_PER_WORD 件を
抜き出す（鍵は位置と seed から決まるので、無作為に選んだのと同じ）。用例は、語の出現を [[ ]] で
囲んだ文字列にする。人は approved 列に、語がその行の語義で使われていれば 1、そうでなければ 0 を
記入する。
"""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow.compute as compute
import pyarrow.parquet as parquet

from lexical_acquisition_mechanisms import word_finalization, word_sense_judgment
from lexical_acquisition_mechanisms.csv_tables import read_rows, require_readable, write_review_rows
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.settings import Settings
from lexical_acquisition_mechanisms.word_selection import SelectedWord
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

COMMAND = "passages sample-for-review"
REVIEW_PATH = Path("data/passages/review.csv")
REVIEWED_ROLES = ("noun", "adjective", "target")
REVIEWED_PASSAGES_PER_WORD = 20


@dataclass
class PassageReviewRow:
    word: str
    role: str
    definition: str
    # 用例の出どころ（Pile の断片・行・出現の位置）
    shard: int
    row: int
    start: int
    # 語の出現を [[ ]] で囲んだ用例
    passage: str
    # 人が記入する（語がその行の語義で使われていれば 1、そうでなければ 0）
    approved: str = ""


def sample_for_review(settings: Settings, arguments: argparse.Namespace) -> None:
    require_readable(REVIEW_PATH)
    step = Step(
        COMMAND,
        inputs={
            word_finalization.OUTPUT_PATH: word_finalization.COMMAND,
            word_sense_judgment.JUDGMENTS_PATH: word_sense_judgment.COMMAND,
        },
        outputs=[REVIEW_PATH],
        settings={"tokenizer": next(iter(settings.models.values())).model_dump()},
    )
    run_unless_unchanged(
        step,
        lambda: build_review(
            settings,
            word_finalization.OUTPUT_PATH,
            word_sense_judgment.JUDGMENTS_PATH,
            REVIEW_PATH,
        ),
        arguments.recompute,
    )


def build_review(
    settings: Settings, words_path: Path, judgments_path: Path, review_path: Path
) -> None:
    words = {
        word.word: word
        for word in read_rows(words_path, SelectedWord)
        if word.role in REVIEWED_ROLES
    }
    tokenizer = WordTokenizer.from_model(next(iter(settings.models.values())))
    judgments = parquet.read_table(judgments_path)
    passed = judgments.filter(judgments.column("passed"))
    rows = []
    for word in words.values():
        of_word = passed.filter(compute.equal(passed.column("word"), word.word))
        sample = of_word.sort_by("key").slice(0, REVIEWED_PASSAGES_PER_WORD).to_pylist()
        rows += [
            PassageReviewRow(
                word.word,
                word.role,
                word.definition,
                judgment["shard"],
                judgment["row"],
                judgment["start"],
                marked_passage(
                    tokenizer, judgment["token_ids"], judgment["occurrence_positions"], word
                ),
            )
            for judgment in sample
        ]
    write_review_rows(review_path, PassageReviewRow, rows, ("word", "shard", "row", "start"))


def marked_passage(
    tokenizer: WordTokenizer, token_ids: Sequence[int], positions: Sequence[int], word: SelectedWord
) -> str:
    """語の全ての出現を [[ ]] で囲んだ用例（改行などの空白は、1つの空白にする）。"""
    pieces = []
    start = 0
    for position in positions:
        pieces.append(tokenizer.decode(token_ids[start:position]))
        pieces.append(f" [[{word.word}]]")
        start = position + word.token_count
    pieces.append(tokenizer.decode(token_ids[start:]))
    return " ".join("".join(pieces).split())
