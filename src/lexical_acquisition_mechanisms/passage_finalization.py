"""判定の精度を確かめ、合格した用例を、学習用・評価用の用例として書き出す（passages finalize）。

1. 人が確認用の CSV（passages sample-for-review が書く data/passages/review.csv）の approved 列に
   記入した値から、語ごとに LLM の判定の精度（合格とした用例のうち、人も 1 とした割合）を出す。
   精度が下限に届かない語があれば止まる（指示文を直して、passages judge-word-sense からやり直す）。
2. 語と分け方（学習用・評価用）ごとに、合格した用例が必要な数あるかを確かめる。足りない語は、
   data/words/excluded_words.csv に足して止まる（words select から選び直すと、代わりの語が
   選ばれる）。
3. 合格した用例を data/passages/passages.parquet に、語ごとの精度と数を
   data/passages/passage_summary.csv に書く。
"""

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as parquet

from lexical_acquisition_mechanisms import (
    passage_review,
    word_finalization,
    word_selection,
    word_sense_judgment,
)
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.passage_review import PassageReviewRow
from lexical_acquisition_mechanisms.settings import Settings
from lexical_acquisition_mechanisms.word_selection import ExcludedWord, SelectedWord
from lexical_acquisition_mechanisms.word_sense_judgment import needed_passing_passages

COMMAND = "passages finalize"
PASSAGES_PATH = Path("data/passages/passages.parquet")
SUMMARY_PATH = Path("data/passages/passage_summary.csv")
SPLITS = ("training", "evaluation")


@dataclass
class PassageSummary:
    word: str
    role: str
    # LLM の判定の精度（確かめなかった語は空）
    precision: str
    training: int
    evaluation: int


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--review-csv",
        type=Path,
        required=True,
        help="approved 列に記入した確認用の CSV（data/passages/review.csv）",
    )


def finalize_passages(settings: Settings, arguments: argparse.Namespace) -> None:
    review_path: Path = arguments.review_csv
    step = Step(
        COMMAND,
        inputs={
            word_finalization.OUTPUT_PATH: word_finalization.COMMAND,
            word_sense_judgment.JUDGMENTS_PATH: word_sense_judgment.COMMAND,
            review_path: passage_review.COMMAND,
        },
        outputs=[PASSAGES_PATH, SUMMARY_PATH],
        settings={
            "passages": settings.passages.model_dump(
                include={"evaluation_passages_per_word", "minimum_judgment_precision"}
            ),
            "unknown_word_acquisition": settings.unknown_word_acquisition.model_dump(
                include={"exposure_levels"}
            ),
            "sense_addition": settings.sense_addition.model_dump(
                include={"passages_per_added_sense"}
            ),
            "evaluation": settings.evaluation.model_dump(include={"sentences_per_natural_word"}),
        },
    )
    run_unless_unchanged(
        step,
        lambda: build_passages(
            settings,
            word_finalization.OUTPUT_PATH,
            word_sense_judgment.JUDGMENTS_PATH,
            review_path,
            word_selection.EXCLUDED_WORDS_PATH,
            (PASSAGES_PATH, SUMMARY_PATH),
        ),
        arguments.recompute,
    )


def build_passages(
    settings: Settings,
    words_path: Path,
    judgments_path: Path,
    review_path: Path,
    excluded_words_path: Path,
    output_paths: tuple[Path, Path],
) -> None:
    passages_path, summary_path = output_paths
    words = [word for word in read_rows(words_path, SelectedWord) if word.role != "unknown"]
    review = read_rows(review_path, PassageReviewRow)
    rerun = f"{COMMAND} --review-csv {review_path}"
    reviewed = {word.word for word in words if word.role in passage_review.REVIEWED_ROLES}
    if {row.word for row in review} != reviewed:
        raise CommandStopped(
            f"{review_path} の語が、words finalize が書いた語と合いません。"
            "書き出し直して、確認し直してください。",
            rerun_from=passage_review.COMMAND,
        )
    unfilled = [row.word for row in review if row.approved.strip() not in {"0", "1"}]
    if unfilled:
        raise CommandStopped(
            f"{review_path} の approved 列が 1・0 でない行が {len(unfilled)} 行あります"
            f"（例: {', '.join(unfilled[:5])}）。記入してください。",
            rerun_from=rerun,
        )
    approvals = Counter(row.word for row in review if row.approved.strip() == "1")
    sizes = Counter(row.word for row in review)
    precision = {word: approvals[word] / sizes[word] for word in sizes}
    minimum = settings.passages.minimum_judgment_precision
    imprecise = {word: value for word, value in precision.items() if value < minimum}
    if imprecise:
        listed = ", ".join(f"{word}: {value:.2f}" for word, value in sorted(imprecise.items()))
        raise CommandStopped(
            f"LLM の判定の精度が {minimum} に届かない語があります（{listed}）。"
            "llm_prompts/ の判定の指示文（judge-noun-sense.md・judge-adjective-sense.md）を"
            "直してください。",
            rerun_from=word_sense_judgment.COMMAND,
        )
    judgments = parquet.read_table(judgments_path)
    passed = judgments.filter(judgments.column("passed")).drop_columns(["passed", "reasons"])
    counts = Counter(
        zip(passed.column("word").to_pylist(), passed.column("split").to_pylist(), strict=True)
    )
    short = []
    for word in words:
        for split in SPLITS:
            needed = needed_passing_passages(word.role, split, settings)
            if counts[(word.word, split)] < needed:
                short.append((word, split, counts[(word.word, split)], needed))
    if short:
        _exclude(excluded_words_path, short)
        listed = ", ".join(
            f"{word.word}（{split} {found} / {needed} 件）" for word, split, found, needed in short
        )
        raise CommandStopped(
            f"判定に合格した用例が足りない語を、{excluded_words_path} に足しました（{listed}）。"
            "words select から実行し直すと、代わりの語が選ばれます。",
            rerun_from=word_selection.COMMAND,
        )
    order = [("word", "ascending"), ("split", "ascending"), ("key", "ascending")]
    parquet.write_table(passed.sort_by(order), passages_path)
    summaries = [
        PassageSummary(
            word.word,
            word.role,
            f"{precision[word.word]:.2f}" if word.word in precision else "",
            counts[(word.word, "training")],
            counts[(word.word, "evaluation")],
        )
        for word in words
    ]
    write_rows(summary_path, PassageSummary, summaries)


def _exclude(excluded_words_path: Path, short: list) -> None:
    excluded = word_selection.read_excluded_words(excluded_words_path)
    known = {(row.word, row.role) for row in excluded}
    for word, split, found, needed in short:
        if (word.word, word.role) not in known:
            reason = f"判定に合格した用例が足りない（{split} {found} / {needed} 件）"
            excluded.append(ExcludedWord(word.word, word.role, reason))
            known.add((word.word, word.role))
    write_rows(excluded_words_path, ExcludedWord, excluded)
