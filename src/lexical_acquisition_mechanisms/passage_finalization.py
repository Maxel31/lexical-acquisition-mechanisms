"""判定の精度を確かめ、合格した用例を、学習用・評価用の用例として書き出す（passages finalize）。

1. 語と分け方（学習用・評価用）ごとに、合格した用例が必要な数あるかを確かめる（足りない語は、
   精度を確かめずに外すので、先に確かめる）。足りない語があれば止まる（passage_shortage）。
   足りないのが自然な語だけで、外しても数の下限を下回らなければ、その語の用例を書かずに続ける。
2. 人が確認用の CSV（passages sample-for-review が書く data/passages/review.csv）の approved 列に
   記入した値から、語ごとに LLM の判定の精度（合格とした用例のうち、人も 1 とした割合）を出す。
   精度が下限に届かない語があれば止まる（指示文を直して、passages judge-word-sense からやり直す）。
3. 合格した用例を data/passages/passages.parquet に、語ごとの精度と数を
   data/passages/passage_summary.csv に書く。
"""

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pyarrow as arrow
import pyarrow.compute as compute
import pyarrow.parquet as parquet

from lexical_acquisition_mechanisms import (
    passage_collection,
    passage_review,
    passage_shortage,
    word_finalization,
    word_selection,
    word_sense_judgment,
)
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.output_records import (
    Step,
    edited_file_settings,
    run_unless_unchanged,
)
from lexical_acquisition_mechanisms.passage_review import PassageReviewRow
from lexical_acquisition_mechanisms.settings import Settings
from lexical_acquisition_mechanisms.word_selection import SelectedWord

ACTION = "passages finalize"
# 実行し直すときに示すコマンド（確認用の CSV は、書き出された場所のもの）
COMMAND = f"{ACTION} --review-csv {passage_review.REVIEW_PATH}"
PASSAGES_PATH = Path("data/passages/passages.parquet")
SUMMARY_PATH = Path("data/passages/passage_summary.csv")


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
        f"{ACTION} --review-csv {review_path}",
        inputs={
            word_finalization.OUTPUT_PATH: word_finalization.COMMAND,
            passage_collection.SUMMARY_PATH: passage_collection.COMMAND,
            word_sense_judgment.JUDGMENTS_PATH: word_sense_judgment.COMMAND,
        },
        outputs=[PASSAGES_PATH, SUMMARY_PATH],
        settings={
            "review_csv": edited_file_settings(review_path, passage_review.COMMAND),
            "passages": settings.passages.model_dump(
                include={"evaluation_passages_per_word", "minimum_judgment_precision"}
            ),
            "unknown_word_acquisition": settings.unknown_word_acquisition.model_dump(
                include={"exposure_levels"}
            ),
            "sense_addition": settings.sense_addition.model_dump(
                include={"passages_per_added_sense"}
            ),
            "evaluation": settings.evaluation.model_dump(include={"passages_per_natural_word"}),
            "word_selection": settings.word_selection.model_dump(
                include={"minimum_natural_nouns_per_category", "minimum_natural_adjectives"}
            ),
        },
    )
    run_unless_unchanged(
        step,
        lambda: build_passages(
            settings,
            word_finalization.OUTPUT_PATH,
            passage_collection.SUMMARY_PATH,
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
    collection_summary_path: Path,
    judgments_path: Path,
    review_path: Path,
    excluded_words_path: Path,
    output_paths: tuple[Path, Path],
) -> None:
    passages_path, summary_path = output_paths
    words = [word for word in read_rows(words_path, SelectedWord) if word.role != "unknown"]
    judgments = parquet.read_table(judgments_path)
    passed = judgments.filter(judgments.column("passed")).drop_columns(["passed", "reasons"])
    counts = Counter(
        zip(passed.column("word").to_pylist(), passed.column("split").to_pylist(), strict=True)
    )
    if shortages := passage_shortage.find_shortages(words, counts, settings):
        # 足りないのが自然な語だけなら、その語の用例を書かずに続ける
        dropped = passage_shortage.drop_natural_words_or_stop(
            shortages,
            words,
            "判定に合格した用例",
            (collection_summary_path, excluded_words_path),
            settings,
        )
        passed = passed.filter(
            compute.invert(compute.is_in(passed.column("word"), arrow.array(sorted(dropped))))
        )
    review = read_rows(review_path, PassageReviewRow)
    rerun = f"{ACTION} --review-csv {review_path}"
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
