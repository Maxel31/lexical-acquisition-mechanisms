"""確認の結果から、以降の手順で使う語の一覧を書く（words finalize）。

人が確認用の CSV（words measure-hidden-state-similarity が書く data/words/review.csv）の
approved 列に記入した値（語がその行のカテゴリ・語義に合えば 1、合わなければ 0）を読む。
- 1・0 でない値があれば、止まる。
- 0 の語があれば、data/words/excluded_words.csv に足して止まる。words select から実行し直すと、
  外した語の代わりの語が選ばれる。
- すべて 1 なら、words select が選んだ語（未知語を含む）を data/words/words.csv に書く。
"""

import argparse
from pathlib import Path

from lexical_acquisition_mechanisms import hidden_state_similarity, word_selection
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.hidden_state_similarity import REVIEWED_ROLES, ReviewRow
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.settings import Settings
from lexical_acquisition_mechanisms.word_selection import ExcludedWord, SelectedWord

COMMAND = "words finalize"
OUTPUT_PATH = Path("data/words/words.csv")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--review-csv",
        type=Path,
        required=True,
        help="approved 列に記入した確認用の CSV（data/words/review.csv）",
    )


def finalize_words(settings: Settings, arguments: argparse.Namespace) -> None:
    review_path: Path = arguments.review_csv
    step = Step(
        COMMAND,
        inputs={
            word_selection.OUTPUT_PATH: word_selection.COMMAND,
            review_path: hidden_state_similarity.COMMAND,
        },
        outputs=[OUTPUT_PATH],
    )
    run_unless_unchanged(
        step,
        lambda: build_final_words(
            word_selection.OUTPUT_PATH,
            review_path,
            word_selection.EXCLUDED_WORDS_PATH,
            OUTPUT_PATH,
        ),
        arguments.recompute,
    )


def build_final_words(
    selected_words_path: Path, review_path: Path, excluded_words_path: Path, output_path: Path
) -> None:
    words = read_rows(selected_words_path, SelectedWord)
    review = read_rows(review_path, ReviewRow)
    reviewed = {(word.word, word.role) for word in words if word.role in REVIEWED_ROLES}
    if {(row.word, row.role) for row in review} != reviewed:
        raise CommandStopped(
            f"{review_path} の語が、words select が選んだ語と合いません（選び直した後の確認用の"
            " CSV ではありません）。書き出し直して、確認し直してください。",
            rerun_from=hidden_state_similarity.COMMAND,
        )
    unfilled = [row.word for row in review if row.approved.strip() not in {"0", "1"}]
    if unfilled:
        raise CommandStopped(
            f"{review_path} の approved 列が 1・0 でない行が {len(unfilled)} 行あります"
            f"（例: {', '.join(unfilled[:5])}）。記入してください。",
            rerun_from=f"{COMMAND} --review-csv {review_path}",
        )
    rejected = [row for row in review if row.approved.strip() == "0"]
    if rejected:
        excluded = word_selection.read_excluded_words(excluded_words_path)
        known = {(row.word, row.role) for row in excluded}
        excluded += [
            ExcludedWord(row.word, row.role, f"{review_path} で approved を 0 にした")
            for row in rejected
            if (row.word, row.role) not in known
        ]
        write_rows(excluded_words_path, ExcludedWord, excluded)
        raise CommandStopped(
            f"approved が 0 の語（{', '.join(row.word for row in rejected)}）を、"
            f"{excluded_words_path} に足しました。words select から実行し直すと、"
            "代わりの語が選ばれます。",
            rerun_from=word_selection.COMMAND,
        )
    write_rows(output_path, SelectedWord, words)
