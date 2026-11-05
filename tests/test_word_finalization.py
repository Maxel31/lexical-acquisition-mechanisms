"""確認の結果から語の一覧を書く処理のテスト。"""

import csv
from pathlib import Path

import pytest

from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.hidden_state_similarity import ReviewRow
from lexical_acquisition_mechanisms.word_finalization import build_final_words
from lexical_acquisition_mechanisms.word_selection import ExcludedWord, SelectedWord

WORDS = [
    SelectedWord("bread", "noun", category="food"),
    SelectedWord("bagel", "target", category="food"),
    SelectedWord("<new_01>", "unknown"),
]


def finalize(tmp_path: Path, approved: dict[str, str]) -> Path:
    write_rows(tmp_path / "selected_words.csv", SelectedWord, WORDS)
    review = [
        ReviewRow(word, role, "food", "", "", approved=value)
        for (word, role), value in zip(
            [("bread", "noun"), ("bagel", "target")], approved.values(), strict=True
        )
    ]
    write_rows(tmp_path / "review.csv", ReviewRow, review)
    output_path = tmp_path / "words.csv"
    build_final_words(
        tmp_path / "selected_words.csv",
        tmp_path / "review.csv",
        tmp_path / "excluded_words.csv",
        output_path,
    )
    return output_path


def test_approved_words_are_written_with_unknown_words(tmp_path: Path) -> None:
    output_path = finalize(tmp_path, {"bread": "1", "bagel": " 1 "})

    with output_path.open(encoding="utf-8") as file:
        assert [row["word"] for row in csv.DictReader(file)] == ["bread", "bagel", "<new_01>"]


def test_rejected_words_are_excluded_once_and_selection_is_rerun(tmp_path: Path) -> None:
    for _ in range(2):
        with pytest.raises(CommandStopped, match="approved が 0 の語（bread）") as stopped:
            finalize(tmp_path, {"bread": "0", "bagel": "1"})
        assert stopped.value.rerun_from == "words select"

    excluded = read_rows(tmp_path / "excluded_words.csv", ExcludedWord)
    assert [(row.word, row.role) for row in excluded] == [("bread", "noun")]
    assert not (tmp_path / "words.csv").exists()


def test_unfilled_rows_stop(tmp_path: Path) -> None:
    with pytest.raises(CommandStopped, match="1・0 でない行が 1 行あります（例: bagel）"):
        finalize(tmp_path, {"bread": "1", "bagel": ""})


def test_review_file_must_match_the_selected_words(tmp_path: Path) -> None:
    write_rows(tmp_path / "selected_words.csv", SelectedWord, WORDS)
    write_rows(tmp_path / "review.csv", ReviewRow, [ReviewRow("bread", "noun", "", "", "", "1")])

    with pytest.raises(CommandStopped, match="選んだ語と合いません") as stopped:
        build_final_words(
            tmp_path / "selected_words.csv",
            tmp_path / "review.csv",
            tmp_path / "excluded_words.csv",
            tmp_path / "words.csv",
        )
    assert stopped.value.rerun_from == "words measure-hidden-state-similarity"
