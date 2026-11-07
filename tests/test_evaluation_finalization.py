"""評価に使う最小対と、語義の追加の課題の問題を書き出す処理のテスト。"""

from pathlib import Path

import pyarrow.parquet as parquet
import pytest

from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.evaluation_finalization import write_final_evaluation_data
from lexical_acquisition_mechanisms.pretrained_model_checks import (
    CheckedMinimalPair,
    MinimalPairReviewRow,
)
from lexical_acquisition_mechanisms.sense_addition_assignments import SenseAdditionAssignment
from lexical_acquisition_mechanisms.settings import load_settings
from lexical_acquisition_mechanisms.task_questions import Paraphrase, TaskExample
from lexical_acquisition_mechanisms.word_selection import SelectedWord

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def pair(donor: str, kind: str, donor_sentence: str, target_sentence: str) -> CheckedMinimalPair:
    return CheckedMinimalPair(
        "car", "vehicle", 1, donor, "", kind, donor_sentence, target_sentence, "", "", 1.0, 1.0
    )


def finalize(
    tmp_path: Path,
    approvals: tuple[str, str],
    checked: list[CheckedMinimalPair] | None = None,
    paraphrase: str = "automobile",
) -> Path:
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    evaluation = settings.evaluation.model_copy(update={"number_of_task_examples": 1})
    settings = settings.model_copy(update={"evaluation": evaluation})
    words = [
        SelectedWord("car", "target", category="vehicle"),
        SelectedWord("pasta", "noun", category="food"),
        SelectedWord("quiet", "adjective"),
    ]
    write_rows(tmp_path / "words.csv", SelectedWord, words)
    assignments = [
        SenseAdditionAssignment(
            "other-category-1", "car", "vehicle", 1, "pasta", "noun", "food", 1
        ),
        SenseAdditionAssignment("adjective-1", "car", "vehicle", 1, "quiet", "adjective", "", 1),
    ]
    write_rows(tmp_path / "assignments.csv", SenseAdditionAssignment, assignments)
    checked = checked or [
        pair("pasta", "other-category", "She cooked the ___.", "She parked the ___."),
        pair("pasta", "other-category", "He ate the ___.", "He drove the ___."),
        pair("quiet", "adjective", "It was so ___.", "It was the ___."),
    ]
    write_rows(tmp_path / "checked.csv", CheckedMinimalPair, checked)
    review = [
        MinimalPairReviewRow(
            "car", "pasta", "", "", "", "She cooked the car.", "She parked the car.", approvals[0]
        ),
        MinimalPairReviewRow(
            "car", "quiet", "", "", "", "It was so car.", "It was the car.", approvals[1]
        ),
    ]
    write_rows(tmp_path / "review.csv", MinimalPairReviewRow, review)
    # 課題ごとに例を4つ置く（最小対の2文で、同じ例が選ばれるかを確かめるため）
    examples = [
        example
        for word in ("soup", "stew", "rice", "bread")
        for example in (
            TaskExample("sense", word, "We ate {word}.", "broth|cat", "broth"),
            TaskExample("category", word, "We ate {word}.", "food|animal", "food"),
            TaskExample("part-of-speech", word, "We ate {word}.", "noun|adjective", "noun"),
        )
    ]
    write_rows(tmp_path / "examples.csv", TaskExample, examples)
    write_rows(tmp_path / "paraphrases.csv", Paraphrase, [Paraphrase("car", "target", paraphrase)])
    inputs = [
        tmp_path / name
        for name in (
            "words.csv",
            "assignments.csv",
            "checked.csv",
            "examples.csv",
            "paraphrases.csv",
        )
    ]
    write_final_evaluation_data(
        settings,
        inputs,
        tmp_path / "review.csv",
        (tmp_path / "pairs.csv", tmp_path / "questions.parquet"),
    )
    return tmp_path


def test_rejected_pairs_are_removed_and_questions_are_built(tmp_path: Path) -> None:
    finalize(tmp_path, ("1", "0"))

    pairs = read_rows(tmp_path / "pairs.csv", CheckedMinimalPair)
    assert [pair.donor for pair in pairs] == ["pasta", "pasta"]
    questions = parquet.read_table(tmp_path / "questions.parquet").to_pylist()
    # 別カテゴリ N = 1 の2組 × 2文 × 2課題。形容詞の組は除いたので、問題がない
    assert len(questions) == 8
    assert {question["kind"] for question in questions} == {"other-category"}
    answers = {(question["cue"], question["task"]): question["answer"] for question in questions}
    assert answers == {
        ("donor", "sense"): "pasta",
        ("target", "sense"): "automobile",
        ("donor", "category"): "food",
        ("target", "category"): "vehicle",
    }
    sense = next(question for question in questions if question["task"] == "sense")
    assert sorted(sense["choices"]) == ["automobile", "pasta"]
    assert "the {word}." in sense["prompt_template"].split("\n\n")[-1]


def test_targets_without_a_paraphrase_get_no_sense_questions(tmp_path: Path) -> None:
    finalize(tmp_path, ("1", "0"), paraphrase="")

    questions = parquet.read_table(tmp_path / "questions.parquet").to_pylist()
    # 別カテゴリ N = 1 の2組 × 2文 × 課題2だけ
    assert len(questions) == 4
    assert {question["task"] for question in questions} == {"category"}


def test_review_rows_are_matched_with_both_sentences(tmp_path: Path) -> None:
    # donor の文が同じで、ターゲット語の文が違う2組。人が 0 にしたのは1組目（parked）
    checked = [
        pair("pasta", "other-category", "She cooked the ___.", "She parked the ___."),
        pair("pasta", "other-category", "She cooked the ___.", "She drove the ___."),
        pair("quiet", "adjective", "It was so ___.", "It was the ___."),
    ]

    finalize(tmp_path, ("0", "1"), checked)

    pairs = read_rows(tmp_path / "pairs.csv", CheckedMinimalPair)
    assert [pair.target_sentence for pair in pairs] == ["She drove the ___.", "It was the ___."]


def test_both_sentences_of_a_pair_share_examples_and_choices(tmp_path: Path) -> None:
    finalize(tmp_path, ("1", "0"))

    questions = parquet.read_table(tmp_path / "questions.parquet").to_pylist()
    pairs: dict[tuple[str, str], dict[str, dict]] = {}
    for question in questions:
        key = (question["donor_sentence"], question["task"])
        pairs.setdefault(key, {})[question["cue"]] = question
    assert len(pairs) == 4
    for pair in pairs.values():
        donor, target = pair["donor"], pair["target"]
        assert donor["choices"] == target["choices"]
        # 違うのは、最後の問題の文（手がかりの語）の行だけ
        donor_lines = donor["prompt_template"].split("\n")
        target_lines = target["prompt_template"].split("\n")
        differing = [
            index
            for index, (first, second) in enumerate(zip(donor_lines, target_lines, strict=True))
            if first != second
        ]
        assert differing == [len(donor_lines) - 3]


def test_unfilled_review_rows_stop(tmp_path: Path) -> None:
    with pytest.raises(CommandStopped, match="1・0 でない行が 1 行あります"):
        finalize(tmp_path, ("1", ""))
