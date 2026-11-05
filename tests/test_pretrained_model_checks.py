"""学習前のモデルでの評価データの確認のテスト。モデルは、決まった対数確率を返す部品で置き換える。"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

from lexical_acquisition_mechanisms import pretrained_model_checks
from lexical_acquisition_mechanisms.bootstrap_statistics import SIGNIFICANCE_LEVEL
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.minimal_pairs import MinimalPairCandidate
from lexical_acquisition_mechanisms.pretrained_model_checks import (
    CheckedMinimalPair,
    CheckPaths,
    MinimalPairReviewRow,
    ParaphraseCheck,
    check_evaluation_data,
    require_usable_tasks,
    usable_tasks,
)
from lexical_acquisition_mechanisms.settings import ModelSource, load_settings
from lexical_acquisition_mechanisms.task_accuracy import TaskAccuracy, binomial_p_value
from lexical_acquisition_mechanisms.task_questions import Paraphrase, TaskExample
from lexical_acquisition_mechanisms.word_selection import SelectedWord
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# 語の最後のトークンの対数確率は、手がかりに合えば -1、合わなければ -3
FITS = {"pasta": ("cooked", "ate"), "car": ("parked", "drove")}


class ModelFollowingCues:
    """最小対の語は手がかりに合うほど出やすく、課題の答えは、選択肢のうち綴りの順で最初の語にする。"""

    def __init__(self, tokenizer: WordTokenizer) -> None:
        self.tokenizer = tokenizer

    def token_log_probabilities(self, inputs: Sequence[Sequence[int]]) -> list[np.ndarray]:
        results = []
        for token_ids in inputs:
            words = self.tokenizer.decode(token_ids).split()
            values = np.zeros(len(token_ids), dtype=np.float32)
            values[-1] = -1.0 if any(cue in words for cue in FITS[words[-1]]) else -3.0
            results.append(values)
        return results

    def next_token_log_probabilities(self, inputs, positions, candidate_ids) -> np.ndarray:
        return np.tile(-np.arange(len(candidate_ids), dtype=np.float32), (len(inputs), 1))


def candidate(donor_sentence: str, target_sentence: str) -> MinimalPairCandidate:
    return MinimalPairCandidate(
        "car",
        "vehicle",
        1,
        "pasta",
        "noun",
        "other-category",
        donor_sentence,
        target_sentence,
        "",
        "",
    )


def check(
    tmp_path: Path, tiny_model_source: ModelSource, candidates: list[MinimalPairCandidate]
) -> CheckPaths:
    """入力を書き、決まった対数確率を返すモデルで確かめる。"""
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    evaluation = settings.evaluation.model_copy(
        update={"minimal_pairs_per_sense_pair": 2, "number_of_task_examples": 1}
    )
    settings = settings.model_copy(
        update={"models": {"tiny": tiny_model_source}, "evaluation": evaluation}
    )
    tokenizer = WordTokenizer.from_model(tiny_model_source)
    paths = CheckPaths(
        *(
            tmp_path / name
            for name in [
                "words.csv",
                "candidates.csv",
                "questions.parquet",
                "examples.csv",
                "paraphrases.csv",
                "checked.csv",
                "review.csv",
                "accuracy.csv",
                "paraphrase_checks.csv",
            ]
        )
    )
    words = [
        SelectedWord(
            word,
            role,
            definition=f"the {word}",
            token_ids=" ".join(map(str, tokenizer.word_token_ids(word))),
        )
        for word, role in (("car", "target"), ("pasta", "noun"))
    ]
    write_rows(paths.words, SelectedWord, words)
    write_rows(paths.candidates, MinimalPairCandidate, candidates)
    template = "Sentence: The {word} is here.\\nQuestion: ?\\nAnswer:"
    questions = [
        {
            "task": "category",
            "word": "cat",
            "prompt_template": template,
            "word_token_ids": tokenizer.word_token_ids("cat"),
            "choices": ["food", "animal"],
            "answer": "animal",
        },
        {
            "task": "sense",
            "word": "cat",
            "prompt_template": template,
            "word_token_ids": tokenizer.word_token_ids("cat"),
            "choices": ["apple", "tiger"],
            "answer": "tiger",
        },
    ] * 10
    parquet.write_table(arrow.Table.from_pylist(questions), paths.natural_questions)
    write_rows(
        paths.examples,
        TaskExample,
        [TaskExample("sense", "soup", "We ate {word}.", "broth|cat", "broth")],
    )
    write_rows(paths.paraphrases, Paraphrase, [Paraphrase("car", "target", "automobile")])

    check_evaluation_data(settings, lambda source: ModelFollowingCues(tokenizer), paths)
    return paths


def test_evaluation_data_is_checked(tmp_path: Path, tiny_model_source: ModelSource) -> None:
    candidates = [
        candidate("She saw the ___ today.", "She met the ___ today."),
        candidate("She cooked the ___ today.", "She parked the ___ today."),
        candidate("He ate the ___ quickly.", "He drove the ___ quickly."),
        candidate("We cooked the ___ again.", "We parked the ___ again."),
    ]

    paths = check(tmp_path, tiny_model_source, candidates)

    # 手がかりが語義を決めない1組目は使わず、通った組を、語義の組ごとに2組まで残す
    checked = read_rows(paths.checked_pairs, CheckedMinimalPair)
    assert [pair.donor_sentence for pair in checked] == [
        "She cooked the ___ today.",
        "He ate the ___ quickly.",
    ]
    assert all(pair.donor_cue_effect == 2.0 and pair.target_cue_effect == 2.0 for pair in checked)
    (review,) = read_rows(paths.review, MinimalPairReviewRow)
    assert (review.donor_sentence, review.target_sentence) == (
        "She cooked the car today.",
        "She parked the car today.",
    )
    accuracy = {row.task: row for row in read_rows(paths.task_accuracy, TaskAccuracy)}
    # カテゴリは全て当たり（使える）、課題1は全て外れ（使えない）
    assert (accuracy["category"].accuracy, accuracy["category"].usable) == (1.0, True)
    assert (accuracy["sense"].accuracy, accuracy["sense"].usable) == (0.0, False)
    (paraphrase,) = read_rows(paths.paraphrase_checks, ParaphraseCheck)
    assert (paraphrase.sentences, paraphrase.paraphrase_share) == (2, 1.0)
    # 2文とも選んでも、偶然（0.5）でも 1/4 の確率で起きるので、偶然の水準を超えるとは言えない
    assert (paraphrase.p_value, paraphrase.above_chance) == (pytest.approx(0.25), False)


def test_sense_pairs_without_checked_minimal_pairs_stop(
    tmp_path: Path, tiny_model_source: ModelSource
) -> None:
    # 手がかりが語義を決めない候補しかない
    candidates = [candidate("She saw the ___ today.", "She met the ___ today.")]

    with pytest.raises(
        CommandStopped, match="1つもない語義の組があります（1 組: \\['car/pasta'\\]）"
    ):
        check(tmp_path, tiny_model_source, candidates)


def test_binomial_p_value() -> None:
    assert binomial_p_value(0, 10, 0.5) == pytest.approx(1.0)
    assert binomial_p_value(10, 10, 0.5) == pytest.approx(0.5**10)
    assert binomial_p_value(9, 10, 0.5) == pytest.approx(11 / 1024)


def test_commands_stop_on_tasks_that_the_model_cannot_solve(tmp_path: Path) -> None:
    rows = [
        TaskAccuracy("tiny", "sense", 40, 30, 0.75, 0.25, 0.001, True),
        TaskAccuracy("tiny", "category", 40, 8, 0.2, 1 / 6, 0.4, False),
        TaskAccuracy("other", "category", 40, 30, 0.75, 1 / 6, 0.001, True),
    ]
    write_rows(tmp_path / "task_accuracy.csv", TaskAccuracy, rows)
    path = tmp_path / "task_accuracy.csv"

    assert usable_tasks("tiny", path) == {"sense"}
    require_usable_tasks("tiny", ["sense"], "command", path)
    with pytest.raises(CommandStopped, match="category"):
        require_usable_tasks("tiny", ["sense", "category"], "command", path)
    # 表にないモデルと、表がないときは、check-with-pretrained-models から実行するように示す
    for model, table in (("absent", path), ("tiny", tmp_path / "missing.csv")):
        with pytest.raises(CommandStopped, match="正解率がありません") as stopped:
            require_usable_tasks(model, ["sense"], "command", table)
        assert stopped.value.rerun_from == "evaluation-data check-with-pretrained-models"


def test_record_has_the_significance_level(monkeypatch: pytest.MonkeyPatch) -> None:
    # 有意水準を変えたら、使える課題の判定をやり直す
    steps = []
    monkeypatch.setattr(
        pretrained_model_checks,
        "run_unless_unchanged",
        lambda step, compute, recompute: steps.append(step),
    )
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    pretrained_model_checks.check_with_pretrained_models(
        settings, argparse.Namespace(recompute=False)
    )
    assert steps[0].settings["significance_level"] == SIGNIFICANCE_LEVEL
