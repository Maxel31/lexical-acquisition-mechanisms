"""2つ目の語義の追加の、課題1の文脈の効果を測る処理のテスト。"""

import json
import math
from pathlib import Path

import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.second_sense_assignments import SecondSenseAssignment
from lexical_acquisition_mechanisms.second_sense_context_effects import (
    ContextEffect,
    context_effects,
    saved_weights,
    write_context_effects,
)
from lexical_acquisition_mechanisms.settings import ModelSource
from lexical_acquisition_mechanisms.word_selection import SelectedWord


def row(word: str, exposure: int, pair: int, cue: str, second: float, weights: str = "final"):
    """Y の割合が second になる、問題 1 つの行（選択肢は X・Y の順）。"""
    return {
        "run": "run-1",
        "weights": weights,
        "unknown_word": word,
        "kind": "other-category",
        "second_exposure": exposure,
        "first_noun": "cola",
        "second_noun": "car",
        "pair": pair,
        "cue": cue,
        "choices": ["cola", "car"],
        "log_probabilities": [math.log(1 - second), math.log(second)],
    }


def test_context_effects_average_pairs_within_each_word_first() -> None:
    rows = [
        # flaid: 最小対 2 つ（文脈の効果 0.4 と 0.2）
        row("flaid", 1000, 0, "second", 0.7),
        row("flaid", 1000, 0, "first", 0.3),
        row("flaid", 1000, 1, "second", 0.5),
        row("flaid", 1000, 1, "first", 0.3),
        # blick: 最小対 1 つ（文脈の効果 0.0）
        row("blick", 1000, 2, "second", 0.4),
        row("blick", 1000, 2, "first", 0.4),
        row("dax", 0, 3, "second", 0.5, weights="before"),
        row("dax", 0, 3, "first", 0.5, weights="before"),
    ]

    effects = context_effects(rows, seed=0)

    assert [(effect.second_exposure, effect.weights, effect.words) for effect in effects] == [
        (0, "before", 1),
        (1000, "final", 2),
    ]
    final = effects[1]
    # 未知語ごとの平均（0.3 と 0.0）の平均
    assert final.context_effect == pytest.approx(0.15)
    assert final.second_share_at_second_cue == pytest.approx((0.6 + 0.4) / 2)
    assert final.lower <= final.context_effect <= final.upper


def test_every_weight_answers_the_questions_of_its_run(
    tmp_path: Path, tiny_model_source: ModelSource
) -> None:
    words = [
        SelectedWord("cola", "noun", token_count=1),
        SelectedWord("car", "noun", token_count=1),
        SelectedWord("flaid", "unknown", token_count=1, token_ids="100300"),
    ]
    write_rows(tmp_path / "words.csv", SelectedWord, words)
    assignments = [
        SecondSenseAssignment(
            "run-1", "flaid", 1, "cola", "food", 3000, "other-category", "car", "vehicle", 100
        )
    ]
    write_rows(tmp_path / "assignments.csv", SecondSenseAssignment, assignments)
    template = (
        "Sentence: We drove the {word}.\nQuestion: the word {word} means: cola / car\nAnswer:"
    )
    questions = [
        {
            "first_noun": "cola",
            "second_noun": "car",
            "kind": "other-category",
            "pair": 0,
            "cue": cue,
            "prompt_template": template,
            "choices": ["cola", "car"],
            "answer": "car",
        }
        for cue in ("first", "second")
    ]
    parquet.write_table(arrow.Table.from_pylist(questions), tmp_path / "questions.parquet")
    names = ("before", "checkpoint-50", "final")
    directories = {("run-1", name): Path(tiny_model_source.repository) for name in names}
    outputs = (tmp_path / "out" / "choices.parquet", tmp_path / "out" / "effects.csv")

    write_context_effects(
        directories,
        lambda source: LanguageModel.load(source, device="cpu"),
        (tmp_path / "words.csv", tmp_path / "assignments.csv", tmp_path / "questions.parquet"),
        outputs,
        seed=0,
    )

    choices = parquet.read_table(outputs[0]).to_pylist()
    assert [(row["weights"], row["cue"]) for row in choices] == [
        (name, cue) for name in names for cue in ("first", "second")
    ]
    assert all(len(row["log_probabilities"]) == 2 for row in choices)
    effects = read_rows(outputs[1], ContextEffect)
    assert [effect.weights for effect in effects] == list(names)


def test_saved_weights_follow_the_steps_of_the_training(tmp_path: Path) -> None:
    def record(steps: int) -> str:
        fingerprint = {"settings": {"checkpoint_fractions": [0.25, 0.5, 0.75]}}
        return json.dumps({"steps": steps, "fingerprint": fingerprint})

    (tmp_path / "completed.json").write_text(record(27), encoding="utf-8")
    assert saved_weights(tmp_path) == ["checkpoint-25", "checkpoint-50", "checkpoint-75", "final"]
    # 2 step の学習では、25% と 50% が同じ step になり、75% は最後の step になる
    (tmp_path / "completed.json").write_text(record(2), encoding="utf-8")
    assert saved_weights(tmp_path) == ["checkpoint-50", "final"]
