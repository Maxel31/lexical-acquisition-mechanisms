"""フェーズの学習で変わった重みを部品ごとに移して測る処理のテスト。"""

import math
from pathlib import Path

import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest
import torch
from transformers import AutoModelForCausalLM

from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.phase_weight_grafting import (
    GraftingSummary,
    summarize,
    write_grafting,
)
from lexical_acquisition_mechanisms.second_sense_assignments import SecondSenseAssignment
from lexical_acquisition_mechanisms.settings import ModelSource
from lexical_acquisition_mechanisms.weight_grafting import component_sets
from lexical_acquisition_mechanisms.word_selection import SelectedWord


def row(configuration: str, mode: str, cue: str, second: float, token_count: int = 1) -> dict:
    return {
        "run": "run-1",
        "configuration": configuration,
        "mode": mode,
        "unknown_word": f"word{token_count}",
        "token_count": token_count,
        "kind": "other-category",
        "second_exposure": 1000,
        "first_noun": "cola",
        "second_noun": "car",
        "pair": 0,
        "cue": cue,
        "choices": ["cola", "car"],
        "log_probabilities": [math.log(1 - second), math.log(second)],
    }


def test_recovered_is_the_share_of_the_change_from_before_to_after() -> None:
    rows = []
    # 文脈の効果: before 0.0、after 0.4、mlp-0 だけを足すと 0.3
    for configuration, mode, effect in (
        ("before", "baseline", 0.0),
        ("after", "baseline", 0.4),
        ("mlp-0", "sufficiency", 0.3),
    ):
        rows += [
            row(configuration, mode, "first", 0.3),
            row(configuration, mode, "second", 0.3 + effect),
        ]

    table = {
        (item.configuration, item.token_count): item
        for item in summarize("second-sense-addition", rows)
    }

    assert table[("mlp-0", "all")].value == pytest.approx(0.3)
    assert table[("mlp-0", "all")].recovered == pytest.approx(0.75)
    assert table[("mlp-0", "1")].recovered == pytest.approx(0.75)
    assert table[("after", "all")].recovered == pytest.approx(1.0)


def test_first_phase_uses_the_share_of_x_at_the_cue_for_x() -> None:
    rows = [row("before", "baseline", "first", 0.6), row("before", "baseline", "second", 0.9)]
    rows += [row("after", "baseline", "first", 0.2), row("after", "baseline", "second", 0.9)]

    table = {item.configuration: item for item in summarize("unknown-word-acquisition", rows)}

    assert table["before"].value == pytest.approx(0.4)
    assert table["after"].value == pytest.approx(0.8)


def test_every_configuration_answers_the_questions_of_the_run(
    tmp_path: Path, tiny_model_source: ModelSource
) -> None:
    words = [
        SelectedWord("cola", "noun", token_count=1),
        SelectedWord("car", "noun", token_count=1),
        SelectedWord("flaid", "unknown", token_count=1, token_ids="100300"),
        SelectedWord("blick", "unknown", token_count=1, token_ids="100301"),
    ]
    write_rows(tmp_path / "words.csv", SelectedWord, words)
    assignments = [
        SecondSenseAssignment(
            "run-1", word, 1, "cola", "food", 3000, "other-category", "car", "vehicle", exposure
        )
        # 10 回の未知語は、2つ目の語義の追加では測らない
        for word, exposure in (("flaid", 1000), ("blick", 10))
    ]
    write_rows(tmp_path / "assignments.csv", SecondSenseAssignment, assignments)
    question = "\nQuestion: the word {word} means: cola / car\nAnswer:"
    questions = [
        {
            "first_noun": "cola",
            "second_noun": "car",
            "kind": "other-category",
            "pair": 0,
            "cue": cue,
            # 2 文の違いは手がかりだけ（同じ文だと、文脈の効果が前後とも 0 になる）
            "prompt_template": f"Sentence: We {verb} the {{word}}.{question}",
            "choices": ["cola", "car"],
            "answer": "car",
        }
        for cue, verb in (("first", "drank"), ("second", "drove"))
    ]
    parquet.write_table(arrow.Table.from_pylist(questions), tmp_path / "questions.parquet")
    after = AutoModelForCausalLM.from_pretrained(tiny_model_source.repository)
    with torch.no_grad():
        for parameter in after.parameters():
            parameter.mul_(1.1)
    after.save_pretrained(tmp_path / "after")
    inputs = (tmp_path / "words.csv", tmp_path / "assignments.csv", tmp_path / "questions.parquet")
    configurations = 2 + 2 * len(component_sets(2))

    for phase, words_measured in (("second-sense-addition", 1), ("unknown-word-acquisition", 2)):
        outputs = (tmp_path / phase / "grafting.parquet", tmp_path / phase / "grafting.csv")
        write_grafting(
            phase,
            {"run-1": (tiny_model_source, tmp_path / "after")},
            lambda source: AutoModelForCausalLM.from_pretrained(source.repository),
            inputs,
            outputs,
        )

        rows = parquet.read_table(outputs[0]).to_pylist()
        assert len(rows) == configurations * words_measured * 2
        summary = read_rows(outputs[1], GraftingSummary)
        after_rows = [item for item in summary if item.configuration == "after"]
        assert all(item.recovered == pytest.approx(1.0) for item in after_rows)
