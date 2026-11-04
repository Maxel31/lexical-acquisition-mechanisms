"""未知語に付けた意味と既知語に足した意味の、head の地図の比較のテスト。"""

from pathlib import Path

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

from lexical_acquisition_mechanisms import head_map_comparison
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.head_map_comparison import (
    HeadMapComparison,
    _spearman,
    write_head_maps,
)
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.settings import ModelSource
from lexical_acquisition_mechanisms.task_prompts import TaskItem, prompt_template
from lexical_acquisition_mechanisms.unknown_word_assignments import UnknownWordAssignment
from lexical_acquisition_mechanisms.word_selection import SelectedWord

CHOICES = ("pasta", "car", "lamp", "cup")


def test_rank_correlation() -> None:
    values = np.array([0.1, 0.5, 0.3, 0.9])

    assert _spearman(values, values * 2) == pytest.approx(1.0)
    assert _spearman(values, -values) == pytest.approx(-1.0)


def test_maps_of_the_same_noun_are_compared(
    tmp_path: Path, tiny_model_source: ModelSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(head_map_comparison, "TOP_HEADS", 2)
    questions = [
        {
            "run": "run-1",
            "unknown_word": "blicket",
            "noun": "pasta",
            "condition": number,
            "task": "sense",
            "prompt_template": prompt_template([], TaskItem("sense", sentence, CHOICES, "pasta")),
            "choices": list(CHOICES),
            "answer": "pasta",
        }
        for number, sentence in enumerate(("{word}.", "Do you know about the {word}."))
    ]
    parquet.write_table(arrow.Table.from_pylist(questions), tmp_path / "questions.parquet")
    # 1トークンの未知語（学習していない番号）。露出0回の文字列は、同じ run・トークン数・綴りの近さ
    assignments = [
        UnknownWordAssignment("run-1", "blicket", 1, "near", 1, 0, "pasta", "food", 10000),
        UnknownWordAssignment("run-1", "zorbat", 1, "near", 1, 1, "car", "vehicle", 0),
    ]
    write_rows(tmp_path / "assignments.csv", UnknownWordAssignment, assignments)
    words = [
        SelectedWord("blicket", "unknown", token_ids="100300"),
        SelectedWord("zorbat", "unknown", token_ids="100301"),
    ]
    write_rows(tmp_path / "words.csv", SelectedWord, words)
    pairs = [
        {
            "run": run,
            "kind": kind,
            "number_of_senses": 1,
            "target": "car",
            "donor": donor,
            "donor_sentence": "sentence",
            "direction": direction,
            "total_effect": 1.0,
            "usable": True,
        }
        for kind, donor, runs in (
            ("other-category", "pasta", ("other-category-1", "noun-control")),
            ("adjective", "quiet", ("adjective-1", "adjective-control")),
        )
        for run in runs
        for direction in ("donor-to-target", "target-to-donor")
    ]
    parquet.write_table(arrow.Table.from_pylist(pairs), tmp_path / "pairs.parquet")
    effects = np.random.default_rng(0).normal(size=(len(pairs), 10)).astype(np.float32)
    np.savez(
        tmp_path / "effects.npz",
        target_position=effects,
        after_target_positions=effects,
        layers=np.repeat([0, 1], 5),
        heads=np.tile([0, 1, 2, 3, -1], 2),
    )

    write_head_maps(
        {"run-1": Path(tiny_model_source.repository)},
        lambda source: LanguageModel.load(source, device="cpu"),
        [
            tmp_path / name
            for name in (
                "questions.parquet",
                "assignments.csv",
                "words.csv",
                "pairs.parquet",
                "effects.npz",
            )
        ],
        (tmp_path / "maps.npz", tmp_path / "comparison.csv"),
        10000,
    )

    maps = np.load(tmp_path / "maps.npz")
    assert list(maps["nouns"]) == ["pasta"]
    # MLP を除いた 2層 × 4 head
    assert maps["new_form"].shape == maps["existing_form"].shape == (1, 8)
    assert np.any(maps["new_form"] != 0)
    # 既存の形の地図は、pasta が donor の組（語義を足した run の2行）から、対照の2行を引いた平均
    expected = effects[[0, 1]].mean(axis=0) - effects[[2, 3]].mean(axis=0)
    heads = np.tile([0, 1, 2, 3, -1], 2) != -1
    np.testing.assert_allclose(maps["existing_form"][0], expected[heads], rtol=1e-5)
    rows = read_rows(tmp_path / "comparison.csv", HeadMapComparison)
    assert [(row.noun, row.unknown_word, row.chance_overlap) for row in rows] == [
        ("pasta", "blicket", 0.25)
    ]
    assert -1 <= rows[0].spearman <= 1
