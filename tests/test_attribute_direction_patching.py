"""属性の方向の成分の入れ替えを測る処理のテスト（極小の、無作為な重みのモデルで）。"""

from pathlib import Path

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

from lexical_acquisition_mechanisms.attribute_direction_patching import write_attribute_patching
from lexical_acquisition_mechanisms.attribute_probes import baseline_heads
from lexical_acquisition_mechanisms.component_adoption import ComponentClass
from lexical_acquisition_mechanisms.csv_tables import write_rows
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.sense_accuracy import SenseAccuracy
from lexical_acquisition_mechanisms.settings import ModelSource
from lexical_acquisition_mechanisms.task_prompts import TaskItem, prompt_template
from lexical_acquisition_mechanisms.word_selection import SelectedWord
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

CHOICES = {"sense": ("vehicle", "fruit"), "category": ("food", "vehicle")}
ANSWERS = {("sense", "donor"): "fruit", ("sense", "target"): "vehicle"}
ANSWERS |= {("category", "donor"): "food", ("category", "target"): "vehicle"}


def test_attribute_directions_are_patched_with_two_baselines(
    tmp_path: Path, tiny_model_source: ModelSource
) -> None:
    tokenizer = WordTokenizer.from_model(tiny_model_source)
    classes = [ComponentClass(0, 1, "none", "adopted", "adopted", "category")]
    write_rows(tmp_path / "classes.csv", ComponentClass, classes)
    baseline = baseline_heads([(0, 1)], 4, 0)[0, 1]
    random_numbers = np.random.default_rng(0)
    arrays = {}
    for head in (1, baseline[1]):
        for attribute in ("category", "part-of-speech"):
            values = {
                "weights": random_numbers.normal(size=(2, 8)),
                "biases": np.zeros(2),
                "mean": np.zeros(8),
                "scale": np.ones(8),
            }
            arrays |= {f"0-{head}-{attribute}-{name}": value for name, value in values.items()}
    np.savez(tmp_path / "probes.npz", **arrays)
    # 2つ目の最小対は2文が同じなので、入れ替えても変わらない
    pairs = {
        "different": ("She ate the {word}.", "She drove the {word}."),
        "same": ("She saw the {word}.", "She saw the {word}."),
    }
    questions = [
        {
            "kind": "other-category",
            "number_of_senses": 4,
            "target": "car",
            "donor": "fruit",
            "donor_sentence": name,
            "cue": cue,
            "task": task,
            "prompt_template": prompt_template(
                [], TaskItem(task, sentence, CHOICES[task], ANSWERS[task, cue])
            ),
            "choices": list(CHOICES[task]),
            "answer": ANSWERS[task, cue],
        }
        for name, sentences in pairs.items()
        for cue, sentence in zip(("donor", "target"), sentences, strict=True)
        for task in ("sense", "category")
    ]
    parquet.write_table(arrow.Table.from_pylist(questions), tmp_path / "questions.parquet")
    accuracy = [
        SenseAccuracy("other-category", 4, "car", sense, 2, 1.0, 0.0, 1.0, 0.5, 0.01, True)
        for sense in ("fruit", "original")
    ]
    write_rows(tmp_path / "accuracy.csv", SenseAccuracy, accuracy)
    car = " ".join(map(str, tokenizer.word_token_ids("car")))
    write_rows(tmp_path / "words.csv", SelectedWord, [SelectedWord("car", "target", token_ids=car)])
    directory = Path(tiny_model_source.repository)
    runs = ("same-category-4", "other-category-4", "adjective-4")

    write_attribute_patching(
        {run: directory for run in runs},
        lambda source: LanguageModel.load(source, device="cpu"),
        tuple(
            tmp_path / name
            for name in (
                "classes.csv",
                "probes.npz",
                "questions.parquet",
                "accuracy.csv",
                "words.csv",
            )
        ),
        tmp_path / "patching.parquet",
        0,
    )

    rows = parquet.read_table(tmp_path / "patching.parquet").to_pylist()
    # 使えない課題2（カテゴリ）は測らない
    only_sense = write_attribute_patching(
        {"other-category-4": directory},
        lambda source: LanguageModel.load(source, device="cpu"),
        tuple(
            tmp_path / name
            for name in (
                "classes.csv",
                "probes.npz",
                "questions.parquet",
                "accuracy.csv",
                "words.csv",
            )
        ),
        tmp_path / "only_sense.parquet",
        0,
        {"sense", "part-of-speech"},
    )
    assert {row["task"] for row in only_sense} == {"sense"}
    assert len(only_sense) * 2 == len(rows)
    # 2組 × 2方向 × 2課題 × 2属性 × 3つの入れ替え。問題があるのは別カテゴリの run だけ
    assert len(rows) == 2 * 2 * 2 * 2 * 3
    assert {row["run"] for row in rows} == {"other-category-4"}
    patched = {(row["patch"], row["patched_head"]) for row in rows}
    assert patched == {("attribute", 1), ("random-head", baseline[1]), ("random-direction", 1)}
    assert {row["dimensions"] for row in rows} == {1}
    same = [row["metric_change"] for row in rows if row["donor_sentence"] == "same"]
    assert same == pytest.approx([0.0] * len(same), abs=1e-5)
    different = [row["metric_change"] for row in rows if row["donor_sentence"] == "different"]
    assert any(abs(value) > 1e-6 for value in different)
