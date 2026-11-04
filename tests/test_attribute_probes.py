"""属性のプローブを学習し、確かめる処理のテスト（極小の、無作為な重みのモデルで）。"""

from pathlib import Path

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet

from lexical_acquisition_mechanisms.attribute_probes import (
    AttributeProbeAccuracy,
    write_attribute_probes,
)
from lexical_acquisition_mechanisms.component_adoption import ComponentClass
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.settings import ModelSource
from lexical_acquisition_mechanisms.task_prompts import TaskItem, prompt_template
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

NATURAL_WORDS = {
    "category": {
        "food": ["soup", "bread", "rice", "apple", "cake"],
        "vehicle": ["bus", "train", "truck", "boat", "plane"],
    },
    "part-of-speech": {
        "noun": ["table", "river", "chair", "house", "stone"],
        "adjective": ["quiet", "happy", "green", "small", "cold"],
    },
}


def question(task: str, sentence: str, choices: tuple[str, ...], answer: str) -> str:
    return prompt_template([], TaskItem(task, sentence, choices, answer))


def test_probes_are_trained_on_natural_words(
    tmp_path: Path, tiny_model_source: ModelSource
) -> None:
    tokenizer = WordTokenizer.from_model(tiny_model_source)
    classes = [
        ComponentClass(0, 1, "none", "adopted", "adopted", "category"),
        ComponentClass(1, -1, "adopted", "adopted", "adopted", "sense-selection"),
    ]
    write_rows(tmp_path / "classes.csv", ComponentClass, classes)
    natural = [
        {
            "task": task,
            "word": word,
            "prompt_template": question(task, sentence, tuple(answers), answer),
            "word_token_ids": tokenizer.word_token_ids(word),
            "choices": list(answers),
            "answer": answer,
        }
        for task, answers in NATURAL_WORDS.items()
        for answer, members in answers.items()
        for word in members
        for sentence in ("I saw the {word} today.", "The {word} was there.")
    ]
    parquet.write_table(arrow.Table.from_pylist(natural), tmp_path / "natural.parquet")
    write_attribute_probes(
        tiny_model_source,
        lambda source: LanguageModel.load(source, device="cpu"),
        (tmp_path / "classes.csv", tmp_path / "natural.parquet"),
        (tmp_path / "probes.npz", tmp_path / "probe_accuracy.csv"),
        ["food", "vehicle"],
        0,
    )

    # MLP は除き、head（0, 1）だけ。2クラスとも、5語のうち1語（2文）を確認用にする
    rows = read_rows(tmp_path / "probe_accuracy.csv", AttributeProbeAccuracy)
    assert [(row.layer, row.head, row.attribute) for row in rows] == [
        (0, 1, "category"),
        (0, 1, "part-of-speech"),
    ]
    assert all(row.held_out_questions == 4 and row.chance == 0.5 for row in rows)
    probes = np.load(tmp_path / "probes.npz")
    assert probes["0-1-category-weights"].shape == (2, 8)
