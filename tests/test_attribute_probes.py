"""属性のプローブを学習し、確かめる処理のテスト（極小の、無作為な重みのモデルで）。"""

from pathlib import Path

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet

from lexical_acquisition_mechanisms.attribute_probes import (
    AttributeProbeAccuracy,
    AttributeProbeTransfer,
    baseline_heads,
    word_level_p_value,
    write_attribute_probes,
    write_probe_transfer,
)
from lexical_acquisition_mechanisms.component_adoption import ComponentClass
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.sense_accuracy import SenseAccuracy
from lexical_acquisition_mechanisms.settings import ModelSource
from lexical_acquisition_mechanisms.task_prompts import TaskItem, prompt_template
from lexical_acquisition_mechanisms.word_selection import SelectedWord
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


def test_probes_are_trained_on_natural_words_and_applied_to_minimal_pairs(
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
    pairs = [
        ("other-category", "pasta", "category", ("She ate the {word}.", "She drove the {word}.")),
        ("adjective", "quiet", "part-of-speech", ("It was so {word}.", "It was the {word}.")),
    ]
    questions = []
    for kind, donor, task, sentences in pairs:
        for cue, sentence in zip(("donor", "target"), sentences, strict=True):
            answers = {"category": ("food", "vehicle"), "part-of-speech": ("adjective", "noun")}
            answer = answers[task][0] if cue == "donor" else answers[task][1]
            questions.append(
                {
                    "kind": kind,
                    "number_of_senses": 1,
                    "target": "car",
                    "donor": donor,
                    "cue": cue,
                    "task": task,
                    "prompt_template": question(task, sentence, answers[task], answer),
                    "answer": answer,
                }
            )
    parquet.write_table(arrow.Table.from_pylist(questions), tmp_path / "questions.parquet")
    # 形容詞の条件の、元の語義は使えないとする
    usable = [
        ("other-category", "pasta"),
        ("other-category", "original"),
        ("adjective", "quiet"),
    ]
    accuracy = [
        SenseAccuracy(kind, 1, "car", sense, 2, 1.0, 0.0, 1.0, 0.5, 0.01, True)
        for kind, sense in usable
    ]
    write_rows(tmp_path / "accuracy.csv", SenseAccuracy, accuracy)
    car = " ".join(map(str, tokenizer.word_token_ids("car")))
    write_rows(tmp_path / "words.csv", SelectedWord, [SelectedWord("car", "target", token_ids=car)])
    directory = Path(tiny_model_source.repository)
    checkpoints = {run: directory for run in ("other-category-1", "adjective-1", "noun-control")}

    def load_model(source: ModelSource) -> LanguageModel:
        return LanguageModel.load(source, device="cpu")

    probes = write_attribute_probes(
        tiny_model_source,
        load_model,
        (tmp_path / "classes.csv", tmp_path / "natural.parquet"),
        (tmp_path / "probes.npz", tmp_path / "probe_accuracy.csv"),
        ["food", "vehicle"],
        0,
    )
    write_probe_transfer(
        checkpoints,
        load_model,
        probes,
        [(0, 1)],
        ["food", "vehicle"],
        (tmp_path / "questions.parquet", tmp_path / "accuracy.csv", tmp_path / "words.csv"),
        tmp_path / "transfer.csv",
        0,
    )

    # MLP は除き、head（0, 1）と、比較の基準の同じ層の head。2クラスとも、5語のうち1語（2文）を
    # 確認用にする
    rows = read_rows(tmp_path / "probe_accuracy.csv", AttributeProbeAccuracy)
    adopted = [(row.layer, row.head, row.attribute) for row in rows if not row.baseline]
    assert adopted == [(0, 1, "category"), (0, 1, "part-of-speech")]
    baseline = {(row.layer, row.head) for row in rows if row.baseline}
    assert len(baseline) == 1 and baseline.pop() in {(0, 0), (0, 2), (0, 3)}
    assert all(row.held_out_questions == 4 and row.chance == 0.5 for row in rows)
    assert all(row.held_out_words == 2 for row in rows)
    probes = np.load(tmp_path / "probes.npz")
    assert probes["0-1-category-weights"].shape == (2, 8)
    transfer = read_rows(tmp_path / "transfer.csv", AttributeProbeTransfer)
    # 手がかりの種類ごとの行（形容詞の条件の、元の語義の文は、手順1で使えないので除く）
    assert [(row.run, row.attribute, row.cue, row.questions, row.targets) for row in transfer] == [
        ("other-category-1", "category", "donor", 1, 1),
        ("other-category-1", "category", "target", 1, 1),
        ("adjective-1", "part-of-speech", "donor", 1, 1),
    ]


def test_word_level_p_values_count_words_as_units() -> None:
    # 8語 × 10問。語の中の正誤がそろっていると、問題の数が多くても、語の数の分しか確かでない
    correct = np.repeat([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0], 10)
    words = [f"word-{index // 10}" for index in range(80)]

    by_word = word_level_p_value(correct, words, 0.5, np.random.default_rng(0))
    by_question = word_level_p_value(
        correct, [str(index) for index in range(80)], 0.5, np.random.default_rng(0)
    )

    assert by_question < 0.001 < by_word


def test_word_level_p_values_need_enough_words() -> None:
    # 語が少ないと、語の中の正誤が全てそろうと標準誤差が 0 になり p が 0 になるので、検定しない
    one_word = word_level_p_value(np.ones(20), ["car"] * 20, 1 / 6, np.random.default_rng(0))
    assert one_word == 1.0
    words = [f"word-{index // 4}" for index in range(20)]
    five_words = word_level_p_value(np.ones(20), words, 1 / 6, np.random.default_rng(0))
    assert five_words < 0.05


def test_baseline_heads_are_other_heads_of_the_same_layer() -> None:
    adopted = [(0, 1), (0, 2), (1, 0), (2, 0), (2, 1), (2, 2)]

    chosen = baseline_heads(adopted, 3, 0)

    assert chosen[0, 1] == chosen[0, 2] == (0, 0)
    assert chosen[1, 0] in {(1, 1), (1, 2)}
    # 層の head が全て採用されていれば、基準はない
    assert (2, 0) not in chosen
