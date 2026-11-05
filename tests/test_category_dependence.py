"""カテゴリを判定する head への依存度を測る処理のテスト（極小の、無作為な重みのモデルで）。"""

from pathlib import Path

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet

from lexical_acquisition_mechanisms.attribute_judgments import AttributeJudgingHead
from lexical_acquisition_mechanisms.attribute_probes import baseline_heads
from lexical_acquisition_mechanisms.category_dependence import write_dependence
from lexical_acquisition_mechanisms.component_adoption import ComponentClass
from lexical_acquisition_mechanisms.csv_tables import write_rows
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.settings import ModelSource
from lexical_acquisition_mechanisms.task_prompts import TaskItem, prompt_template
from lexical_acquisition_mechanisms.unknown_word_assignments import UnknownWordAssignment
from lexical_acquisition_mechanisms.word_selection import SelectedWord
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

NAMES = ("judging.csv", "directions.npz", "classes.csv", "probes.npz", "questions.parquet")
NAMES += ("assignments.csv", "words.csv", "natural.parquet")


def write_natural_questions(path: Path, tokenizer: WordTokenizer, sentences: list[str]) -> None:
    """自然な語（cake）の課題2のカテゴリの問題（無効化で置き換える平均を求める問題）。"""
    rows = [
        {
            "word": "cake",
            "task": "category",
            "prompt_template": prompt_template(
                [], TaskItem("category", sentence, ("food", "vehicle"), "food")
            ),
            "word_token_ids": tokenizer.word_token_ids("cake"),
            "answer": "food",
        }
        for sentence in sentences
    ]
    parquet.write_table(arrow.Table.from_pylist(rows), path)


def write_inputs(tmp_path: Path, tokenizer: WordTokenizer, judging: bool) -> list[Path]:
    rows = [
        AttributeJudgingHead(0, 1, "category", 0.8, 0.7, 0.9, True, True, True, judging),
        # 品詞の head と、基準を満たさない head は使わない
        AttributeJudgingHead(0, 2, "part-of-speech", 0.8, 0.7, 0.9, True, True, True, True),
        AttributeJudgingHead(1, 0, "category", 0.8, 0.7, 0.9, True, False, True, False),
    ]
    write_rows(tmp_path / "judging.csv", AttributeJudgingHead, rows)
    random_numbers = np.random.default_rng(0)
    basis, _ = np.linalg.qr(random_numbers.normal(size=(8, 1)))
    np.savez(
        tmp_path / "directions.npz",
        **{"0-1-category-basis": basis, "0-1-category-mean": random_numbers.normal(size=8)},
    )
    classes = [ComponentClass(0, 1, "none", "adopted", "adopted", "category")]
    write_rows(tmp_path / "classes.csv", ComponentClass, classes)
    baseline = baseline_heads([(0, 1)], 4, 0)[0, 1]
    probe = {
        "weights": random_numbers.normal(size=(2, 8)),
        "biases": np.zeros(2),
        "mean": np.zeros(8),
        "scale": np.ones(8),
    }
    np.savez(
        tmp_path / "probes.npz",
        **{f"0-{baseline[1]}-category-{name}": value for name, value in probe.items()},
    )
    questions = [
        {
            "run": "run-1",
            "unknown_word": "blicket",
            "noun": "pasta",
            "condition": number,
            "task": task,
            "prompt_template": prompt_template(
                [], TaskItem(task, sentence, ("food", "vehicle"), "food")
            ),
            "choices": ["food", "vehicle"],
            "answer": "food",
        }
        for number, sentence in enumerate(("{word}.", "Do you know about the {word}."))
        for task in ("sense", "category")
    ]
    parquet.write_table(arrow.Table.from_pylist(questions), tmp_path / "questions.parquet")
    assignment = UnknownWordAssignment("run-1", "blicket", 2, "near", 1, 0, "pasta", "food", 10)
    write_rows(tmp_path / "assignments.csv", UnknownWordAssignment, [assignment])
    words = [
        SelectedWord(word, role, token_ids=" ".join(map(str, tokenizer.word_token_ids(word))))
        for word, role in (("blicket", "unknown"), ("pasta", "noun"))
    ]
    write_rows(tmp_path / "words.csv", SelectedWord, words)
    write_natural_questions(tmp_path / "natural.parquet", tokenizer, ["I ate the {word}."])
    return [tmp_path / name for name in NAMES]


def test_dependence_is_measured_for_each_ablation(
    tmp_path: Path, tiny_model_source: ModelSource
) -> None:
    tokenizer = WordTokenizer.from_model(tiny_model_source)
    inputs = write_inputs(tmp_path, tokenizer, judging=True)

    write_dependence(
        {"run-1": Path(tiny_model_source.repository)},
        lambda source: LanguageModel.load(source, device="cpu"),
        inputs,
        tmp_path / "dependence.parquet",
        0,
    )

    rows = parquet.read_table(tmp_path / "dependence.parquet").to_pylist()
    assert [row["ablation"] for row in rows] == [
        "category-direction",
        "random-head",
        "random-direction",
    ]
    for row in rows:
        assert (row["noun_category"], row["exposure"], row["conditions"]) == ("food", 10, 2)
        # 依存度と、使える組かは fit-regression で決める
        assert "dependence" not in row and "usable" not in row
    # 無効化で、対数確率が実際に変わる
    assert any(row["known_drop"] != 0 and row["unknown_drop"] != 0 for row in rows)
    # 置き換える平均は、その run の重みで、自然な語の問題から求める（問題を替えると変わる）
    write_natural_questions(inputs[-1], tokenizer, ["The {word} was here.", "We saw a {word}."])
    write_dependence(
        {"run-1": Path(tiny_model_source.repository)},
        lambda source: LanguageModel.load(source, device="cpu"),
        inputs,
        tmp_path / "other.parquet",
        0,
    )
    other = parquet.read_table(tmp_path / "other.parquet").to_pylist()
    assert other[0]["known_drop"] != rows[0]["known_drop"]


def test_nothing_is_measured_without_category_judging_heads(
    tmp_path: Path, tiny_model_source: ModelSource
) -> None:
    inputs = write_inputs(tmp_path, WordTokenizer.from_model(tiny_model_source), judging=False)

    def load_model(source: ModelSource) -> LanguageModel:
        raise AssertionError("モデルを読まない")

    write_dependence({"run-1": Path("unused")}, load_model, inputs, tmp_path / "out.parquet", 0)

    assert parquet.read_table(tmp_path / "out.parquet").num_rows == 0
