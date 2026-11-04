"""中立文脈の選び方のテスト。モデルは、決まった対数確率を返す部品で置き換える。"""

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.csv_tables import write_rows
from lexical_acquisition_mechanisms.neutral_contexts import build_neutral_contexts
from lexical_acquisition_mechanisms.settings import ModelSource, Settings, load_settings
from lexical_acquisition_mechanisms.word_selection import SelectedWord
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOUNS = {"pasta": "food", "bread": "food", "hog": "animal", "cat": "animal"}
CANDIDATES = ["Do you know about the", "Let me cook the", "I heard about the"]


class ModelPreferringFoodAfterCooking:
    """名詞の最後のトークンの対数確率を、"Let me cook the" の後の食べ物は -1、ほかは -2 にする。"""

    def __init__(self, tokenizer: WordTokenizer) -> None:
        self.tokenizer = tokenizer

    def token_log_probabilities(self, inputs: Sequence[Sequence[int]]) -> list[np.ndarray]:
        results = []
        for token_ids in inputs:
            text = self.tokenizer.decode(token_ids)
            values = np.zeros(len(token_ids), dtype=np.float32)
            food = NOUNS[text.split()[-1]] == "food"
            values[-1] = -1.0 if text.startswith("Let me cook") and food else -2.0
            results.append(values)
        return results


@pytest.fixture
def settings(tiny_model_source: ModelSource) -> Settings:
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    words = settings.words.model_copy(update={"categories": settings.words.categories[:2]})
    evaluation = settings.evaluation.model_copy(
        update={
            "neutral_context_candidates": CANDIDATES,
            "number_of_neutral_contexts": 2,
            "maximum_neutral_context_deviation": 0.3,
        }
    )
    return settings.model_copy(
        update={"models": {"tiny": tiny_model_source}, "words": words, "evaluation": evaluation}
    )


def run(tmp_path: Path, settings: Settings) -> list[dict[str, str]]:
    tokenizer = WordTokenizer.from_model(settings.models["tiny"])
    words = [
        SelectedWord(
            word,
            "natural-noun",
            category=category,
            token_ids=" ".join(map(str, tokenizer.word_token_ids(word))),
        )
        for word, category in NOUNS.items()
    ]
    selected_words_path = tmp_path / "selected_words.csv"
    write_rows(selected_words_path, SelectedWord, words)
    contexts_path = tmp_path / "neutral_contexts.csv"

    build_neutral_contexts(
        settings,
        lambda source: ModelPreferringFoodAfterCooking(tokenizer),
        selected_words_path,
        contexts_path,
        tmp_path / "checks.csv",
    )

    with contexts_path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_context_that_favors_a_category_is_not_used(tmp_path: Path, settings: Settings) -> None:
    rows = run(tmp_path, settings)

    # "Let me cook the" は、食べ物の名詞を相対的に 1/3 出やすくする（ほかの文脈は 1/6 出にくい）
    assert [(row["number"], row["context"]) for row in rows] == [
        ("0", ""),
        ("1", "Do you know about the"),
        ("2", "I heard about the"),
    ]
    with (tmp_path / "checks.csv").open(encoding="utf-8") as file:
        checks = list(csv.DictReader(file))
    cooking_food = next(
        row for row in checks if row["context"] == "Let me cook the" and row["category"] == "food"
    )
    assert float(cooking_food["deviation"]) == pytest.approx(1 / 3)
    assert cooking_food["passed"] == "False"


def test_too_few_contexts_stop(tmp_path: Path, settings: Settings) -> None:
    evaluation = settings.evaluation.model_copy(update={"number_of_neutral_contexts": 3})

    with pytest.raises(CommandStopped, match="条件に合う中立文脈が 2 種しかありません"):
        run(tmp_path, settings.model_copy(update={"evaluation": evaluation}))
