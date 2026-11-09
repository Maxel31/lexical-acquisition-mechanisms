"""内部表現の類似度と確認用の一覧のテスト。モデルは、決まった内部表現を返す部品で置き換える。"""

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from lexical_acquisition_mechanisms.csv_tables import write_rows
from lexical_acquisition_mechanisms.hidden_state_similarity import build_hidden_state_similarity
from lexical_acquisition_mechanisms.neutral_contexts import NeutralContext
from lexical_acquisition_mechanisms.settings import ModelSource, Settings, load_settings
from lexical_acquisition_mechanisms.similarity_ratings import SimilarityRating
from lexical_acquisition_mechanisms.word_selection import SelectedWord
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# (語, 役割, カテゴリ, 内部表現)。hog は動物だが、内部表現は食べ物に近い
WORDS = [
    ("bagel", "target", "food", [1.0, 0.0]),
    ("bunny", "target", "animal", [0.0, 1.0]),
    ("pasta", "noun", "food", [1.0, 0.0]),
    ("bread", "noun", "food", [1.0, 0.1]),
    ("hog", "noun", "animal", [1.0, 0.2]),
    ("cat", "noun", "animal", [0.0, 1.0]),
    ("soup", "natural-noun", "food", [1.0, 0.0]),
    ("orange", "unknown", "", [1.0, 0.0]),
]
STATES = {word: state for word, _, _, state in WORDS}


class ModelWithFixedStates:
    """入力の最後の語の内部表現を、どの層でも STATES のものにする（層は2つ）。"""

    number_of_layers = 2

    def __init__(self, tokenizer: WordTokenizer) -> None:
        self.tokenizer = tokenizer

    def residual_streams(
        self, inputs: Sequence[Sequence[int]], positions: Sequence[int]
    ) -> np.ndarray:
        states = [STATES[self.tokenizer.decode(token_ids).split()[-1]] for token_ids in inputs]
        return np.repeat(np.array(states, dtype=np.float32)[:, None, :], 3, axis=1)


@pytest.fixture
def settings(tiny_model_source: ModelSource) -> Settings:
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    return settings.model_copy(update={"models": {"tiny": tiny_model_source}})


def test_similarities_checks_and_review_rows(tmp_path: Path, settings: Settings) -> None:
    tokenizer = WordTokenizer.from_model(settings.models["tiny"])
    words = [
        SelectedWord(
            word,
            role,
            category=category,
            token_ids=" ".join(map(str, tokenizer.word_token_ids(word))),
        )
        for word, role, category, _ in WORDS
    ]
    write_rows(tmp_path / "selected_words.csv", SelectedWord, words)
    contexts = [NeutralContext(0, ""), NeutralContext(1, "Do you know about the")]
    write_rows(tmp_path / "neutral_contexts.csv", NeutralContext, contexts)
    ratings = [
        SimilarityRating(target, "", noun, "", 5 if same else 2)
        for target, noun, same in [
            ("bagel", "pasta", True),
            ("bagel", "bread", True),
            ("bagel", "hog", False),
            ("bagel", "cat", False),
            ("bunny", "pasta", False),
            ("bunny", "bread", False),
            ("bunny", "hog", True),
            ("bunny", "cat", True),
        ]
    ]
    write_rows(tmp_path / "similarity_ratings.csv", SimilarityRating, ratings)
    outputs = (tmp_path / "similarity.csv", tmp_path / "checks.csv", tmp_path / "review.csv")

    build_hidden_state_similarity(
        settings,
        lambda source: ModelWithFixedStates(tokenizer),
        tmp_path / "selected_words.csv",
        tmp_path / "neutral_contexts.csv",
        tmp_path / "similarity_ratings.csv",
        outputs,
    )

    similarity, checks, review = (
        list(csv.DictReader(path.open(encoding="utf-8"))) for path in outputs
    )
    assert len(similarity) == 2 * 4
    values = {(row["target"], row["noun"]): float(row["cosine_similarity"]) for row in similarity}
    assert values[("bagel", "pasta")] == pytest.approx(1.0)
    # 同じカテゴリの組が上に来ない比べ方は、bunny と hog の組 < bagel と hog の組の1つだけ
    area = {row["measure"]: float(row["area_under_curve"]) for row in checks}
    assert area == pytest.approx({"tiny": 15 / 16, "llm": 1.0})
    shares = {row["word"]: row["same_category_share_of_nearest"] for row in review}
    assert shares == {
        "bagel": "1.00",
        "bunny": "1.00",
        "pasta": "1.00",
        "bread": "1.00",
        "hog": "0.00",
        "cat": "1.00",
        "soup": "",
    }
    assert all(row["approved"] == "" for row in review)
