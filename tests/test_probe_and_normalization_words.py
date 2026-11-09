"""プローブの語と、正規化の基準の語のテスト。"""

from collections import Counter
from pathlib import Path

import pytest

from lexical_acquisition_mechanisms.adjective_candidates import AdjectiveCandidate
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.noun_candidates import NounCandidate
from lexical_acquisition_mechanisms.probe_and_normalization_words import (
    NormalizationWord,
    ProbeWord,
    split_into_tokens,
    write_probe_and_normalization_words,
)
from lexical_acquisition_mechanisms.settings import ModelSource, load_settings
from lexical_acquisition_mechanisms.word_selection import SelectedWord
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOUNS = "pasta bread cheese soup dog cat giraffe lasagna dumpling crocodile hedgehog caterpillar"
NOUNS += " armadillo chinchilla salamander ravioli tortellini"


@pytest.fixture
def tokenizer(tiny_model_source: ModelSource) -> WordTokenizer:
    return WordTokenizer.from_model(tiny_model_source)


def test_probe_and_normalization_words(
    tmp_path: Path, tiny_model_source: ModelSource, tokenizer: WordTokenizer
) -> None:
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    evaluation = settings.evaluation.model_copy(
        update={"probe_words_per_token_count": 4, "normalization_nouns_per_token_count": 2}
    )
    settings = settings.model_copy(
        update={"models": {"tiny": tiny_model_source}, "evaluation": evaluation}
    )
    words = [
        SelectedWord("pasta", "noun", token_count=1),
        SelectedWord("bread", "noun", token_count=1),
        SelectedWord("<new_01>", "unknown", token_count=1, token_ids="100278"),
    ]
    write_rows(tmp_path / "words.csv", SelectedWord, words)
    nouns = [
        NounCandidate(word, "food", "x", estimated_pile_count=1000 - index)
        for index, word in enumerate(NOUNS.split())
    ]
    write_rows(tmp_path / "nouns.csv", NounCandidate, nouns)
    write_rows(tmp_path / "adjectives.csv", AdjectiveCandidate, [AdjectiveCandidate("quiet", "x")])
    outputs = (tmp_path / "probe.csv", tmp_path / "normalization.csv")

    write_probe_and_normalization_words(
        settings,
        (tmp_path / "words.csv", tmp_path / "nouns.csv", tmp_path / "adjectives.csv"),
        outputs,
    )

    probe = read_rows(outputs[0], ProbeWord)
    kinds = Counter((row.token_count, row.kind) for row in probe)
    # 1トークンの負例は、未知語に使わなかった未使用のトークン番号（100279〜100351）
    assert kinds[(1, "untrained-token")] == 73
    assert kinds[(1, "real-word")] == 4
    assert all(row.text not in {"pasta", "bread"} for row in probe)
    real = {row.text for row in probe if row.kind == "real-word"}
    for row in probe:
        assert len(row.token_ids.split()) == row.token_count
        if row.kind == "nonword":
            assert row.text not in real
    assert {row.split for row in probe} == {"training", "held-out"}
    normalization = read_rows(outputs[1], NormalizationWord)
    assert [row.text for row in normalization if row.kind == "known-noun"] == [
        "giraffe",
        "lasagna",
        "armadillo",
        "chinchilla",
    ]
    split = [row for row in normalization if row.source_noun == "pasta"]
    assert [row.token_count for row in split] == [2, 3]
    assert all(row.text.replace("|", "") == " pasta" for row in split)


def test_words_are_split_into_single_tokens(tokenizer: WordTokenizer) -> None:
    pieces = split_into_tokens("pasta", 3, tokenizer)

    assert "".join(pieces) == " pasta"
    assert len(tokenizer.split_word_token_ids(pieces)) == 3
