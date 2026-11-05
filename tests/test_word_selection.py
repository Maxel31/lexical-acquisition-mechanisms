"""語を選ぶ処理のテスト。候補の CSV は、頻度などを決めて手元に作る。"""

import csv
from pathlib import Path

import pytest

from lexical_acquisition_mechanisms import word_selection
from lexical_acquisition_mechanisms.adjective_candidates import AdjectiveCandidate
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.csv_tables import write_rows
from lexical_acquisition_mechanisms.noun_candidates import NounCandidate
from lexical_acquisition_mechanisms.settings import ModelSource, Settings, load_settings
from lexical_acquisition_mechanisms.word_selection import CandidateTables, build_selected_words

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# (語, カテゴリ, 先頭スペース付きのトークン数, Pile での出現回数の常用対数)
NOUNS = [
    ("pasta", "food", 1, 5.0),
    ("bread", "food", 1, 5.2),
    ("cheese", "food", 1, 5.6),
    ("soup", "food", 1, 4.6),
    ("bagel", "food", 1, 3.45),
    ("lasagna", "food", 2, 3.5),
    ("meatball", "food", 2, 3.9),
    ("ravioli", "food", 3, 3.4),
    ("tortellini", "food", 3, 2.9),
    ("pig", "animal", 1, 5.25),
    ("hog", "animal", 1, 5.3),
    ("cat", "animal", 1, 5.9),
    ("dog", "animal", 1, 6.5),
    ("bunny", "animal", 1, 3.36),
    ("catfish", "animal", 2, 3.42),
    ("giraffe", "animal", 2, 3.31),
    ("armadillo", "animal", 3, 3.47),
    ("chinchilla", "animal", 3, 3.0),
]
# (語, Pile での出現回数の常用対数, 形容詞として使われていた割合, カテゴリとの結びつきの最大)
ADJECTIVES = [
    ("big", 5.25, 0.95, 2),
    ("large", 5.3, 0.95, 2),
    ("happy", 5.6, 0.95, 1),
    ("quiet", 5.0, 0.95, 2),
    ("bright", 6.0, 0.95, 3),
    ("green", 5.5, 0.95, 3),
    ("delicious", 5.9, 0.95, 6),
    ("cold", 5.3, 0.6, 2),
]


def noun_candidate(
    word: str,
    category: str,
    token_count: int,
    frequency: float,
    share: float = 0.95,
    rejected_reason: str = "",
) -> NounCandidate:
    count = round(10**frequency)
    return NounCandidate(
        word,
        category,
        "x",
        rejected_reason=rejected_reason,
        synset=f"{word}.n.01",
        token_count=token_count,
        estimated_pile_count=count,
        estimated_pile_passages=count,
        judged_usages=50,
        intended_usage_share=share,
    )


def adjective_candidate(
    word: str, frequency: float, share: float, association: int
) -> AdjectiveCandidate:
    count = round(10**frequency)
    return AdjectiveCandidate(
        word,
        "x",
        synset=f"{word}.a.01",
        token_count=1,
        estimated_pile_count=count,
        estimated_pile_passages=count,
        judged_usages=50,
        adjective_usage_share=share,
        largest_category_association=association,
    )


@pytest.fixture
def settings(
    tiny_model_source: ModelSource, wordnet_directory: Path, monkeypatch: pytest.MonkeyPatch
) -> Settings:
    """食べ物と動物の2カテゴリで、名詞を各2語選ぶ。"""
    monkeypatch.setattr(word_selection, "NOUNS_PER_CATEGORY", 2)
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    words = settings.words.model_copy(update={"categories": settings.words.categories[:2]})
    return settings.model_copy(update={"models": {"tiny": tiny_model_source}, "words": words})


def select(
    tmp_path: Path, settings: Settings, nouns: list[NounCandidate]
) -> dict[str, dict[str, str]]:
    tables = CandidateTables(
        nouns=tmp_path / "noun_candidates.csv", adjectives=tmp_path / "adjective_candidates.csv"
    )
    write_rows(tables.nouns, NounCandidate, nouns)
    adjectives = [adjective_candidate(*adjective) for adjective in ADJECTIVES]
    write_rows(tables.adjectives, AdjectiveCandidate, adjectives)
    output_path = tmp_path / "selected_words.csv"
    build_selected_words(settings, tables, output_path)
    with output_path.open(encoding="utf-8") as file:
        return {row["word"]: row for row in csv.DictReader(file)}


def words_with_role(rows: dict[str, dict[str, str]], role: str) -> set[str]:
    return {word for word, row in rows.items() if row["role"] == role}


def test_nouns_match_in_frequency_across_categories_without_synonyms(
    tmp_path: Path, settings: Settings
) -> None:
    candidates = [noun_candidate(*noun) for noun in NOUNS]
    candidates += [
        noun_candidate("mouse", "animal", 1, 5.5, share=0.5),
        noun_candidate("chicken", "animal", 1, 5.5, rejected_reason="複数のカテゴリで挙がった"),
    ]

    rows = select(tmp_path, settings, candidates)

    # 両カテゴリに候補がある範囲（10^5.25〜10^5.6）の、2つの目標の値に近い語。
    # pig は hog の同義語なので選ばず、mouse は意図した語義の割合が低いので選ばない
    assert words_with_role(rows, "noun") == {"bread", "cheese", "hog", "cat"}
    assert rows["cheese"]["category"] == "food"
    assert rows["cheese"]["token_count"] == "1"


def test_target_words_match_the_multi_token_frequencies(tmp_path: Path, settings: Settings) -> None:
    rows = select(tmp_path, settings, [noun_candidate(*noun) for noun in NOUNS])

    # 1トークンの語も、多トークンの語の頻度（10^3.4 あたり）に合わせて選ぶ。
    # catfish は、最初のトークン " cat" が名詞 cat と重なるので選ばない
    assert words_with_role(rows, "target") == {
        "bagel",
        "lasagna",
        "ravioli",
        "bunny",
        "giraffe",
        "armadillo",
    }
    assert rows["lasagna"]["token_count"] == "2"
    assert len(rows["lasagna"]["token_ids"].split()) == 2


def test_adjectives_match_the_noun_frequencies(tmp_path: Path, settings: Settings) -> None:
    rows = select(tmp_path, settings, [noun_candidate(*noun) for noun in NOUNS])

    # 名詞（10^5.2・5.3・5.6・5.9）の頻度の低い順に、最も近い形容詞を選ぶ。large は big の同義語、
    # delicious は食べ物との結びつきが強く、cold は形容詞として使われる割合が低いので選ばない
    assert words_with_role(rows, "adjective") == {"big", "green", "happy", "bright"}
    assert rows["happy"]["category"] == ""
    assert float(rows["happy"]["usage_share"]) == pytest.approx(0.95)


def test_missing_candidates_stop_with_the_category(tmp_path: Path, settings: Settings) -> None:
    candidates = [noun_candidate(*noun) for noun in NOUNS if noun[0] not in {"pig", "hog", "cat"}]

    with pytest.raises(CommandStopped, match="カテゴリ animal の名詞（1トークン）") as stopped:
        select(tmp_path, settings, candidates)
    assert stopped.value.rerun_from == "words generate-noun-candidates"
