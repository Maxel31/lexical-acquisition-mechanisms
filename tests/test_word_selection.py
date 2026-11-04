"""語を選ぶ処理のテスト。候補の CSV は、頻度などを決めて手元に作る。"""

import csv
from pathlib import Path

import numpy as np
import pytest

from lexical_acquisition_mechanisms import word_selection
from lexical_acquisition_mechanisms.adjective_candidates import AdjectiveCandidate
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.csv_tables import write_rows
from lexical_acquisition_mechanisms.noun_candidates import NounCandidate
from lexical_acquisition_mechanisms.settings import ModelSource, Settings, load_settings
from lexical_acquisition_mechanisms.unknown_word_candidates import UnknownWordCandidate
from lexical_acquisition_mechanisms.word_selection import SelectionInputs, build_selected_words
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

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
# 未知語の候補（語, 綴りの近さ, near の元の語, トークン数, トークン番号, 出現回数）
UNKNOWN_WORDS = [
    UnknownWordCandidate("glorpa", "near", "happy", 2, "99001 99002", 0),
    UnknownWordCandidate("vibbet", "near", "gentle", 2, "99003 99004", 3),
    UnknownWordCandidate("zubbet", "near", "gentle", 2, "99005 99006", 0),
    UnknownWordCandidate("zuffle", "far", "", 2, "99005 99007", 0),
    UnknownWordCandidate("snorf", "far", "", 2, "99008 99009", 0),
    UnknownWordCandidate("thwomp", "near", "gentle", 3, "99010 99011 99012", 0),
    UnknownWordCandidate("spleng", "far", "", 3, "99013 99014 99015", 0),
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
    """食べ物と動物の2カテゴリで、名詞を各2語、自然な名詞を各3語、自然な形容詞を2語選ぶ。

    未知語は、1トークンを2個、2・3トークンを綴りの近さごとに1個選ぶ。
    """
    monkeypatch.setattr(word_selection, "NOUNS_PER_CATEGORY", 2)
    monkeypatch.setattr(word_selection, "ONE_TOKEN_UNKNOWN_WORDS", 2)
    monkeypatch.setattr(word_selection, "UNKNOWN_WORDS_PER_GROUP", 1)
    monkeypatch.setattr(word_selection, "NATURAL_NOUNS_PER_CATEGORY", 3)
    monkeypatch.setattr(word_selection, "NATURAL_ADJECTIVES", 2)
    monkeypatch.setattr(word_selection, "PART_OF_SPEECH_NOUNS_PER_CATEGORY", 1)
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    words = settings.words.model_copy(update={"categories": settings.words.categories[:2]})
    return settings.model_copy(update={"models": {"tiny": tiny_model_source}, "words": words})


def select(
    tmp_path: Path,
    settings: Settings,
    nouns: list[NounCandidate],
    unknown_words: list[UnknownWordCandidate] = UNKNOWN_WORDS,
) -> dict[str, dict[str, str]]:
    tables = SelectionInputs(
        nouns=tmp_path / "noun_candidates.csv",
        adjectives=tmp_path / "adjective_candidates.csv",
        unknown_words=tmp_path / "unknown_word_candidates.csv",
        base_sequences=tmp_path / "base_sequences.npy",
    )
    tokenizer = WordTokenizer.from_model(next(iter(settings.models.values())))
    tokens = tokenizer.encode(["We ate bread, cheese and bread with breadsticks."])
    np.save(tables.base_sequences, np.array([tokens], dtype=np.uint32))
    write_rows(tables.nouns, NounCandidate, nouns)
    adjectives = [adjective_candidate(*adjective) for adjective in ADJECTIVES]
    write_rows(tables.adjectives, AdjectiveCandidate, adjectives)
    write_rows(tables.unknown_words, UnknownWordCandidate, unknown_words)
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
    # 基データ（の代わりの1文）での出現回数。" breadsticks" の一部は数えない
    assert rows["bread"]["base_data_count"] == "2"
    assert rows["cheese"]["base_data_count"] == "1"


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


def test_natural_words_are_the_remaining_words_of_each_token_count(
    tmp_path: Path, settings: Settings
) -> None:
    rows = select(tmp_path, settings, [noun_candidate(*noun) for noun in NOUNS])

    # 管理対象の語と、その同義語（pig・large）を除いた語から、1・2・3トークンの順に選ぶ
    natural_nouns = words_with_role(rows, "natural-noun")
    assert {"meatball", "tortellini", "dog", "catfish", "chinchilla"} <= natural_nouns
    assert len(natural_nouns & {"pasta", "soup"}) == 1
    assert words_with_role(rows, "natural-adjective") == {"quiet", "delicious"}
    # 品詞のプローブには、1トークンの名詞（各カテゴリ1語）と、自然な形容詞を使う
    probe_words = {word for word, row in rows.items() if row["part_of_speech_probe"] == "True"}
    assert probe_words == (natural_nouns & {"pasta", "soup"}) | {"dog", "quiet", "delicious"}


def test_unknown_words_have_first_tokens_of_their_own(tmp_path: Path, settings: Settings) -> None:
    tokenizer = WordTokenizer.from_model(next(iter(settings.models.values())))
    bread = tokenizer.word_token_ids("bread")[0]
    unknown_words = [UnknownWordCandidate("breadu", "near", "x", 2, f"{bread} 99000", 0)]
    unknown_words += UNKNOWN_WORDS

    rows = select(tmp_path, settings, [noun_candidate(*noun) for noun in NOUNS], unknown_words)

    # breadu は最初のトークンが名詞 bread と、zuffle は zubbet と重なる。glorpa は元の語が
    # 管理対象の形容詞（happy）、vibbet はコーパスに現れるので選ばない
    assert words_with_role(rows, "unknown") == {
        "<new_01>",
        "<new_02>",
        "zubbet",
        "snorf",
        "thwomp",
        "spleng",
    }
    assert rows["<new_01>"]["token_ids"] == "100278"
    assert rows["<new_01>"]["spelling_closeness"] == ""
    assert rows["zubbet"]["spelling_closeness"] == "near"
    assert rows["zubbet"]["source_word"] == "gentle"


def test_missing_unknown_words_stop_with_the_group(tmp_path: Path, settings: Settings) -> None:
    unknown_words = [word for word in UNKNOWN_WORDS if word.word != "spleng"]

    with pytest.raises(CommandStopped, match="3トークン・far の未知語") as stopped:
        select(tmp_path, settings, [noun_candidate(*noun) for noun in NOUNS], unknown_words)
    assert stopped.value.rerun_from == "words generate-unknown-word-candidates"


def test_missing_candidates_stop_with_the_category(tmp_path: Path, settings: Settings) -> None:
    candidates = [noun_candidate(*noun) for noun in NOUNS if noun[0] not in {"pig", "hog", "cat"}]

    with pytest.raises(CommandStopped, match="カテゴリ animal の名詞（1トークン）") as stopped:
        select(tmp_path, settings, candidates)
    assert stopped.value.rerun_from == "words generate-noun-candidates"
