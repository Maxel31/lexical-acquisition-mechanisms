"""WordNet での語義とカテゴリのテスト（初回だけ WordNet をダウンロードする）。"""

import stat
from pathlib import Path

import pytest

from lexical_acquisition_mechanisms import wordnet_senses
from lexical_acquisition_mechanisms.settings import load_settings
from lexical_acquisition_mechanisms.wordnet_senses import (
    adjective_senses,
    noun_senses,
    other_parts_of_speech,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = load_settings(REPOSITORY_ROOT / "settings.yaml").words.categories


@pytest.fixture(scope="module", autouse=True)
def wordnet_in_temporary_directory(tmp_path_factory: pytest.TempPathFactory):
    original = wordnet_senses.NLTK_DATA_DIRECTORY
    wordnet_senses.NLTK_DATA_DIRECTORY = tmp_path_factory.mktemp("data") / "nltk_data"
    wordnet_senses.load_wordnet.cache_clear()
    try:
        wordnet_senses.load_wordnet()
    except (RuntimeError, OSError) as error:
        pytest.skip(f"WordNet を用意できません: {error}")
    yield wordnet_senses.NLTK_DATA_DIRECTORY
    wordnet_senses.NLTK_DATA_DIRECTORY = original
    wordnet_senses.load_wordnet.cache_clear()


def test_download_directory_is_private(wordnet_in_temporary_directory: Path) -> None:
    mode = wordnet_in_temporary_directory.stat().st_mode
    assert not mode & (stat.S_IWGRP | stat.S_IWOTH)


def test_noun_senses_belong_to_categories_under_their_roots() -> None:
    assert {sense.categories for sense in noun_senses("pasta", CATEGORIES)} == {("food",)}
    # 靴は clothing.n.01 の下になく、footwear.n.02 の下にある
    assert ("clothing",) in [sense.categories for sense in noun_senses("shoe", CATEGORIES)]


def test_word_in_two_categories_is_visible() -> None:
    categories = {name for sense in noun_senses("chicken", CATEGORIES) for name in sense.categories}

    assert categories == {"food", "animal"}


def test_senses_are_only_those_headed_by_the_word_itself() -> None:
    """複数形などから元の形に戻した語義（例: cars で car の語義）は含めない。"""
    assert noun_senses("cars", CATEGORIES) == []
    assert noun_senses("car", CATEGORIES)[0].synset == "car.n.01"


def test_adjective_senses_and_other_parts_of_speech() -> None:
    assert adjective_senses("quiet")[0].synset == "quiet.a.01"
    assert adjective_senses("pasta") == []
    assert other_parts_of_speech("quiet") == ["v", "r"]
    assert other_parts_of_speech("pasta") == []
