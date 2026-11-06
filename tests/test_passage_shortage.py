"""用例や候補が足りない語で止まる処理のテスト。"""

from pathlib import Path

import pytest

from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.passage_collection import CollectionSummary
from lexical_acquisition_mechanisms.passage_shortage import (
    drop_natural_words_or_stop,
    find_shortages,
    natural_word_problems,
    stop_for_shortage,
)
from lexical_acquisition_mechanisms.settings import Settings, load_settings
from lexical_acquisition_mechanisms.word_selection import ExcludedWord, SelectedWord

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings() -> Settings:
    """名詞は学習用 3 件と評価用 1 件の合格が必要。"""
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    passages = settings.passages.model_copy(update={"evaluation_passages_per_word": 1})
    exposure = settings.unknown_word_acquisition.model_copy(update={"exposure_levels": [0, 3]})
    return settings.model_copy(update={"passages": passages, "unknown_word_acquisition": exposure})


def test_shortages_compare_with_the_needed_number_times_the_factor(settings: Settings) -> None:
    words = [SelectedWord("pasta", "noun")]
    counts = {("pasta", "training"): 5, ("pasta", "evaluation"): 1}

    assert find_shortages(words, counts, settings) == []
    shortages = find_shortages(words, counts, settings, factor=2)
    assert [(shortage.split, shortage.found, shortage.needed) for shortage in shortages] == [
        ("training", 5, 3),
        ("evaluation", 1, 1),
    ]


def test_only_words_without_more_occurrences_are_excluded(
    tmp_path: Path, settings: Settings
) -> None:
    words = [SelectedWord("pasta", "noun"), SelectedWord("bread", "noun")]
    summary = [
        CollectionSummary("pasta", "noun", 4, 10, 8, "", False),
        CollectionSummary("bread", "noun", 4, 10, 8, "", True),
    ]
    write_rows(tmp_path / "collection_summary.csv", CollectionSummary, summary)
    counts = {(word.word, split): 1 for word in words for split in ("training", "evaluation")}
    shortages = find_shortages(words, counts, settings)

    with pytest.raises(CommandStopped, match="bread（training 1 / 3 件）") as stopped:
        stop_for_shortage(
            shortages, "候補", tmp_path / "collection_summary.csv", tmp_path / "excluded.csv"
        )

    # 外した語があれば、words select から（occurrence_margin を大きくするのは、その前に）
    assert stopped.value.rerun_from == "words select"
    excluded = read_rows(tmp_path / "excluded.csv", ExcludedWord)
    assert [(row.word, row.reason) for row in excluded] == [
        ("pasta", "候補が足りない（training 1 / 3 件）")
    ]


def natural_words_settings(settings: Settings, nouns: int, adjectives: int) -> Settings:
    selection = settings.word_selection.model_copy(
        update={
            "minimum_natural_nouns_per_category": nouns,
            "minimum_natural_adjectives": adjectives,
        }
    )
    evaluation = settings.evaluation.model_copy(update={"passages_per_natural_word": 1})
    return settings.model_copy(update={"word_selection": selection, "evaluation": evaluation})


NATURAL_WORDS = [
    SelectedWord("soup", "natural-noun", category="food"),
    SelectedWord("stew", "natural-noun", category="food"),
    SelectedWord("calm", "natural-adjective"),
    SelectedWord("quiet", "natural-adjective"),
]


def test_short_natural_words_are_dropped_while_enough_are_left(
    tmp_path: Path, settings: Settings
) -> None:
    settings = natural_words_settings(settings, nouns=1, adjectives=1)
    counts = {("stew", "evaluation"): 1, ("calm", "evaluation"): 1}
    shortages = find_shortages(NATURAL_WORDS, counts, settings)
    paths = (tmp_path / "collection_summary.csv", tmp_path / "excluded.csv")

    dropped = drop_natural_words_or_stop(shortages, NATURAL_WORDS, "用例", paths, settings)

    assert dropped == {"soup", "quiet"}
    assert not paths[1].exists()


@pytest.mark.parametrize(
    "nouns, adjectives, problem",
    [(2, 1, "カテゴリ food の自然な名詞 1 語"), (1, 2, "自然な形容詞 1 語")],
)
def test_natural_words_below_the_minimum_stop(
    tmp_path: Path, settings: Settings, nouns: int, adjectives: int, problem: str
) -> None:
    settings = natural_words_settings(settings, nouns, adjectives)
    counts = {("stew", "evaluation"): 1, ("calm", "evaluation"): 1}
    shortages = find_shortages(NATURAL_WORDS, counts, settings)
    summary = [CollectionSummary(word.word, word.role, 1, 2, 2, "", True) for word in NATURAL_WORDS]
    write_rows(tmp_path / "collection_summary.csv", CollectionSummary, summary)

    assert natural_word_problems(NATURAL_WORDS, {"soup", "quiet"}, settings) == [problem]
    with pytest.raises(CommandStopped, match="soup（evaluation 0 / 1 件）"):
        drop_natural_words_or_stop(
            shortages,
            NATURAL_WORDS,
            "用例",
            (tmp_path / "collection_summary.csv", tmp_path / "excluded.csv"),
            settings,
        )
