"""2つ目の語義の追加の割り当てのテスト。"""

from collections import Counter

import pytest

from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.second_sense_assignments import (
    run_names,
    second_sense_assignments,
)
from lexical_acquisition_mechanisms.settings import SecondSenseAdditionSettings
from lexical_acquisition_mechanisms.unknown_word_assignments import UnknownWordAssignment
from lexical_acquisition_mechanisms.word_selection import SelectedWord

CATEGORIES = ["animal", "building", "clothing", "food", "furniture", "vehicle"]
NOUNS = [f"{category}-{number}" for category in CATEGORIES for number in range(4)]
WORDS = [SelectedWord(noun, "noun", category=noun.split("-")[0]) for noun in NOUNS]
FIRST_LEVELS = [0, 1, 10, 100, 1000, 3000, 5000, 9900]
SECTION = SecondSenseAdditionSettings(
    minimum_first_exposure=1000,
    second_exposure_levels=[0, 10, 100, 1000],
)
PASSAGE_COUNTS = dict.fromkeys(NOUNS, 10000)


def first_assignments() -> list[UnknownWordAssignment]:
    rows = []
    for run_index, run in enumerate(run_names()):
        for index in range(8):
            noun = NOUNS[(index * 3 + run_index) % len(NOUNS)]
            exposure = FIRST_LEVELS[(index + run_index) % len(FIRST_LEVELS)]
            rows.append(
                UnknownWordAssignment(
                    run, f"<new_{index}>", 1, "", 1, index, noun, noun.split("-")[0], exposure
                )
            )
    return rows


def test_only_learned_words_get_a_second_sense_with_every_level_in_each_run() -> None:
    first = first_assignments()
    rows = second_sense_assignments(SECTION, 0, WORDS, first, PASSAGE_COUNTS)

    assert rows == second_sense_assignments(SECTION, 0, WORDS, first, PASSAGE_COUNTS)
    for run in run_names():
        in_run = [row for row in rows if row.run == run]
        assert len(in_run) == 4
        assert all(row.first_exposure >= 1000 for row in in_run)
        assert Counter(row.second_exposure for row in in_run) == Counter([0, 10, 100, 1000])
        # Y の用例は、未知語の獲得で使った分と合わせて、学習用の用例の 95% に収まる
        used = {row.noun: row.exposure for row in first if row.run == run}
        for row in in_run:
            used[row.second_noun] = used.get(row.second_noun, 0) + row.second_exposure
        assert all(used[row.second_noun] <= 9500 for row in in_run)
    for row in rows:
        assert row.second_noun != row.first_noun
        assert row.second_category == row.second_noun.split("-")[0]
        same = row.second_category == row.first_category
        assert same == (row.kind == "same-category")
    assert {row.kind for row in rows} == {"same-category", "other-category"}


def test_stops_when_learned_words_do_not_divide_into_the_levels() -> None:
    section = SECTION.model_copy(update={"minimum_first_exposure": 3000})

    with pytest.raises(CommandStopped) as stopped:
        second_sense_assignments(section, 0, WORDS, first_assignments(), PASSAGE_COUNTS)

    assert stopped.value.rerun_from == "assignments build --experiment second-sense-addition"


def test_stops_when_no_noun_has_passages_left() -> None:
    counts = dict.fromkeys(NOUNS, 500)

    with pytest.raises(CommandStopped, match="用例が残っている名詞がありません"):
        second_sense_assignments(SECTION, 0, WORDS, first_assignments(), counts)
