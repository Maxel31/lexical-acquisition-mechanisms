"""未知語の獲得の割り当てのテスト。"""

import csv
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.settings import Settings, load_settings
from lexical_acquisition_mechanisms.unknown_word_assignments import (
    UnknownWordAssignment,
    write_unknown_word_assignments,
)
from lexical_acquisition_mechanisms.word_selection import SelectedWord

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def words(settings: Settings) -> list[SelectedWord]:
    rows = [SelectedWord(f"<new_{number:02d}>", "unknown", token_count=1) for number in range(24)]
    for token_count in (2, 3):
        for closeness in ("near", "far"):
            rows += [
                SelectedWord(
                    f"{closeness}{token_count}-{number}",
                    "unknown",
                    token_count=token_count,
                    spelling_closeness=closeness,
                )
                for number in range(12)
            ]
    for category in settings.words.categories:
        rows += [
            SelectedWord(
                f"{category.name}-{number}",
                "noun",
                category=category.name,
                estimated_pile_count=1000 * number,
            )
            for number in range(12)
        ]
    return rows


@pytest.fixture
def assignments(tmp_path: Path) -> list[UnknownWordAssignment]:
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    write_rows(tmp_path / "words.csv", SelectedWord, words(settings))
    write_unknown_word_assignments(
        settings, tmp_path / "words.csv", tmp_path / "assignments.csv", tmp_path / "overview.csv"
    )
    return read_rows(tmp_path / "assignments.csv", UnknownWordAssignment)


def test_each_unknown_word_meets_every_exposure_and_noun_once(
    assignments: list[UnknownWordAssignment],
) -> None:
    by_word = defaultdict(list)
    for assignment in assignments:
        by_word[assignment.unknown_word].append(assignment)
    assert len(by_word) == 72
    levels = [0, 1, 10, 100, 1000, 3000, 5000, 10000]
    for rows in by_word.values():
        assert sorted(row.exposure for row in rows) == levels
        assert len({row.noun for row in rows}) == 8
    # 組ごとに、名詞と露出回数の組み合わせは重ならず、1つの run で同じ名詞は1回だけ
    for group in range(1, 10):
        rows = [row for row in assignments if row.group == group]
        assert len({(row.noun, row.exposure) for row in rows}) == 64
        for run in {row.run for row in rows}:
            assert len({row.noun for row in rows if row.run == run}) == 8
    # 1 run に差し込む用例の数は 9 × 19,111
    for run in {row.run for row in assignments}:
        assert sum(row.exposure for row in assignments if row.run == run) == 171_999


def test_nouns_and_closeness_are_balanced(assignments: list[UnknownWordAssignment]) -> None:
    # 各トークン数に、各カテゴリの名詞が4語ずつ入る
    nouns = {(row.token_count, row.noun_category, row.noun) for row in assignments}
    per_cell = Counter((token_count, category) for token_count, category, _ in nouns)
    assert set(per_cell.values()) == {4}
    # 2・3トークンの組は near 4個・far 4個で、各 run に露出0回の near と far がある
    for token_count in (2, 3):
        rows = [row for row in assignments if row.token_count == token_count]
        for group in {row.group for row in rows}:
            members = {
                (row.unknown_word, row.spelling_closeness) for row in rows if row.group == group
            }
            assert Counter(closeness for _, closeness in members) == {"near": 4, "far": 4}
        for run in {row.run for row in rows}:
            zero = {row.spelling_closeness for row in rows if row.run == run and row.exposure == 0}
            assert zero == {"near", "far"}


def test_overview_has_a_row_per_unknown_word(tmp_path: Path, assignments) -> None:
    with (tmp_path / "overview.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 72
    assert list(rows[0]) == ["unknown_word", *(f"run-{number}" for number in range(1, 9))]
    assert " / " in rows[0]["run-1"]
