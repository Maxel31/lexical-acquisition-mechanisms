"""語義の追加の割り当てのテスト。"""

from collections import defaultdict
from pathlib import Path

import pytest

from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.hidden_state_similarity import HiddenStateSimilarity
from lexical_acquisition_mechanisms.sense_addition_assignments import (
    OtherCategoryPair,
    SenseAdditionAssignment,
    write_sense_addition_assignments,
)
from lexical_acquisition_mechanisms.settings import load_settings
from lexical_acquisition_mechanisms.similarity_ratings import SimilarityRating
from lexical_acquisition_mechanisms.word_selection import SelectedWord

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# 別カテゴリでも意味が近い組（並べ替えで避ける）
CLOSE_PAIR = ("animal-target-1", "food-0")


@pytest.fixture
def assignments(tmp_path: Path) -> list[SenseAdditionAssignment]:
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    categories = [category.name for category in settings.words.categories]
    words = []
    for category in categories:
        words += [
            SelectedWord(
                f"{category}-{number}", "noun", category=category, estimated_pile_count=number
            )
            for number in range(12)
        ]
        words += [
            SelectedWord(
                f"{category}-target-{count}", "target", category=category, token_count=count
            )
            for count in (1, 2, 3)
        ]
    words += [
        SelectedWord(f"adjective-{number}", "adjective", estimated_pile_count=number)
        for number in range(72)
    ]
    write_rows(tmp_path / "words.csv", SelectedWord, words)
    targets = [word for word in words if word.role == "target"]
    nouns = [word for word in words if word.role == "noun"]
    similarities = []
    ratings = []
    for target in targets:
        for noun in nouns:
            same = target.category == noun.category
            close = same or (target.word, noun.word) == CLOSE_PAIR
            similarities.append(
                HiddenStateSimilarity(
                    target.word,
                    target.category,
                    noun.word,
                    noun.category,
                    "tiny",
                    0.8 if close else 0.2,
                )
            )
            ratings.append(SimilarityRating(target.word, "", noun.word, "", 5 if close else 2))
    write_rows(tmp_path / "similarity.csv", HiddenStateSimilarity, similarities)
    write_rows(tmp_path / "ratings.csv", SimilarityRating, ratings)
    outputs = (tmp_path / "assignments.csv", tmp_path / "pairs.csv")

    write_sense_addition_assignments(
        settings,
        tmp_path / "words.csv",
        tmp_path / "similarity.csv",
        tmp_path / "ratings.csv",
        outputs,
    )

    return read_rows(outputs[0], SenseAdditionAssignment)


def donors(assignments: list[SenseAdditionAssignment], run: str) -> dict[str, list[str]]:
    by_target = defaultdict(list)
    for row in sorted(assignments, key=lambda row: row.order):
        if row.run == run:
            by_target[row.target].append(row.donor)
    return by_target


def test_runs_follow_the_assignment_rules(assignments: list[SenseAdditionAssignment]) -> None:
    assert {row.run for row in assignments} == {
        f"{kind}-{number}"
        for kind in ("same-category", "other-category", "adjective")
        for number in (1, 2, 4)
    }
    for number in (1, 2, 4):
        same = donors(assignments, f"same-category-{number}")
        other = donors(assignments, f"other-category-{number}")
        # 同カテゴリと別カテゴリで、使う donor の集合は同じ（違うのは割り当て先だけ）
        assert {d for ds in same.values() for d in ds} == {d for ds in other.values() for d in ds}
        for run in (f"same-category-{number}", f"other-category-{number}", f"adjective-{number}"):
            assigned = donors(assignments, run)
            assert all(len(target_donors) == number for target_donors in assigned.values())
            # 1つの run の中で、同じ donor を2つのターゲット語に割り当てない
            all_donors = [donor for target_donors in assigned.values() for donor in target_donors]
            assert len(all_donors) == len(set(all_donors))
        rows = [row for row in assignments if row.run == f"other-category-{number}"]
        for target in donors(assignments, f"other-category-{number}"):
            categories = {row.donor_category for row in rows if row.target == target}
            # 足す語義は全て互いに別のカテゴリで、ターゲット語のカテゴリでもない
            assert len(categories) == number
            assert target.split("-")[0] not in categories
    # 数 N は入れ子になる
    for kind in ("same-category", "other-category", "adjective"):
        one, two, four = (donors(assignments, f"{kind}-{number}") for number in (1, 2, 4))
        assert all(
            four[target][:2] == two[target] and two[target][:1] == one[target] for target in four
        )


def test_other_category_pairs_are_distant(tmp_path: Path, assignments) -> None:
    pairs = read_rows(tmp_path / "pairs.csv", OtherCategoryPair)
    assert len(pairs) == 18 * 4
    assert all(pair.distant for pair in pairs)
    assert CLOSE_PAIR not in {(pair.target, pair.noun) for pair in pairs}
