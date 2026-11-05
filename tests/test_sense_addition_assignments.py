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
# 別カテゴリでも意味が近い組（並べ替えで避ける）。枠（food, 1トークン）の1番目の名詞 food-0 は、
# 並べ替えなければ、building の1トークンのターゲット語に足す
CLOSE_PAIR = ("building-target-1", "food-0")


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


def test_orders_of_donors_do_not_follow_their_frequency(
    assignments: list[SenseAdditionAssignment],
) -> None:
    # 名詞の頻度（estimated_pile_count）はカテゴリの中の番号、形容詞は番号。並べ替えなければ、
    # 1番目の donor は、どれも枠（ターゲット語に配った4語）の中で最も頻度が低い
    first = [row for row in assignments if row.run == "same-category-1"]
    ranks = []
    for row in first:
        # 同じ枠の4語（same-category-4 で、同じターゲット語に足す語）の中での、頻度の順位
        slot = sorted(
            (
                other.donor
                for other in assignments
                if other.run == "same-category-4" and other.target == row.target
            ),
            key=lambda donor: int(donor.rsplit("-", 1)[1]),
        )
        ranks.append(slot.index(row.donor))
    assert 1.0 < sum(ranks) / len(ranks) < 2.0
    adjectives = [int(row.donor.split("-")[1]) for row in assignments if row.run == "adjective-1"]
    # 頻度の下位18語（平均 8.5）ではなく、全体の平均（35.5）に近い
    assert 25 < sum(adjectives) / len(adjectives) < 45


def test_frequency_of_first_donors_is_the_same_for_every_token_count(
    assignments: list[SenseAdditionAssignment],
) -> None:
    # トークン数のアブレーションに頻度が交絡しないように、1番目の donor の頻度（名詞は枠の中の
    # 順位、形容詞は番号）の平均は、ターゲット語のトークン数によらない
    def mean_by_token_count(run: str, frequency) -> dict[int, float]:
        values = defaultdict(list)
        for row in assignments:
            # 枠 (food, 1トークン) は、CLOSE_PAIR を避けるために並べ替えるので、名詞では比べない
            if row.run == run and not (run.startswith("same") and row.target_category == "food"):
                values[row.target_token_count].append(frequency(row))
        return {count: sum(numbers) / len(numbers) for count, numbers in values.items()}

    def slot_rank(row: SenseAdditionAssignment) -> int:
        slot = sorted(
            (
                other.donor
                for other in assignments
                if other.run == "same-category-4" and other.target == row.target
            ),
            key=lambda donor: int(donor.rsplit("-", 1)[1]),
        )
        return slot.index(row.donor)

    nouns = mean_by_token_count("same-category-1", slot_rank)
    adjectives = mean_by_token_count("adjective-1", lambda row: int(row.donor.split("-")[1]))
    assert len(set(nouns.values())) == 1
    # 形容詞は、配る順（1〜18・18〜1）があるので、ほぼそろう（72語の番号で1以内）
    assert max(adjectives.values()) - min(adjectives.values()) <= 1


def test_frequency_of_adjective_donors_does_not_follow_the_number_of_senses(
    assignments: list[SenseAdditionAssignment],
) -> None:
    # N と donor の頻度が一緒に動かない: どの N でも、形容詞の番号（頻度の順）の平均が、全体の平均
    # （35.5）から1以内
    for number in (1, 2, 4):
        numbers = [
            int(row.donor.split("-")[1]) for row in assignments if row.run == f"adjective-{number}"
        ]
        assert abs(sum(numbers) / len(numbers) - 35.5) <= 1
