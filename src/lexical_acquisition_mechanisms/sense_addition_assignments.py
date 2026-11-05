"""語義の追加の割り当て（assignments build --experiment sense-addition）。

run ごとに、各ターゲット語に、どの語（donor）の語義を足すかを決める。donor の用例の donor を
ターゲット語に置き換えて差し込む。
1. 各カテゴリの名詞12語を、ターゲット語のトークン数（1・2・3）ごとに4語ずつの「枠」に分ける
   （未知語の獲得で、名詞をトークン数に分けたのと同じ分け方）。形容詞72語は、頻度の順に、
   ターゲット語18語へ 1〜18・18〜1・… の順に4語ずつ配る。枠の中と、配った4語の中の順番
   （1〜4番目）は、頻度の低い順から、カテゴリごとに SHIFTS_BY_CATEGORY だけずらす（N と donor の
   頻度が一緒に動くと、N の比較に頻度が交絡するので、N=1・2 の donor の頻度の平均が、N=4（4語の
   全て）に近くなるように）。どのトークン数でも同じずらし方にして、トークン数の比較にも頻度が
   交絡しにくくする。
2. 足す語義の種類と数 N（1・2・4）ごとの run で、カテゴリ k・トークン数 j のターゲット語に、
   - 同カテゴリ（same-category-N）: 枠 (k, j) の1〜N番目
   - 別カテゴリ（other-category-N）: 枠 (k+1, j) の1番目、枠 (k+2, j) の2番目、…、枠 (k+N, j) の
     N番目（k+n は6で割った余り）
   - 形容詞（adjective-N）: 配った形容詞の1〜N番目
   を割り当てる。対照の run（noun-control・adjective-control）は、何も置き換えない。
3. 別カテゴリの組（ターゲット語と、足す名詞）は、意味が遠い組にする。内部表現の類似度（全モデルの
   平均）と LLM の評定がどちらも、全ての組の中央値以下の組を「遠い」とし、枠の中の順番を、
   別カテゴリの組が全て遠くなる並べ替え（元の順に近いものから探す）に替える。そうできない枠は、
   遠くない組が最も少ない並べ替えにして、確認の表に書く。
"""

import argparse
import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lexical_acquisition_mechanisms import (
    hidden_state_similarity,
    similarity_ratings,
    word_finalization,
)
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.hidden_state_similarity import HiddenStateSimilarity
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.settings import Settings
from lexical_acquisition_mechanisms.similarity_ratings import SimilarityRating
from lexical_acquisition_mechanisms.unknown_word_assignments import (
    TOKEN_COUNTS,
    nouns_by_token_count,
)
from lexical_acquisition_mechanisms.word_selection import SelectedWord

EXPERIMENT = "sense-addition"
COMMAND = f"assignments build --experiment {EXPERIMENT}"
OUTPUT_DIRECTORY = Path("data/assignments") / EXPERIMENT
ASSIGNMENTS_PATH = OUTPUT_DIRECTORY / "assignments.csv"
PAIR_CHECKS_PATH = OUTPUT_DIRECTORY / "other_category_pairs.csv"
NUMBERS_OF_SENSES = (1, 2, 4)
KINDS = ("same-category", "other-category", "adjective")
CONTROL_RUNS = ("noun-control", "adjective-control")
DONORS_PER_TARGET = 4
# カテゴリの順番ごとの、枠の中と配った形容詞の中の順番をずらす量（6カテゴリ。0〜3 を1回ずつと、
# 1・3 をもう1回）。実データ（2026-11-06）で、N=1・2・4 の donor の log10 頻度の平均が最もそろった
# 並び（名詞 5.955・5.952・5.991、形容詞 6.020・6.028・6.049）
SHIFTS_BY_CATEGORY = (0, 1, 2, 3, 1, 3)


@dataclass
class SenseAdditionAssignment:
    run: str
    target: str
    target_category: str
    target_token_count: int
    donor: str
    donor_role: str
    donor_category: str
    # ターゲット語に足す語義の中の順番（1〜N）
    order: int


@dataclass
class OtherCategoryPair:
    target: str
    noun: str
    cosine_similarity: float
    llm_similarity: int
    # 内部表現の類似度と LLM の評定がどちらも、全ての組の中央値以下か
    distant: bool


def run_names() -> list[str]:
    """11の run の名前（足す語義の種類 × 数の9つと、対照の2つ）。"""
    runs = [f"{kind}-{number}" for kind in KINDS for number in NUMBERS_OF_SENSES]
    return [*runs, *CONTROL_RUNS]


def build_sense_addition_assignments(settings: Settings, arguments: argparse.Namespace) -> None:
    step = Step(
        COMMAND,
        inputs={
            word_finalization.OUTPUT_PATH: word_finalization.COMMAND,
            hidden_state_similarity.SIMILARITY_PATH: hidden_state_similarity.COMMAND,
            similarity_ratings.OUTPUT_PATH: similarity_ratings.COMMAND,
        },
        outputs=[ASSIGNMENTS_PATH, PAIR_CHECKS_PATH],
        settings={"categories": [category.name for category in settings.words.categories]},
    )
    run_unless_unchanged(
        step,
        lambda: write_sense_addition_assignments(
            settings,
            word_finalization.OUTPUT_PATH,
            hidden_state_similarity.SIMILARITY_PATH,
            similarity_ratings.OUTPUT_PATH,
            (ASSIGNMENTS_PATH, PAIR_CHECKS_PATH),
        ),
        arguments.recompute,
    )


def write_sense_addition_assignments(
    settings: Settings,
    words_path: Path,
    similarity_path: Path,
    ratings_path: Path,
    output_paths: tuple[Path, Path],
) -> None:
    assignments_path, pair_checks_path = output_paths
    words = read_rows(words_path, SelectedWord)
    categories = [category.name for category in settings.words.categories]
    targets = {(word.category, word.token_count): word for word in words if word.role == "target"}
    distances = _pair_distances(similarity_path, ratings_path)
    partition = nouns_by_token_count(words, settings)
    slots = {}
    for token_count in TOKEN_COUNTS:
        for index, category in enumerate(categories):
            nouns = [noun for noun in partition[token_count] if noun.category == category]
            nouns = _rotated(nouns, _shift(index))
            # 枠 (k, j) の n 番目の名詞は、別カテゴリで、ターゲット語 (k - n, j) に足す
            partners = [
                targets[(categories[(index - order) % len(categories)], token_count)]
                for order in range(1, DONORS_PER_TARGET + 1)
            ]
            slots[(category, token_count)] = _order_slot(nouns, partners, distances)
    adjectives = _deal_adjectives(words, list(targets.values()), categories)
    assignments = []
    for run in run_names():
        if run in CONTROL_RUNS:
            continue
        kind, _, number = run.rpartition("-")
        for (category, token_count), target in targets.items():
            index = categories.index(category)
            if kind == "same-category":
                donors = slots[(category, token_count)][: int(number)]
            elif kind == "other-category":
                donors = [
                    slots[(categories[(index + order) % len(categories)], token_count)][order - 1]
                    for order in range(1, int(number) + 1)
                ]
            else:
                donors = adjectives[target.word][: int(number)]
            assignments += [
                SenseAdditionAssignment(
                    run,
                    target.word,
                    target.category,
                    target.token_count,
                    donor.word,
                    donor.role,
                    donor.category,
                    order,
                )
                for order, donor in enumerate(donors, start=1)
            ]
    write_rows(assignments_path, SenseAdditionAssignment, assignments)
    write_rows(
        pair_checks_path,
        OtherCategoryPair,
        [
            OtherCategoryPair(target, noun, *distances[(target, noun)])
            for target, noun in sorted(
                (row.target, row.donor)
                for row in assignments
                if row.run == f"other-category-{max(NUMBERS_OF_SENSES)}"
            )
        ],
    )


def _pair_distances(
    similarity_path: Path, ratings_path: Path
) -> dict[tuple[str, str], tuple[float, int, bool]]:
    """ターゲット語と名詞の組ごとの、内部表現の類似度（全モデルの平均）・LLM の評定・遠いか。"""
    cosines: dict[tuple[str, str], list[float]] = {}
    for row in read_rows(similarity_path, HiddenStateSimilarity):
        cosines.setdefault((row.target, row.noun), []).append(row.cosine_similarity)
    ratings = {
        (row.target, row.noun): row.llm_similarity
        for row in read_rows(ratings_path, SimilarityRating)
    }
    mean_cosines = {pair: float(np.mean(values)) for pair, values in cosines.items()}
    cosine_median = float(np.median(list(mean_cosines.values())))
    rating_median = float(np.median([value for value in ratings.values() if value > 0]))
    return {
        pair: (
            cosine,
            ratings[pair],
            cosine <= cosine_median and 0 < ratings[pair] <= rating_median,
        )
        for pair, cosine in mean_cosines.items()
    }


def _order_slot(
    nouns: Sequence[SelectedWord],
    partners: Sequence[SelectedWord],
    distances: Mapping[tuple[str, str], tuple[float, int, bool]],
) -> list[SelectedWord]:
    """別カテゴリの組（n 番目の名詞と n 番目の相手）が遠くなる並べ替え（元の順に近いものから）。"""

    def near_pairs(order: Sequence[SelectedWord]) -> int:
        return sum(
            not distances[(partner.word, noun.word)][2]
            for noun, partner in zip(order, partners, strict=True)
        )

    orders = list(itertools.permutations(nouns))
    return list(min(orders, key=near_pairs))


def _deal_adjectives(
    words: Sequence[SelectedWord], targets: Sequence[SelectedWord], categories: Sequence[str]
) -> dict[str, list[SelectedWord]]:
    """形容詞を頻度の順に、ターゲット語へ 1〜18・18〜1・… の順に配る（順番は、ターゲット語の
    カテゴリの順番だけずらす）。
    """
    adjectives = sorted(
        (word for word in words if word.role == "adjective"),
        key=lambda adjective: (adjective.estimated_pile_count, adjective.word),
    )
    dealt: dict[str, list[SelectedWord]] = {target.word: [] for target in targets}
    snake = [*targets, *reversed(targets)]
    for index, adjective in enumerate(adjectives):
        dealt[snake[index % len(snake)].word].append(adjective)
    return {
        target.word: _rotated(dealt[target.word], _shift(categories.index(target.category)))
        for target in targets
    }


def _shift(category_index: int) -> int:
    return SHIFTS_BY_CATEGORY[category_index % len(SHIFTS_BY_CATEGORY)]


def _rotated(words: Sequence[SelectedWord], shift: int) -> list[SelectedWord]:
    """頻度の低い順の語を、shift だけずらした順（shift 番目の語が1番目になる）。"""
    start = shift % len(words) if words else 0
    return [*words[start:], *words[:start]]
