"""未知語の獲得の割り当て（assignments build --experiment unknown-word-acquisition）。

run ごとに、各未知語に、どの名詞の用例を何件差し込むか（露出回数）を決める。
1. 未知語72個と名詞72語を、8個と8語ずつの9つの組にする。
   - トークン数（1・2・3）ごとに3組。2・3トークンの組は、near 4個・far 4個。
   - 名詞は、カテゴリごとに頻度の順に、トークン数へ 1・2・3・3・2・1・… の順に配る（各トークン数に
     各カテゴリ4語）。トークン数の中では、カテゴリの順に、3つの組へ順に配る。
2. 組の中で、run r（0〜7）の、未知語 c（0〜7）の露出回数の番号を r + c、名詞の番号を 2r + c に
   する（8元体 GF(8) での計算。2つは直交するラテン方格になる）。
   - 各未知語は、8つの run で、8つの露出回数に1回ずつ当たり、組の8語の名詞と1回ずつ組む。
   - 名詞と露出回数の組み合わせは重ならない。1つの run の中で、同じ名詞を2つの未知語に
     割り当てない。
3. 露出回数0の未知語は、各 run で各組に1個ある（run r では、組の中の番号 r の未知語）。2・3トークン
   では、各 run に near と far の両方が入るように、組の中の並びを組ごとに変える（1組目は near を
   前半に、2組目は後半に置く）。正規化の基準（露出0回の、同じトークン数・近さの未知語の平均）に
   使うため。
対照の run（control）には何も差し込まない。
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from lexical_acquisition_mechanisms import word_finalization
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.settings import Settings
from lexical_acquisition_mechanisms.word_selection import SelectedWord

EXPERIMENT = "unknown-word-acquisition"
COMMAND = f"assignments build --experiment {EXPERIMENT}"
OUTPUT_DIRECTORY = Path("data/assignments") / EXPERIMENT
ASSIGNMENTS_PATH = OUTPUT_DIRECTORY / "assignments.csv"
OVERVIEW_PATH = OUTPUT_DIRECTORY / "overview.csv"
GROUP_SIZE = 8
GROUPS_PER_TOKEN_COUNT = 3
TOKEN_COUNTS = (1, 2, 3)
# GF(8) の掛け算に使う既約多項式 x^3 + x + 1
IRREDUCIBLE_POLYNOMIAL = 0b1011
# 名詞の番号を決めるラテン方格の係数（GF(8) の元 x）
NOUN_SQUARE_FACTOR = 2


@dataclass
class UnknownWordAssignment:
    run: str
    unknown_word: str
    token_count: int
    spelling_closeness: str
    # 組の番号（1〜9）と、組の中の番号（0〜7）
    group: int
    position: int
    noun: str
    noun_category: str
    exposure: int


def run_names() -> list[str]:
    """割り当てのある8つの run の名前（run-1〜run-8）。対照の run は control。"""
    return [f"run-{number}" for number in range(1, GROUP_SIZE + 1)]


def build_unknown_word_assignments(settings: Settings, arguments: argparse.Namespace) -> None:
    step = Step(
        COMMAND,
        inputs={word_finalization.OUTPUT_PATH: word_finalization.COMMAND},
        outputs=[ASSIGNMENTS_PATH, OVERVIEW_PATH],
        settings={
            "unknown_word_acquisition": settings.unknown_word_acquisition.model_dump(
                include={"exposure_levels"}
            ),
            "categories": [category.name for category in settings.words.categories],
        },
    )
    run_unless_unchanged(
        step,
        lambda: write_unknown_word_assignments(
            settings, word_finalization.OUTPUT_PATH, ASSIGNMENTS_PATH, OVERVIEW_PATH
        ),
        arguments.recompute,
    )


def write_unknown_word_assignments(
    settings: Settings, words_path: Path, assignments_path: Path, overview_path: Path
) -> None:
    words = read_rows(words_path, SelectedWord)
    levels = settings.unknown_word_acquisition.exposure_levels
    assignments = []
    for group_number, (unknown_words, nouns) in enumerate(_groups(words, settings), start=1):
        for run_index, run in enumerate(run_names()):
            for position, unknown_word in enumerate(unknown_words):
                noun = nouns[_add(_multiply(NOUN_SQUARE_FACTOR, run_index), position)]
                assignments.append(
                    UnknownWordAssignment(
                        run,
                        unknown_word.word,
                        unknown_word.token_count,
                        unknown_word.spelling_closeness,
                        group_number,
                        position,
                        noun.word,
                        noun.category,
                        levels[_add(run_index, position)],
                    )
                )
    write_rows(assignments_path, UnknownWordAssignment, assignments)
    _write_overview(assignments, overview_path)


def _groups(
    words: list[SelectedWord], settings: Settings
) -> list[tuple[list[SelectedWord], list[SelectedWord]]]:
    """9つの組（未知語8個と名詞8語）。"""
    unknown_by_token_count = {
        token_count: _unknown_word_groups(
            [word for word in words if word.role == "unknown" and word.token_count == token_count]
        )
        for token_count in TOKEN_COUNTS
    }
    nouns_by_token_count: dict[int, list[SelectedWord]] = {count: [] for count in TOKEN_COUNTS}
    snake = [*TOKEN_COUNTS, *reversed(TOKEN_COUNTS)]
    for category in settings.words.categories:
        nouns = sorted(
            (word for word in words if word.role == "noun" and word.category == category.name),
            key=lambda noun: (noun.estimated_pile_count, noun.word),
        )
        for index, noun in enumerate(nouns):
            nouns_by_token_count[snake[index % len(snake)]].append(noun)
    groups = []
    for token_count in TOKEN_COUNTS:
        nouns = nouns_by_token_count[token_count]
        noun_groups = [
            nouns[start::GROUPS_PER_TOKEN_COUNT] for start in range(GROUPS_PER_TOKEN_COUNT)
        ]
        groups += list(zip(unknown_by_token_count[token_count], noun_groups, strict=True))
    for unknown_words, nouns in groups:
        if len(unknown_words) != GROUP_SIZE or len(nouns) != GROUP_SIZE:
            raise ValueError(
                f"組の未知語と名詞は {GROUP_SIZE} 個ずつにしてください"
                f"（未知語 {len(unknown_words)} 個、名詞 {len(nouns)} 語）。"
            )
    return groups


def _unknown_word_groups(unknown_words: list[SelectedWord]) -> list[list[SelectedWord]]:
    """同じトークン数の未知語を3組にする。

    2・3トークンは near 4個・far 4個にし、near を、1組目は前半、2組目は後半、3組目は前後の半分ずつに
    置く。
    """
    if unknown_words[0].token_count == 1:
        starts = range(0, len(unknown_words), GROUP_SIZE)
        return [unknown_words[start : start + GROUP_SIZE] for start in starts]
    half = GROUP_SIZE // 2
    near = [word for word in unknown_words if word.spelling_closeness == "near"]
    far = [word for word in unknown_words if word.spelling_closeness == "far"]
    groups = []
    for index in range(GROUPS_PER_TOKEN_COUNT):
        group_near = near[index * half : (index + 1) * half]
        group_far = far[index * half : (index + 1) * half]
        if index == 0:
            groups.append(group_near + group_far)
        elif index == 1:
            groups.append(group_far + group_near)
        else:
            groups.append(group_near[:2] + group_far[:2] + group_near[2:] + group_far[2:])
    return groups


def _add(first: int, second: int) -> int:
    """GF(8) の足し算（ビットごとの排他的論理和）。"""
    return first ^ second


def _multiply(first: int, second: int) -> int:
    """GF(8) の掛け算（x^3 + x + 1 で割った余りの、多項式の掛け算）。"""
    product = 0
    for bit in range(3):
        if second >> bit & 1:
            product ^= first << bit
    for bit in (4, 3):
        if product >> bit & 1:
            product ^= IRREDUCIBLE_POLYNOMIAL << (bit - 3)
    return product


def _write_overview(assignments: list[UnknownWordAssignment], path: Path) -> None:
    """人が確かめやすい一覧（行が未知語、列が run、セルが「名詞 / 露出回数」）。"""
    runs = run_names()
    cells: dict[str, dict[str, str]] = {}
    for assignment in assignments:
        row = cells.setdefault(assignment.unknown_word, {"unknown_word": assignment.unknown_word})
        row[assignment.run] = f"{assignment.noun} / {assignment.exposure}"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["unknown_word", *runs])
        writer.writeheader()
        writer.writerows(cells.values())
