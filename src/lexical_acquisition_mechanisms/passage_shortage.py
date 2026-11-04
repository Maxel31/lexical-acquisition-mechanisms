"""語と分け方ごとの、判定に合格する必要がある用例の数と、数が足りない語で止まる処理。

passages judge-word-sense（判定に回す候補の数を、判定の前に）と passages finalize（判定に合格した
用例の数を）が使う。足りない語があれば止まる。
- Pile の出現を全て残しても足りない語は、data/words/excluded_words.csv に足す（words select から
  選び直すと、代わりの語が選ばれる）。
- 残した出現が残す数に届いた語（Pile にはさらに出現がありうる）は外さず、
  passages.collection_factor を大きくして passages collect からやり直すように示す。
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from lexical_acquisition_mechanisms import passage_collection, word_selection
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.csv_tables import read_rows
from lexical_acquisition_mechanisms.passage_collection import CollectionSummary
from lexical_acquisition_mechanisms.settings import Settings
from lexical_acquisition_mechanisms.word_selection import ExcludedWord, SelectedWord

SPLITS = ("training", "evaluation")


@dataclass(frozen=True)
class Shortage:
    word: str
    role: str
    split: str
    found: int
    needed: int


def needed_passing_passages(role: str, split: str, settings: Settings) -> int:
    """語の役割と分け方ごとの、合格する必要がある用例の数（ターゲット語と自然な語は評価用だけ）。"""
    if split == "training":
        if role == "noun":
            return max(settings.unknown_word_acquisition.exposure_levels)
        if role == "adjective":
            return settings.sense_addition.passages_per_added_sense
        return 0
    if role.startswith("natural-"):
        return settings.evaluation.passages_per_natural_word
    return settings.passages.evaluation_passages_per_word


def find_shortages(
    words: Sequence[SelectedWord],
    counts: Mapping[tuple[str, str], int],
    settings: Settings,
    factor: float = 1.0,
) -> list[Shortage]:
    """語と分け方ごとに、数（counts）が、合格する必要がある数 × factor に届かない組。"""
    shortages = []
    for word in words:
        for split in SPLITS:
            needed = needed_passing_passages(word.role, split, settings)
            if counts.get((word.word, split), 0) < needed * factor:
                found = counts.get((word.word, split), 0)
                shortages.append(Shortage(word.word, word.role, split, found, needed))
    return shortages


def stop_for_shortage(
    shortages: Sequence[Shortage],
    what: str,
    collection_summary_path: Path,
    excluded_words_path: Path,
) -> NoReturn:
    """what（例: 判定に合格した用例）が足りない語で止まる。

    Pile にさらに出現がありうる語は外さない。
    """
    at_limit = {
        row.word
        for row in read_rows(collection_summary_path, CollectionSummary)
        if row.reached_limit
    }
    exhausted = [shortage for shortage in shortages if shortage.word not in at_limit]
    more_in_pile = [shortage for shortage in shortages if shortage.word in at_limit]
    messages = []
    if exhausted:
        word_selection.add_excluded_words(
            excluded_words_path,
            [
                ExcludedWord(
                    shortage.word,
                    shortage.role,
                    f"{what}が足りない（{shortage.split} {shortage.found} / {shortage.needed} 件）",
                )
                for shortage in exhausted
            ],
        )
        messages.append(
            f"Pile の出現を全て残しても、{what}が足りない語を、{excluded_words_path} に"
            f"足しました（{listed(exhausted)}）。words select から実行し直すと、代わりの語が"
            "選ばれます。"
        )
    if more_in_pile:
        messages.append(
            f"{what}が足りない語のうち、Pile にさらに出現がありうる語は外していません"
            f"（{listed(more_in_pile)}）。settings.yaml の passages.collection_factor を大きくして"
            f"ください（外すときは、{excluded_words_path} に足して words select から）。"
        )
    raise CommandStopped(
        "".join(messages),
        rerun_from=word_selection.COMMAND if exhausted else passage_collection.COMMAND,
    )


def listed(shortages: Sequence[Shortage]) -> str:
    return ", ".join(
        f"{shortage.word}（{shortage.split} {shortage.found} / {shortage.needed} 件）"
        for shortage in shortages
    )
