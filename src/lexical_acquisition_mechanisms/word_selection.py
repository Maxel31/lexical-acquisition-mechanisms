"""語を選ぶ（words select）。

語の候補（words generate-noun-candidates の CSV）から、実験で操作する語（管理対象の語）を選び、
data/words/selected_words.csv に書く。
- 名詞72語: 各カテゴリ12語。先頭スペース付きで1トークン。
どの語も、Pile の見本で意図した語義で使われていた割合が高く、Pile で取れる用例の見積もり（意図した
使われ方の割合を掛けたもの）が必要な数より十分に多い候補から選ぶ。すでに選んだ語の同義語
（WordNet で、その語を見出しとする語義が重なる語）は選ばない。
頻度は、Pile での標準の形の出現回数の対数で比べる。
- 名詞は、カテゴリの間で頻度の分布をそろえる。全カテゴリに候補がある頻度の範囲に、12個の目標の値を
  等間隔に置き、カテゴリごとに、それぞれの値に最も近い候補を選ぶ。
"""

import argparse
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from lexical_acquisition_mechanisms import noun_candidates
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.noun_candidates import NounCandidate
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.settings import Settings
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer
from lexical_acquisition_mechanisms.wordnet_senses import noun_senses

COMMAND = "words select"
OUTPUT_PATH = Path("data/words/selected_words.csv")
NOUNS_PER_CATEGORY = 12


@dataclass
class SelectedWord:
    word: str
    # 役割（noun: 名詞72語）
    role: str
    category: str = ""
    token_count: int = 0
    # 先頭スペース付きの形のトークン番号（空白区切り）
    token_ids: str = ""
    synset: str = ""
    definition: str = ""
    # Pile での標準の形の出現回数と、用例として切り出せる文章の数の見積もり
    estimated_pile_count: int = 0
    estimated_pile_passages: int = 0
    # Pile の見本で、意図した語義で使われていた割合
    usage_share: float = 0.0


class ChosenWords:
    """選んだ語と、その語義（WordNet の synset）。選んだ語と、その同義語は、もう選ばない。"""

    def __init__(self, senses: Callable[[str], frozenset[str]]) -> None:
        self._senses = senses
        self._words: set[str] = set()
        self._synsets: set[str] = set()

    def can_take(self, word: str) -> bool:
        return word not in self._words and not self._senses(word) & self._synsets

    def take(self, word: str) -> None:
        self._words.add(word)
        self._synsets |= self._senses(word)


def select_words(settings: Settings, arguments: argparse.Namespace) -> None:
    record_settings = {
        "words": settings.words.model_dump(),
        "word_selection": settings.word_selection.model_dump(),
        "passages": settings.passages.model_dump(),
        "unknown_word_acquisition": settings.unknown_word_acquisition.model_dump(),
        "tokenizer": next(iter(settings.models.values())).model_dump(),
    }
    inputs = {noun_candidates.OUTPUT_PATH: noun_candidates.COMMAND}
    step = Step(COMMAND, inputs=inputs, outputs=[OUTPUT_PATH], settings=record_settings)
    run_unless_unchanged(
        step,
        lambda: build_selected_words(settings, noun_candidates.OUTPUT_PATH, OUTPUT_PATH),
        arguments.recompute,
    )


def build_selected_words(settings: Settings, noun_candidates_path: Path, output_path: Path) -> None:
    tokenizer = WordTokenizer.from_model(next(iter(settings.models.values())))
    candidates = read_rows(noun_candidates_path, NounCandidate)
    chosen = ChosenWords(_noun_synsets)
    nouns = _select_nouns(candidates, settings, chosen)
    rows = [_row(noun, "noun", tokenizer) for noun in nouns]
    write_rows(output_path, SelectedWord, rows)


def _select_nouns(
    candidates: Sequence[NounCandidate], settings: Settings, chosen: ChosenWords
) -> list[NounCandidate]:
    """各カテゴリ12語の、1トークンの名詞。カテゴリの間で頻度の分布をそろえる。"""
    needed = (
        max(settings.unknown_word_acquisition.exposure_levels)
        + settings.passages.evaluation_passages_per_word
    )
    by_category = {
        category.name: [
            candidate
            for candidate in candidates
            if candidate.category == category.name
            and candidate.token_count == 1
            and _usable(candidate, settings, needed)
        ]
        for category in settings.words.categories
    }
    for name, usable in by_category.items():
        if len(usable) < NOUNS_PER_CATEGORY:
            raise _shortage(f"カテゴリ {name} の名詞（1トークン）", len(usable), needed, settings)
    low = max(min(map(_frequency, usable)) for usable in by_category.values())
    high = min(max(map(_frequency, usable)) for usable in by_category.values())
    if low > high:
        raise CommandStopped(
            "名詞の候補の頻度の範囲が、カテゴリの間で重なりません。"
            "words.nouns_per_listing を増やしてください。",
            rerun_from=noun_candidates.COMMAND,
        )
    goals = [
        low + (high - low) * (index + 0.5) / NOUNS_PER_CATEGORY
        for index in range(NOUNS_PER_CATEGORY)
    ]
    nouns = []
    for name, usable in by_category.items():
        for picked, goal in enumerate(goals):
            remaining = [candidate for candidate in usable if chosen.can_take(candidate.word)]
            if not remaining:
                description = f"カテゴリ {name} の名詞（同義語を除く）"
                raise _shortage(description, picked, needed, settings)
            noun = _nearest(remaining, goal)
            chosen.take(noun.word)
            nouns.append(noun)
    return nouns


def _usable(candidate: NounCandidate, settings: Settings, needed_passages: int) -> bool:
    selection = settings.word_selection
    return (
        not candidate.rejected_reason
        and candidate.judged_usages >= selection.minimum_judged_usages
        and candidate.intended_usage_share >= selection.minimum_usage_share
        and candidate.estimated_pile_passages * candidate.intended_usage_share
        >= selection.passage_margin * needed_passages
    )


def _frequency(candidate: NounCandidate) -> float:
    return math.log10(max(candidate.estimated_pile_count, 1))


def _nearest(candidates: Sequence[NounCandidate], goal: float) -> NounCandidate:
    """頻度が goal に最も近い候補（同じなら、語の順で先のもの）。"""
    return min(
        candidates, key=lambda candidate: (abs(_frequency(candidate) - goal), candidate.word)
    )


def _shortage(
    description: str, found: int, needed_passages: int, settings: Settings
) -> CommandStopped:
    selection = settings.word_selection
    return CommandStopped(
        f"{description}で、条件を満たす候補が足りません（{found} 語）。"
        f"条件: 意図した語義で使われる割合が {selection.minimum_usage_share} 以上、"
        f"Pile で取れる用例の見積もりが {selection.passage_margin * needed_passages:,.0f} 件以上。"
        "words.nouns_per_listing か words.listing_kinds を増やすか、"
        "word_selection の条件を緩めてください。",
        rerun_from=noun_candidates.COMMAND,
    )


@cache
def _noun_synsets(word: str) -> frozenset[str]:
    return frozenset(sense.synset for sense in noun_senses(word, []))


def _row(candidate: NounCandidate, role: str, tokenizer: WordTokenizer) -> SelectedWord:
    token_ids = tokenizer.word_token_ids(candidate.word)
    return SelectedWord(
        word=candidate.word,
        role=role,
        category=candidate.category,
        token_count=len(token_ids),
        token_ids=" ".join(map(str, token_ids)),
        synset=candidate.synset,
        definition=candidate.definition,
        estimated_pile_count=candidate.estimated_pile_count,
        estimated_pile_passages=candidate.estimated_pile_passages,
        usage_share=candidate.intended_usage_share,
    )
