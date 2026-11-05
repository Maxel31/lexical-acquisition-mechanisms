"""語を選ぶ（words select）。

語の候補（words generate-noun-candidates などの CSV）から、実験で操作する語（管理対象の語）を
選び、data/words/selected_words.csv に書く。
- 名詞72語: 各カテゴリ12語。先頭スペース付きで1トークン。
- ターゲット語18語: 各カテゴリから、先頭スペース付きで1・2・3トークンの語を1語ずつ。最初の
  トークンが、名詞やほかのターゲット語と重ならない。
- 形容詞72語: 先頭スペース付きで1トークン。どのカテゴリとも結びつきが強くない。語義の追加で
  足す語義1つあたりに差し込む用例の数（K）を、露出回数の最大まで大きくしても用例が足りるように、
  名詞と同じ数の用例が取れる候補から選ぶ。
- 未知語72個: 1トークンの24個は、使われていないトークン番号（表記は <new_01> などの管理用の名前）。
  2・3トークンの各24個は、一度も現れない候補から、綴りの近さ（near・far）ごとに12個ずつ、候補の
  無作為な順に選ぶ。最初のトークンが、ほかの未知語や管理対象の語と重ならず、near の元の語が
  管理対象の語でないものに限る。
- 自然な語（実験で操作しない実在語。課題の例やプローブの学習に使う）: 各カテゴリ30語の名詞と、
  形容詞60語。名詞は、カテゴリごとに1・2・3トークンの語を順に（なければ、あるトークン数の語を）、
  無作為に選ぶ。品詞のプローブには、形容詞と同じ1トークンの名詞を、各カテゴリ10語使う。
どの語も、Pile の見本で意図した語義（形容詞は、形容詞として）で使われていた割合が高く、Pile で
取れる用例の見積もり（意図した使われ方の割合を掛けたもの）が必要な数より十分に多い候補から選ぶ。
すでに選んだ語の同義語（WordNet で、その語を見出しとする語義が重なる語）は選ばない。
頻度は、Pile での標準の形の出現回数の対数で比べる。
- 名詞は、カテゴリの間で頻度の分布をそろえる。全カテゴリに候補がある頻度の範囲に、12個の目標の値を
  等間隔に置き、カテゴリごとに、それぞれの値に最も近い候補を選ぶ。
- ターゲット語は、トークン数の間で頻度をそろえる。多トークンの候補の頻度のうち、各組（カテゴリと
  トークン数）で最も近い候補との差の合計が最も小さい値を目標にし、組ごとに最も近い候補を選ぶ
  （多トークンの語は低頻度に偏るので、そちらに合わせる）。
- 形容詞は、名詞72語と頻度の分布をそろえる。頻度の低い名詞から順に、最も近い候補を選ぶ。
選んだ語ごとに、基データでの出現回数（分析の共変量）も数える。
data/words/excluded_words.csv にある語（確認で外した語など）は選ばない。
"""

import argparse
import math
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

from lexical_acquisition_mechanisms import (
    adjective_candidates,
    base_data,
    noun_candidates,
    unknown_word_candidates,
)
from lexical_acquisition_mechanisms.adjective_candidates import AdjectiveCandidate
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.noun_candidates import NounCandidate
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.settings import Settings
from lexical_acquisition_mechanisms.unknown_word_candidates import UnknownWordCandidate
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer
from lexical_acquisition_mechanisms.wordnet_senses import adjective_senses, noun_senses

COMMAND = "words select"
OUTPUT_PATH = Path("data/words/selected_words.csv")
EXCLUDED_WORDS_PATH = Path("data/words/excluded_words.csv")
NOUNS_PER_CATEGORY = 12
TARGET_TOKEN_COUNTS = (1, 2, 3)
NATURAL_NOUNS_PER_CATEGORY = 30
NATURAL_ADJECTIVES = 60
# 品詞のプローブに使う、カテゴリごとの自然な名詞の数
PART_OF_SPEECH_NOUNS_PER_CATEGORY = 10
# 1トークンの未知語の数と、2・3トークンの未知語の、トークン数と綴りの近さの組ごとの数
ONE_TOKEN_UNKNOWN_WORDS = 24
UNKNOWN_WORDS_PER_GROUP = 12


@dataclass(frozen=True)
class SelectionInputs:
    """words select が読む、候補の CSV と基データと、外す語の一覧（なくてもよい）。"""

    nouns: Path = noun_candidates.OUTPUT_PATH
    adjectives: Path = adjective_candidates.OUTPUT_PATH
    unknown_words: Path = unknown_word_candidates.OUTPUT_PATH
    base_sequences: Path = base_data.output_paths()["base"]
    excluded_words: Path = EXCLUDED_WORDS_PATH


@dataclass
class ExcludedWord:
    word: str
    role: str
    # 外した理由
    reason: str


@dataclass
class SelectedWord:
    word: str
    # 役割（noun: 名詞72語、target: ターゲット語、adjective: 形容詞72語、unknown: 未知語、
    # natural-noun・natural-adjective: 自然な名詞・形容詞）
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
    # Pile の見本で、意図した語義（形容詞は、形容詞として）で使われていた割合
    usage_share: float = 0.0
    # 品詞のプローブの学習に使う自然な語か
    part_of_speech_probe: bool = False
    # 未知語の綴りの近さ（near・far。1トークンの未知語は空）と、near の元の実在語
    spelling_closeness: str = ""
    source_word: str = ""
    # 基データでの、先頭スペース付きの形の出現回数（直後に英字が続くものは数えない）
    base_data_count: int = 0


class ChosenWords:
    """選んだ語と、その語義（WordNet の synset）。選んだ語と、その同義語は、もう選ばない。"""

    def __init__(
        self, senses: Callable[[str], frozenset[str]], excluded: Iterable[str] = ()
    ) -> None:
        self._senses = senses
        # 外す語は、選んだ語と同じく選ばない（同義語は選んでよい）
        self._words: set[str] = set(excluded)
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
        "passages": settings.passages.model_dump(include={"evaluation_passages_per_word"}),
        "unknown_word_acquisition": settings.unknown_word_acquisition.model_dump(
            include={"exposure_levels"}
        ),
        "evaluation": settings.evaluation.model_dump(include={"sentences_per_natural_word"}),
        "tokenizer": next(iter(settings.models.values())).model_dump(),
    }
    tables = SelectionInputs()
    # 外す語の一覧は、人の確認から書かれる（処理の記録がない）ので、中身を設定として記録する
    excluded = read_excluded_words(tables.excluded_words)
    record_settings["excluded_words"] = sorted({row.word for row in excluded})
    inputs = {
        tables.nouns: noun_candidates.COMMAND,
        tables.adjectives: adjective_candidates.COMMAND,
        tables.unknown_words: unknown_word_candidates.COMMAND,
        tables.base_sequences: base_data.COMMAND,
    }
    step = Step(COMMAND, inputs=inputs, outputs=[OUTPUT_PATH], settings=record_settings)
    run_unless_unchanged(
        step,
        lambda: build_selected_words(settings, tables, OUTPUT_PATH),
        arguments.recompute,
    )


def build_selected_words(settings: Settings, tables: SelectionInputs, output_path: Path) -> None:
    tokenizer = WordTokenizer.from_model(next(iter(settings.models.values())))
    listed_nouns = read_rows(tables.nouns, NounCandidate)
    listed_adjectives = read_rows(tables.adjectives, AdjectiveCandidate)
    excluded = {row.word for row in read_excluded_words(tables.excluded_words)}
    chosen = ChosenWords(_synsets, excluded)
    nouns = _select_nouns(listed_nouns, settings, chosen)
    targets = _select_target_words(listed_nouns, settings, chosen, nouns, tokenizer)
    adjectives = _select_adjectives(listed_adjectives, settings, chosen, nouns)
    rows = [_row(noun, "noun", tokenizer) for noun in nouns]
    rows += [_row(target, "target", tokenizer) for target in targets]
    rows += [_row(adjective, "adjective", tokenizer) for adjective in adjectives]
    listed_unknown_words = read_rows(tables.unknown_words, UnknownWordCandidate)
    rows += _select_unknown_words(listed_unknown_words, rows, tokenizer)
    rows += _select_natural_words(listed_nouns, listed_adjectives, settings, chosen, tokenizer)
    counts = base_data.count_word_occurrences(
        np.load(tables.base_sequences, mmap_mode="r"),
        {row.word: [int(token) for token in row.token_ids.split()] for row in rows},
        tokenizer.word_boundaries().starts_with_letter,
    )
    for row in rows:
        row.base_data_count = counts[row.word]
    write_rows(output_path, SelectedWord, rows)


def read_excluded_words(path: Path = EXCLUDED_WORDS_PATH) -> list[ExcludedWord]:
    return read_rows(path, ExcludedWord) if path.exists() else []


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


def _select_target_words(
    candidates: Sequence[NounCandidate],
    settings: Settings,
    chosen: ChosenWords,
    nouns: Sequence[NounCandidate],
    tokenizer: WordTokenizer,
) -> list[NounCandidate]:
    """各カテゴリの、1・2・3トークンの名詞を1語ずつ。トークン数の間で頻度をそろえる。"""
    needed = settings.passages.evaluation_passages_per_word
    first_token = {
        candidate.word: tokenizer.word_token_ids(candidate.word)[0]
        for candidate in candidates
        if not candidate.rejected_reason
    }
    taken_first_tokens = {first_token[noun.word] for noun in nouns}

    def available(candidate: NounCandidate) -> bool:
        return (
            chosen.can_take(candidate.word)
            and first_token[candidate.word] not in taken_first_tokens
        )

    cells = {
        (category.name, token_count): [
            candidate
            for candidate in candidates
            if candidate.category == category.name
            and candidate.token_count == token_count
            and _usable(candidate, settings, needed)
            and available(candidate)
        ]
        for category in settings.words.categories
        for token_count in TARGET_TOKEN_COUNTS
    }
    for (name, token_count), usable in cells.items():
        if not usable:
            description = f"カテゴリ {name} の、{token_count}トークンのターゲット語"
            raise _shortage(description, 0, needed, settings)
    multi_token_frequencies = [
        _frequency(candidate)
        for (_, token_count), usable in cells.items()
        if token_count > 1
        for candidate in usable
    ]
    goal = min(
        multi_token_frequencies,
        key=lambda value: (
            sum(_distance(_nearest(usable, value), value) for usable in cells.values()),
            value,
        ),
    )
    targets = []
    for (name, token_count), usable in cells.items():
        remaining = [candidate for candidate in usable if available(candidate)]
        if not remaining:
            description = f"カテゴリ {name} の、{token_count}トークンのターゲット語（同義語を除く）"
            raise _shortage(description, 0, needed, settings)
        target = _nearest(remaining, goal)
        chosen.take(target.word)
        taken_first_tokens.add(first_token[target.word])
        targets.append(target)
    return targets


def _select_adjectives(
    candidates: Sequence[AdjectiveCandidate],
    settings: Settings,
    chosen: ChosenWords,
    nouns: Sequence[NounCandidate],
) -> list[AdjectiveCandidate]:
    """名詞72語と同じ数の形容詞。名詞と頻度の分布をそろえる。"""
    needed = (
        max(settings.unknown_word_acquisition.exposure_levels)
        + settings.passages.evaluation_passages_per_word
    )
    maximum_association = settings.word_selection.maximum_category_association
    usable = [
        candidate
        for candidate in candidates
        if _usable(candidate, settings, needed)
        and candidate.largest_category_association <= maximum_association
    ]
    adjectives = []
    for noun in sorted(nouns, key=_frequency):
        remaining = [candidate for candidate in usable if chosen.can_take(candidate.word)]
        if not remaining:
            description = f"形容詞（カテゴリとの結びつきが {maximum_association} 以下）"
            raise _shortage(
                description,
                len(adjectives),
                needed,
                settings,
                listing="words.adjectives_per_listing",
                rerun_from=adjective_candidates.COMMAND,
            )
        adjective = _nearest(remaining, _frequency(noun))
        chosen.take(adjective.word)
        adjectives.append(adjective)
    return adjectives


def _select_unknown_words(
    candidates: Sequence[UnknownWordCandidate],
    managed: Sequence[SelectedWord],
    tokenizer: WordTokenizer,
) -> list[SelectedWord]:
    """未知語。最初のトークンが、ほかの未知語や管理対象の語（managed）と重ならないものを選ぶ。"""
    new_token_ids = tokenizer.unused_token_ids(ONE_TOKEN_UNKNOWN_WORDS)
    rows = [
        SelectedWord(f"<new_{number:02d}>", "unknown", token_count=1, token_ids=str(token_id))
        for number, token_id in enumerate(new_token_ids, start=1)
    ]
    taken_first_tokens = {row.token_ids.split()[0] for row in managed}
    managed_words = {row.word for row in managed}
    for token_count in (2, 3):
        for closeness in ("near", "far"):
            group = []
            for candidate in candidates:
                first_token = candidate.token_ids.split()[0]
                if (
                    len(group) < UNKNOWN_WORDS_PER_GROUP
                    and candidate.occurrences == 0
                    and candidate.token_count == token_count
                    and candidate.spelling_closeness == closeness
                    and first_token not in taken_first_tokens
                    and candidate.source_word not in managed_words
                ):
                    taken_first_tokens.add(first_token)
                    group.append(_unknown_word_row(candidate))
            if len(group) < UNKNOWN_WORDS_PER_GROUP:
                raise CommandStopped(
                    f"{token_count}トークン・{closeness} の未知語で、最初のトークンがほかの語と"
                    f"重ならない候補が足りません（{len(group)} 個）。settings.yaml の"
                    " unknown_words.candidates_per_group を増やしてください。",
                    rerun_from=unknown_word_candidates.COMMAND,
                )
            rows += group
    return rows


def _unknown_word_row(candidate: UnknownWordCandidate) -> SelectedWord:
    return SelectedWord(
        candidate.word,
        "unknown",
        token_count=candidate.token_count,
        token_ids=candidate.token_ids,
        spelling_closeness=candidate.spelling_closeness,
        source_word=candidate.source_word,
    )


def _select_natural_words(
    listed_nouns: Sequence[NounCandidate],
    listed_adjectives: Sequence[AdjectiveCandidate],
    settings: Settings,
    chosen: ChosenWords,
    tokenizer: WordTokenizer,
) -> list[SelectedWord]:
    """自然な名詞（各カテゴリ30語）と形容詞（60語）を、無作為に選ぶ。"""
    needed = settings.evaluation.sentences_per_natural_word
    random_numbers = random.Random(settings.words.seed)
    rows = []
    for category in settings.words.categories:
        buckets = []
        for token_count in TARGET_TOKEN_COUNTS:
            bucket = [
                candidate
                for candidate in listed_nouns
                if candidate.category == category.name
                and candidate.token_count == token_count
                and _usable(candidate, settings, needed)
            ]
            random_numbers.shuffle(bucket)
            buckets.append(bucket)
        natural_nouns = _take_in_turn(buckets, NATURAL_NOUNS_PER_CATEGORY, chosen)
        if len(natural_nouns) < NATURAL_NOUNS_PER_CATEGORY:
            description = f"カテゴリ {category.name} の自然な名詞"
            raise _shortage(description, len(natural_nouns), needed, settings)
        one_token = [noun.word for noun in natural_nouns if noun.token_count == 1]
        for noun in natural_nouns:
            row = _row(noun, "natural-noun", tokenizer)
            row.part_of_speech_probe = noun.word in one_token[:PART_OF_SPEECH_NOUNS_PER_CATEGORY]
            rows.append(row)
    usable = [candidate for candidate in listed_adjectives if _usable(candidate, settings, needed)]
    random_numbers.shuffle(usable)
    natural_adjectives = _take_in_turn([usable], NATURAL_ADJECTIVES, chosen)
    if len(natural_adjectives) < NATURAL_ADJECTIVES:
        raise _shortage(
            "自然な形容詞",
            len(natural_adjectives),
            needed,
            settings,
            listing="words.adjectives_per_listing",
            rerun_from=adjective_candidates.COMMAND,
        )
    for adjective in natural_adjectives:
        row = _row(adjective, "natural-adjective", tokenizer)
        row.part_of_speech_probe = True
        rows.append(row)
    return rows


def _take_in_turn[Candidate: (NounCandidate, AdjectiveCandidate)](
    buckets: Sequence[list[Candidate]], count: int, chosen: ChosenWords
) -> list[Candidate]:
    """組から順に1つずつ、選べる候補を count 個まで取る（候補がなくなった組は飛ばす）。"""
    taken: list[Candidate] = []
    while len(taken) < count and any(buckets):
        for bucket in buckets:
            while bucket and len(taken) < count:
                candidate = bucket.pop(0)
                if chosen.can_take(candidate.word):
                    chosen.take(candidate.word)
                    taken.append(candidate)
                    break
    return taken


def _usable(
    candidate: NounCandidate | AdjectiveCandidate, settings: Settings, needed_passages: int
) -> bool:
    selection = settings.word_selection
    share = _share(candidate)
    return (
        not candidate.rejected_reason
        and candidate.judged_usages >= selection.minimum_judged_usages
        and share >= selection.minimum_usage_share
        and candidate.estimated_pile_passages * share >= selection.passage_margin * needed_passages
    )


def _share(candidate: NounCandidate | AdjectiveCandidate) -> float:
    """Pile の見本で、意図した語義（形容詞は、形容詞として）で使われていた割合。"""
    if isinstance(candidate, NounCandidate):
        return candidate.intended_usage_share
    return candidate.adjective_usage_share


def _frequency(candidate: NounCandidate | AdjectiveCandidate) -> float:
    return math.log10(max(candidate.estimated_pile_count, 1))


def _distance(candidate: NounCandidate | AdjectiveCandidate, goal: float) -> float:
    return abs(_frequency(candidate) - goal)


def _nearest[Candidate: (NounCandidate, AdjectiveCandidate)](
    candidates: Sequence[Candidate], goal: float
) -> Candidate:
    """頻度が goal に最も近い候補（同じなら、語の順で先のもの）。"""
    return min(candidates, key=lambda candidate: (_distance(candidate, goal), candidate.word))


def _shortage(
    description: str,
    found: int,
    needed_passages: int,
    settings: Settings,
    listing: str = "words.nouns_per_listing",
    rerun_from: str = noun_candidates.COMMAND,
) -> CommandStopped:
    selection = settings.word_selection
    return CommandStopped(
        f"{description}で、条件を満たす候補が足りません（{found} 語）。"
        f"条件: 意図した使われ方の割合が {selection.minimum_usage_share} 以上、"
        f"Pile で取れる用例の見積もりが {selection.passage_margin * needed_passages:,.0f} 件以上。"
        f"{listing} か words.listing_kinds を増やすか、word_selection の条件を緩めてください。",
        rerun_from=rerun_from,
    )


@cache
def _synsets(word: str) -> frozenset[str]:
    """語を見出しとする、名詞と形容詞の語義。"""
    senses = [*noun_senses(word, []), *adjective_senses(word)]
    return frozenset(sense.synset for sense in senses)


def _row(
    candidate: NounCandidate | AdjectiveCandidate, role: str, tokenizer: WordTokenizer
) -> SelectedWord:
    token_ids = tokenizer.word_token_ids(candidate.word)
    return SelectedWord(
        word=candidate.word,
        role=role,
        category=candidate.category if isinstance(candidate, NounCandidate) else "",
        token_count=len(token_ids),
        token_ids=" ".join(map(str, token_ids)),
        synset=candidate.synset,
        definition=candidate.definition,
        estimated_pile_count=candidate.estimated_pile_count,
        estimated_pile_passages=candidate.estimated_pile_passages,
        usage_share=_share(candidate),
    )
