"""語を選ぶ（words select）。

語の候補（words generate-noun-candidates などの CSV）から、実験で操作する語（管理対象の語）を
選び、data/words/selected_words.csv に書く。
- 名詞72語: 各カテゴリ12語。先頭スペース付きで1トークン。
- ターゲット語18語: 各カテゴリから、先頭スペース付きで1・2・3トークンの語を1語ずつ。最初の
  トークンが、名詞やほかのターゲット語と重ならない。
- 形容詞72語: 先頭スペース付きで1トークン。どのカテゴリとも結びつきが強くない。語義の追加で
  足す語義1つあたりに差し込む用例の数（K）を、露出回数の最大まで大きくしても用例が足りるように、
  名詞と同じ数の用例が取れる候補から選ぶ。トークンが、名詞やターゲット語の最初のトークンと
  重ならない（例: ターゲット語 " high|chair" と形容詞 " high"。重なると、そのトークンの学習の量が
  run の間で変わるため）。
- 未知語72個: 1トークンの24個は、使われていないトークン番号（表記は <new_01> などの管理用の名前）。
  2・3トークンの各24個は、一度も現れない候補から、綴りの近さ（near・far）ごとに12個ずつ、候補の
  無作為な順に選ぶ。最初のトークンが、ほかの未知語や管理対象の語と重ならず、near の元の語が
  管理対象の語でないものに限る。
- 自然な語（実験で操作しない実在語。課題の例やプローブの学習に使う）: 各カテゴリ30語の名詞と、
  形容詞60語。名詞は、カテゴリごとに1・2・3トークンの語を順に（なければ、あるトークン数の語を）、
  無作為に選ぶ。品詞のプローブには、形容詞と同じ1トークンの名詞を、形容詞と同じ60語使う（カテゴリから
  順に1語ずつ。1トークンの自然な名詞が足りなければ、1トークンの候補が残るカテゴリから順に、自然な
  名詞を足す。足したカテゴリの自然な名詞は30語より多くなる）。
どの語も、Pile の見本で意図した語義（形容詞は、形容詞として）で使われていた割合が高く（自然な語は、
それより低い下限。設計書は自然な語に単義を求めていない）、Pile で取れる用例の見積もり（意図した
使われ方の割合を掛けたもの）が必要な数より十分に多い候補から選ぶ。
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
# 実験で操作する語（管理対象の語）の役割
MANAGED_ROLES = ("noun", "target", "adjective")
NOUNS_PER_CATEGORY = 12
TARGET_TOKEN_COUNTS = (1, 2, 3)
NATURAL_NOUNS_PER_CATEGORY = 30
NATURAL_ADJECTIVES = 60
# 品詞のプローブに使う、1トークンの自然な名詞の数（自然な形容詞と同じ数）
PART_OF_SPEECH_NOUNS = 60
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

    def can_take(self, word: str, allowed_synsets: frozenset[str] = frozenset()) -> bool:
        """allowed_synsets の語義は、選んだ語と重なっても選んでよい（同義語を許すときに使う）。"""
        overlap = (self._senses(word) & self._synsets) - allowed_synsets
        return word not in self._words and not overlap

    @property
    def synsets(self) -> frozenset[str]:
        return frozenset(self._synsets)

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
        "evaluation": settings.evaluation.model_dump(include={"passages_per_natural_word"}),
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
    taken_first_tokens = {tokenizer.word_token_ids(word.word)[0] for word in [*nouns, *targets]}
    adjectives = _select_adjectives(
        listed_adjectives, settings, chosen, nouns, taken_first_tokens, tokenizer
    )
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


def add_excluded_words(path: Path, words: Iterable[ExcludedWord]) -> None:
    """外す語を足す（同じ語と役割の組が既にあれば、足さない）。"""
    excluded = read_excluded_words(path)
    known = {(row.word, row.role) for row in excluded}
    for word in words:
        if (word.word, word.role) not in known:
            excluded.append(word)
            known.add((word.word, word.role))
    write_rows(path, ExcludedWord, excluded)


def _select_nouns(
    candidates: Sequence[NounCandidate], settings: Settings, chosen: ChosenWords
) -> list[NounCandidate]:
    """各カテゴリ12語の、1トークンの名詞。カテゴリの間で頻度の分布をそろえる。"""
    needed = (
        max(settings.unknown_word_acquisition.exposure_levels)
        + settings.passages.evaluation_passages_per_word
    )
    shares = settings.word_selection.minimum_noun_usage_share_by_category
    by_category = {
        category.name: [
            candidate
            for candidate in candidates
            if candidate.category == category.name
            and candidate.token_count == 1
            and _usable(candidate, settings, needed, minimum_share=shares.get(category.name))
        ]
        for category in settings.words.categories
    }
    for name, usable in by_category.items():
        if len(usable) < NOUNS_PER_CATEGORY:
            description = f"カテゴリ {name} の名詞（1トークン）"
            raise _shortage(description, len(usable), needed, settings, share=shares.get(name))
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
    allowing_synonyms = settings.word_selection.categories_allowing_synonyms
    for name, usable in by_category.items():
        before = chosen.synsets
        for picked, goal in enumerate(goals):
            # 同義語を許すカテゴリでは、このカテゴリで選んだ名詞の語義だけは重なってもよい
            # （ほかのカテゴリの語の同義語は選ばない）
            own = chosen.synsets - before if name in allowing_synonyms else frozenset()
            remaining = [candidate for candidate in usable if chosen.can_take(candidate.word, own)]
            if not remaining:
                description = f"カテゴリ {name} の名詞（同義語を除く）"
                raise _shortage(description, picked, needed, settings, share=shares.get(name))
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
    taken_first_tokens: set[int],
    tokenizer: WordTokenizer,
) -> list[AdjectiveCandidate]:
    """名詞72語と同じ数の形容詞。名詞と頻度の分布をそろえる。

    トークンが taken_first_tokens（名詞とターゲット語の最初のトークン）にある候補は使わない。
    """
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
        and tokenizer.word_token_ids(candidate.word)[0] not in taken_first_tokens
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
    """自然な名詞（各カテゴリ30語。品詞のプローブ用に足す語を除く）と形容詞（60語）を、無作為に選ぶ。

    自然な名詞の候補が30語に足りないカテゴリは、word_selection.minimum_natural_nouns_per_category
    語以上あれば、あるだけ選んで表示する（それより少なければ止まる）。
    """
    needed = settings.evaluation.passages_per_natural_word
    random_numbers = random.Random(settings.words.seed)
    natural: dict[str, list[NounCandidate]] = {}
    spare_one_token = []
    for category in settings.words.categories:
        buckets = []
        for token_count in TARGET_TOKEN_COUNTS:
            bucket = [
                candidate
                for candidate in listed_nouns
                if candidate.category == category.name
                and candidate.token_count == token_count
                and _usable(candidate, settings, needed, natural=True)
            ]
            random_numbers.shuffle(bucket)
            buckets.append(bucket)
        natural[category.name] = _take_in_turn(buckets, NATURAL_NOUNS_PER_CATEGORY, chosen)
        found = len(natural[category.name])
        if found < settings.word_selection.minimum_natural_nouns_per_category:
            description = f"カテゴリ {category.name} の自然な名詞"
            raise _shortage(description, found, needed, settings, natural=True)
        if found < NATURAL_NOUNS_PER_CATEGORY:
            print(
                f"カテゴリ {category.name} の自然な名詞は、条件を満たす候補が {found} 語で、"
                f"{NATURAL_NOUNS_PER_CATEGORY} 語に足りないので、{found} 語を選びます。"
            )
        # _take_in_turn が取らずに残した、1トークンの候補
        spare_one_token.append(buckets[0])
    probe_nouns = _take_in_turn(
        [[noun for noun in nouns if noun.token_count == 1] for nouns in natural.values()],
        PART_OF_SPEECH_NOUNS,
        ChosenWords(lambda word: frozenset()),
    )
    # 1トークンの自然な名詞が足りなければ、1トークンの候補が残るカテゴリから順に、自然な名詞を足す
    extra = _take_in_turn(spare_one_token, PART_OF_SPEECH_NOUNS - len(probe_nouns), chosen)
    for noun in extra:
        natural[noun.category].append(noun)
    probe_words = {noun.word for noun in [*probe_nouns, *extra]}
    if len(probe_words) < PART_OF_SPEECH_NOUNS:
        description = "1トークンの自然な名詞（品詞のプローブ用）"
        raise _shortage(description, len(probe_words), needed, settings, natural=True)
    rows = []
    for nouns in natural.values():
        for noun in nouns:
            row = _row(noun, "natural-noun", tokenizer)
            row.part_of_speech_probe = noun.word in probe_words
            rows.append(row)
    usable = [
        candidate
        for candidate in listed_adjectives
        if _usable(candidate, settings, needed, natural=True)
    ]
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
            natural=True,
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
    candidate: NounCandidate | AdjectiveCandidate,
    settings: Settings,
    needed_passages: int,
    natural: bool = False,
    minimum_share: float | None = None,
) -> bool:
    """候補が使えるか。natural なら、自然な語の割合の下限で判断する。

    minimum_share があれば、意図した語義で使われる割合の下限に、設定の代わりに使う。
    """
    selection = settings.word_selection
    share = _share(candidate)
    if minimum_share is None:
        adjective = isinstance(candidate, AdjectiveCandidate)
        minimum_share = _minimum_share(settings, adjective, natural)
    return (
        not candidate.rejected_reason
        and candidate.judged_usages >= selection.minimum_judged_usages
        and _meaning_share(candidate) >= minimum_share
        and candidate.estimated_pile_passages * share >= selection.passage_margin * needed_passages
    )


def _share(candidate: NounCandidate | AdjectiveCandidate) -> float:
    """Pile の見本で、用例として使える使われ方の割合。

    名詞は意図した語義で字義どおりに、形容詞は形容詞として使われていた割合。
    """
    if isinstance(candidate, NounCandidate):
        return candidate.intended_usage_share
    return candidate.adjective_usage_share


def _meaning_share(candidate: NounCandidate | AdjectiveCandidate) -> float:
    """「ほぼ単義（形容詞は、ほとんど形容詞としてだけ使われる）」かを判断する割合。

    名詞は、意図した語義で使われていた割合（比喩・慣用句・複合名詞の一部を含む。別の語義・名前・
    名詞でない使い方を除いた割合）。
    """
    if isinstance(candidate, NounCandidate):
        return candidate.same_meaning_usage_share
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


def _minimum_share(settings: Settings, adjective: bool, natural: bool) -> float:
    """意図した語義（形容詞は、形容詞として）で使われる割合の下限。"""
    selection = settings.word_selection
    if natural:
        return selection.minimum_natural_word_usage_share
    if adjective:
        return selection.minimum_adjective_usage_share
    return selection.minimum_usage_share


def _shortage(
    description: str,
    found: int,
    needed_passages: int,
    settings: Settings,
    listing: str = "words.nouns_per_listing",
    rerun_from: str = noun_candidates.COMMAND,
    natural: bool = False,
    share: float | None = None,
) -> CommandStopped:
    selection = settings.word_selection
    adjective = listing == "words.adjectives_per_listing"
    minimum_share = _minimum_share(settings, adjective, natural) if share is None else share
    return CommandStopped(
        f"{description}で、条件を満たす候補が足りません（{found} 語）。"
        f"条件: 意図した語義（形容詞は、形容詞として）で使われる割合が"
        f" {minimum_share} 以上（名詞は比喩・慣用句を含む）、"
        f"Pile で取れる用例の見積もり（字義どおりの使われ方の割合を掛けた数）が"
        f" {selection.passage_margin * needed_passages:,.0f} 件以上。"
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
