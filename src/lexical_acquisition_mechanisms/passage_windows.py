"""出現の前後の文字列から、語を含む256トークンの用例の候補を切り出して確かめる。

passages collect の後半。

1. 前後の文字列の端の、途中で切れた語を除き、出現の前・出現（先頭スペース付きの形）・後に分けて
   トークン化する（分けても、まとめてトークン化したのと同じトークン列になる）。
2. 出現を含む位置に、用例の長さの窓を、無作為に置く。窓の始まりと終わりは、語の途中にしない
   （前のトークンが英字で終わり、後のトークンが英字で始まる所で切らない）。窓に入らなければ
   （文書が短い）、使わない。
3. 次の窓は使わない。
   - 語の標準の形でない形（例: "Pasta"、" pastas"、" cherries"、" quieter"。inflected_forms）を含む
     （置き換えた後のトークン列をそろえ、元の語の別の形が残らないようにするため）
   - ほかの管理対象の語を、どの形でも含む（1つの文章で操作するのは1語だけにするため）
4. 窓の中の、標準の形の出現の位置（トークンの位置）を記録する。
"""

import multiprocessing
import re
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

from lexical_acquisition_mechanisms.pile_word_usages import (
    NOT_AFTER_LETTER,
    NOT_BEFORE_LETTER,
    WORD_PATTERN,
    occurrence_key,
)
from lexical_acquisition_mechanisms.settings import ModelSource
from lexical_acquisition_mechanisms.word_tokenizer import WordBoundaries, WordTokenizer

# 窓の位置を決める鍵の seed に足す数（出現を選んだ鍵と別の数にするため）
WINDOW_SEED_OFFSET = 1
CHUNK_SIZE = 2000


@dataclass(frozen=True)
class WindowSettings:
    tokens_per_passage: int
    context_characters: int
    seed: int
    # 語ごとの、標準の形以外の形（小文字）
    other_forms: Mapping[str, frozenset[str]]
    # 管理対象の語（名詞・形容詞・ターゲット語）の全ての形（小文字）から、その語への対応
    managed_forms: Mapping[str, str]


@dataclass
class PassageWindow:
    token_ids: list[int]
    # 窓の中の、語の標準の形の出現（最初のトークン）の位置
    occurrence_positions: list[int]


def cut_windows(
    occurrences: Sequence[dict[str, Any]],
    source: ModelSource,
    settings: WindowSettings,
    number_of_processes: int,
) -> list[PassageWindow | str]:
    """出現ごとに、窓か、使わない理由を返す。"""
    chunks = [
        occurrences[start : start + CHUNK_SIZE] for start in range(0, len(occurrences), CHUNK_SIZE)
    ]
    if number_of_processes == 1:
        _initialize_worker(source, settings)
        results = map(_cut_chunk, chunks)
    else:
        context = multiprocessing.get_context("spawn")
        executor = ProcessPoolExecutor(
            max_workers=number_of_processes,
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(source, settings),
        )
        with executor:
            results = list(executor.map(_cut_chunk, chunks))
    return [window for chunk in results for window in chunk]


# 別のプロセスで使う、トークナイザなど（_initialize_worker が入れる）
_worker: dict[str, Any] = {}


def _initialize_worker(source: ModelSource, settings: WindowSettings) -> None:
    tokenizer = WordTokenizer.from_model(source)
    _worker.update(
        tokenizer=tokenizer,
        boundaries=tokenizer.word_boundaries(),
        settings=settings,
    )
    _form_patterns.clear()


# 語ごとの、全ての形を大文字と小文字を区別せずに探す正規表現（_form_pattern が作る）
_form_patterns: dict[str, re.Pattern[str]] = {}


def _cut_chunk(occurrences: Iterable[dict[str, Any]]) -> list[PassageWindow | str]:
    return [cut_window(occurrence, **_worker) for occurrence in occurrences]


def cut_window(
    occurrence: dict[str, Any],
    tokenizer: WordTokenizer,
    boundaries: WordBoundaries,
    settings: WindowSettings,
) -> PassageWindow | str:
    """1つの出現から窓を切り出す。使わないときは、その理由を返す。"""
    word = occurrence["word"]
    context = occurrence["context"]
    offset = occurrence["start"] - occurrence["context_start"]
    left = 0 if occurrence["context_start"] == 0 else _first_space(context)
    reaches_end = len(context) < offset + len(word) + settings.context_characters
    right = len(context) if reaches_end else _last_space(context)
    parts = [context[left : offset - 1], " " + word, context[offset + len(word) : right]]
    token_ids, spans = tokenizer.encode_with_spans(parts)
    length = settings.tokens_per_passage
    lowest = max(0, spans[1].stop - length)
    highest = min(spans[1].start, len(token_ids) - length)
    starts = [
        start
        for start in range(lowest, highest + 1)
        if not boundaries.inside_word(token_ids, start)
        and not boundaries.inside_word(token_ids, start + length)
    ]
    if not starts:
        return "文書が短いか、語の区切りで切れない"
    choice = occurrence_key(
        settings.seed + WINDOW_SEED_OFFSET,
        occurrence["shard"],
        occurrence["row"],
        occurrence["start"],
    )
    start = starts[int(choice * len(starts))]
    window = token_ids[start : start + length]
    text = tokenizer.decode(window)
    standard_forms = 0
    for match in _form_pattern(word, settings).finditer(text):
        if match.group() != word or match.start() == 0 or text[match.start() - 1] != " ":
            return "語の標準の形でない形を含む"
        standard_forms += 1
    words = {match.lower() for match in WORD_PATTERN.findall(text)}
    if {settings.managed_forms[form] for form in words if form in settings.managed_forms} - {word}:
        return "ほかの管理対象の語を含む"
    word_ids = token_ids[spans[1].start : spans[1].stop]
    positions = [
        position - start
        for position in range(start, start + length - len(word_ids) + 1)
        if token_ids[position : position + len(word_ids)] == word_ids
        and not boundaries.inside_word(token_ids, position + len(word_ids))
    ]
    if len(positions) != standard_forms:
        return "語の出現とトークンの区切りが合わない"
    return PassageWindow(window, positions)


def _form_pattern(word: str, settings: WindowSettings) -> re.Pattern[str]:
    if word not in _form_patterns:
        forms = "|".join(map(re.escape, [word, *sorted(settings.other_forms.get(word, ()))]))
        _form_patterns[word] = re.compile(
            rf"{NOT_AFTER_LETTER}(?:{forms}){NOT_BEFORE_LETTER}", re.IGNORECASE
        )
    return _form_patterns[word]


def _first_space(text: str) -> int:
    return next((index for index, letter in enumerate(text) if letter.isspace()), 0)


def _last_space(text: str) -> int:
    indices = range(len(text) - 1, -1, -1)
    return next((index for index in indices if text[index].isspace()), len(text))
