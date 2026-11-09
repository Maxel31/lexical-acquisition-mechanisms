"""出現の前後の文字列から、語を含む256トークンの用例の候補を切り出して確かめる。

passages collect の後半。

1. 前後の文字列の端の、途中で切れた語を除き、出現の前・出現（先頭スペース付きの形）・後に分けて
   トークン化する（分けても、まとめてトークン化したのと同じトークン列になる）。
2. 出現を含む位置に、用例の長さの窓を、無作為に置く。窓の始まりと、窓の次のトークンは、語の途中に
   しない（英字で始まるトークンにしない）。窓に入らなければ（文書が短い）、使わない。
3. 次の窓は使わない。
   - 語の標準の形でない形（例: "Pasta"、" pastas"）を含む（置き換えた後のトークン列をそろえるため）
   - ほかの管理対象の語を、どの形でも含む（1つの文章で操作するのは1語だけにするため）
4. 窓の中の、標準の形の出現の位置（トークンの位置）を記録する。
"""

import multiprocessing
import re
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from lexical_acquisition_mechanisms.pile_word_usages import WORD_PATTERN, occurrence_key
from lexical_acquisition_mechanisms.settings import ModelSource
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

# 窓の位置を決める鍵の seed に足す数（出現を選んだ鍵と別の数にするため）
WINDOW_SEED_OFFSET = 1
CHUNK_SIZE = 2000


@dataclass(frozen=True)
class WindowSettings:
    tokens_per_passage: int
    context_characters: int
    seed: int
    # 管理対象の語（名詞・形容詞・ターゲット語）
    managed_words: frozenset[str]


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
        continues_word=tokenizer.word_continuation_flags(),
        settings=settings,
    )


def _cut_chunk(occurrences: Iterable[dict[str, Any]]) -> list[PassageWindow | str]:
    return [cut_window(occurrence, **_worker) for occurrence in occurrences]


def cut_window(
    occurrence: dict[str, Any],
    tokenizer: WordTokenizer,
    continues_word: np.ndarray,
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
        if not continues_word[token_ids[start]]
        and (start + length == len(token_ids) or not continues_word[token_ids[start + length]])
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
    forms = re.compile(rf"(?<![A-Za-z]){re.escape(word)}(?:e?s)?(?![A-Za-z])", re.IGNORECASE)
    for match in forms.finditer(text):
        if match.group() != word or match.start() == 0 or text[match.start() - 1] != " ":
            return "語の標準の形でない形を含む"
        standard_forms += 1
    if _words_in(text) & (settings.managed_words - {word}):
        return "ほかの管理対象の語を含む"
    word_ids = token_ids[spans[1].start : spans[1].stop]
    positions = [
        position - start
        for position in range(start, start + length - len(word_ids) + 1)
        if token_ids[position : position + len(word_ids)] == word_ids
        and (
            position + len(word_ids) == len(token_ids)
            or not continues_word[token_ids[position + len(word_ids)]]
        )
    ]
    if len(positions) != standard_forms:
        return "語の出現とトークンの区切りが合わない"
    return PassageWindow(window, positions)


def _words_in(text: str) -> set[str]:
    """文章の中の英語の語（小文字にし、複数形の s・es を除いた形も入れる）。"""
    words = {match.lower() for match in WORD_PATTERN.findall(text)}
    return (
        words
        | {word[:-1] for word in words if word.endswith("s")}
        | {word[:-2] for word in words if word.endswith("es")}
    )


def _first_space(text: str) -> int:
    return next((index for index, letter in enumerate(text) if letter.isspace()), 0)


def _last_space(text: str) -> int:
    indices = range(len(text) - 1, -1, -1)
    return next((index for index in indices if text[index].isspace()), len(text))
