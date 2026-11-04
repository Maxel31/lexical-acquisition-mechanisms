"""Pile の一部（先頭の断片）での、語の出現回数と使われ方の見本。

語の候補を選ぶときに、Pile での頻度と、使われ方（意図した語義・品詞で使われる割合）を見積もる。
数えるのは標準の形（先頭スペース付き・小文字。例: " pasta"）で、直後が英字でないものだけ。
見本は、出現ごとに、出現の位置（断片・行・文字の位置）と seed から決まる鍵を付け、鍵の小さいものから
選ぶ（無作為に選んだのと同じになり、並列の数や、一緒に調べる語によらず同じ見本になる）。
"""

import hashlib
import heapq
import re
from collections.abc import Callable, Collection, Iterator
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from lexical_acquisition_mechanisms import pile
from lexical_acquisition_mechanisms.pile import PileDocument

WORD_PATTERN = re.compile(r"[A-Za-z]+")
# 見本として、出現の前後に残す文字数
CONTEXT_CHARACTERS = 300
NUMBER_OF_PROCESSES = 16


@dataclass
class WordUsages:
    # 調べた断片での、標準の形の出現回数
    count: int = 0
    # 見本（鍵, 出現の前後の文字列）。鍵の小さい順
    examples: list[tuple[float, str]] = field(default_factory=list)

    @property
    def example_texts(self) -> list[str]:
        return [text for _, text in self.examples]


def sample_word_usages(
    words: Collection[str],
    number_of_shards: int,
    examples_per_word: int,
    seed: int,
    number_of_processes: int = NUMBER_OF_PROCESSES,
    download_directory: Path = pile.DOWNLOAD_DIRECTORY,
    download: Callable[[str, Path], Path] = pile.download_shard,
) -> dict[str, WordUsages]:
    """Pile の先頭の number_of_shards 個の断片で、各語の出現回数を数え、見本を選ぶ。"""
    scan = partial(
        scan_documents, words=frozenset(words), examples_per_word=examples_per_word, seed=seed
    )
    totals = {word: WordUsages() for word in words}
    shard_results = pile.scan_shards(
        range(number_of_shards), scan, number_of_processes, download_directory, download
    )
    for shard_result in shard_results:
        for word, usages in shard_result.items():
            total = totals[word]
            total.count += usages.count
            total.examples = heapq.nsmallest(
                examples_per_word, total.examples + usages.examples, key=lambda pair: pair[0]
            )
    return totals


def scan_documents(
    documents: Iterator[PileDocument],
    words: frozenset[str],
    examples_per_word: int,
    seed: int,
) -> dict[str, WordUsages]:
    """1つの断片の文書で、語の標準の形を数え、鍵の小さい見本を残す。"""
    counts: dict[str, int] = {}
    # 語ごとに、鍵の大きい見本が先頭に来るヒープ（鍵の符号を変えて入れる）
    heaps: dict[str, list[tuple[float, str]]] = {}
    for document in documents:
        text = document.text
        for match in WORD_PATTERN.finditer(text):
            word = match.group()
            start = match.start()
            if word not in words or start == 0 or text[start - 1] != " ":
                continue
            counts[word] = counts.get(word, 0) + 1
            key = _key(seed, document.shard, document.row, start)
            heap = heaps.setdefault(word, [])
            if len(heap) < examples_per_word:
                heapq.heappush(heap, (-key, _context(text, start, match.end())))
            elif key < -heap[0][0]:
                heapq.heapreplace(heap, (-key, _context(text, start, match.end())))
    return {
        word: WordUsages(
            count, sorted((-negative_key, example) for negative_key, example in heaps[word])
        )
        for word, count in counts.items()
    }


def _key(seed: int, shard: int, row: int, start: int) -> float:
    """出現の位置と seed から決まる、0 以上 1 未満の数。"""
    digest = hashlib.blake2b(f"{seed}:{shard}:{row}:{start}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def _context(text: str, start: int, end: int) -> str:
    """出現の前後 CONTEXT_CHARACTERS 文字。出現は [[ ]] で囲む。

    改行などの空白は、1つの空白にする。
    """
    before = text[max(0, start - CONTEXT_CHARACTERS) : start]
    after = text[end : end + CONTEXT_CHARACTERS]
    return " ".join(f"{before}[[{text[start:end]}]]{after}".split())
