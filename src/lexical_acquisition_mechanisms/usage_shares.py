"""Pile の見本で、語が意図した語義・品詞で使われている割合を、LLM の判定から出す。"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lexical_acquisition_mechanisms import pile
from lexical_acquisition_mechanisms.llm_client import LlmClient, LlmRequest
from lexical_acquisition_mechanisms.pile_word_usages import WordUsages


@dataclass(frozen=True)
class UsageShare:
    # Pile の先頭の断片での、標準の形の出現回数と、Pile 全体での見積もり
    pile_sample_count: int
    estimated_pile_count: int
    # Pile 全体で、用例として切り出せる文章の数の見積もり
    estimated_pile_passages: int
    # LLM が判定できた見本の数と、そのうち条件（判定の答えについての、名前ごとの条件）を満たした割合
    judged_usages: int
    shares: dict[str, float]


def usage_shares(
    words: Sequence[str],
    usages: Mapping[str, WordUsages],
    pile_sample_shards: int,
    make_request: Callable[[str, str], LlmRequest],
    conditions: Mapping[str, Callable[[dict[str, Any]], bool]],
    client: LlmClient,
) -> dict[str, UsageShare]:
    """語ごとに、見本の判定（make_request(語, 見本) への答え）が、条件ごとに真の割合を出す。

    答えが得られなかった見本は、割合の分母に入れない。
    """
    requests = []
    owners = []
    for word in words:
        for passage in usages[word].example_texts:
            requests.append(make_request(word, passage))
            owners.append(word)
    judged = dict.fromkeys(words, 0)
    satisfied = {word: dict.fromkeys(conditions, 0) for word in words}
    for word, answer in zip(owners, client.ask_all(requests), strict=True):
        if answer is not None:
            judged[word] += 1
            for name, condition in conditions.items():
                satisfied[word][name] += bool(condition(answer))
    scale = pile.NUMBER_OF_SHARDS / pile_sample_shards
    return {
        word: UsageShare(
            pile_sample_count=usages[word].count,
            estimated_pile_count=round(usages[word].count * scale),
            estimated_pile_passages=round(usages[word].passage_count * scale),
            judged_usages=judged[word],
            shares={
                name: count / judged[word] if judged[word] else 0.0
                for name, count in satisfied[word].items()
            },
        )
        for word in words
    }
