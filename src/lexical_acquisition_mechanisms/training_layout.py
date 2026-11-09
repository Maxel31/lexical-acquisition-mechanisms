"""用例を基データに差し込む位置と、差し込んだ学習データのトークン列（両方の実験で共通）。

- 基データ（4,096トークンの系列を学習順に並べたもの）を1本のトークン列とみなし、文書の区切り
  （<|endoftext|> の直後）に、用例（用例のトークン + <|endoftext|>）を差し込む。
- 差し込んだ後のトークン列を4,096トークンずつの系列に分けたとき、用例が系列の区切りをまたがない
  位置にだけ差し込む。最後の、4,096トークンに満たない系列は、学習に使わない（そこには差し込まない）。
- 用例は、差し込んだ後の列の全体に均等に散らす。i 番目の用例の目標の位置を
  (i + 0.5) × 散らす長さ ÷ 用例の数 とし、その位置以降で、上の条件を満たす最初の文書の区切りに置く。
  散らす長さは、使う長さと、最後の文書の区切りのうち短い方から、系列4つ分を除いたもの。
- 位置は、用例の数と長さと基データだけで決まる（同じ実験の全 run で共通。違うのは中身だけ）。
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

TAIL_SEQUENCES = 4


@dataclass(frozen=True)
class InsertionLayout:
    # 用例 i を差し込む、基データの位置（その位置のトークンの前に差し込む。同じ位置に続けて置く
    # こともある）と、差し込んだ後の列での、用例の始まりの位置
    base_positions: np.ndarray
    output_starts: np.ndarray
    # 用例1件の長さ（後に付ける <|endoftext|> を含む）
    passage_length: int
    # 差し込んだ後の列のうち、学習に使う長さ（系列の長さの倍数）
    used_length: int


def plan_insertions(
    base_tokens: np.ndarray,
    end_of_text_id: int,
    number_of_passages: int,
    passage_length: int,
    sequence_length: int,
) -> InsertionLayout:
    """用例（区切りを含めて passage_length トークン）を number_of_passages 件差し込む位置。"""
    total_length = len(base_tokens) + number_of_passages * passage_length
    used_length = total_length // sequence_length * sequence_length
    # 文書の区切りの直後（差し込める位置）
    boundaries = np.flatnonzero(base_tokens == end_of_text_id) + 1
    if len(boundaries) == 0:
        raise ValueError("基データに文書の区切りがありません。")
    # 目標の位置は、最後の文書の区切りより前に散らす（最後の文書が長くても置けるように、
    # 系列 TAIL_SEQUENCES 個分の余裕を残す）
    last_start = int(boundaries[-1]) + (number_of_passages - 1) * passage_length
    span = min(used_length, last_start) - TAIL_SEQUENCES * sequence_length
    base_positions = np.zeros(number_of_passages, dtype=np.int64)
    output_starts = np.zeros(number_of_passages, dtype=np.int64)
    candidate = 0
    for index in range(number_of_passages):
        goal = (index + 0.5) * span / number_of_passages
        # 基データの位置 b に置くと、差し込んだ後の始まりは b + index × passage_length
        lowest = max(0, int(np.ceil(goal)) - index * passage_length)
        candidate = max(candidate, int(np.searchsorted(boundaries, lowest)))
        while True:
            if candidate >= len(boundaries):
                raise ValueError(f"用例 {index + 1} 件目を差し込める文書の区切りがありません。")
            start = int(boundaries[candidate]) + index * passage_length
            if start + passage_length > used_length:
                raise ValueError(f"用例 {index + 1} 件目が、学習に使う長さに収まりません。")
            if start % sequence_length + passage_length <= sequence_length:
                break
            candidate += 1
        base_positions[index] = boundaries[candidate]
        output_starts[index] = start
    return InsertionLayout(base_positions, output_starts, passage_length, used_length)


def insert_passages(
    base_tokens: np.ndarray,
    layout: InsertionLayout,
    passages: Sequence[Sequence[int]],
    end_of_text_id: int,
) -> np.ndarray:
    """基データに、用例（それぞれの後に <|endoftext|> を付ける）を差し込んだトークン列。"""
    pieces = []
    previous = 0
    for base_position, passage in zip(layout.base_positions, passages, strict=True):
        if len(passage) + 1 != layout.passage_length:
            raise ValueError(
                f"用例の長さ {len(passage)} が、"
                f"{layout.passage_length - 1} トークンではありません。"
            )
        pieces.append(base_tokens[previous:base_position])
        pieces.append(np.asarray([*passage, end_of_text_id], dtype=base_tokens.dtype))
        previous = base_position
    pieces.append(base_tokens[previous:])
    return np.concatenate(pieces)[: layout.used_length]
