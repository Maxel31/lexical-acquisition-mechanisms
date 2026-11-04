"""用例の語の置き換え（両方の実験で共通の規則）。

- 置き換えるのは、用例の中の語の標準の形（先頭スペース付き）の出現（passages collect が記録した
  トークンの位置）だけ。所有格の 's など、後ろのトークンはそのまま残す。
- 置き換えた語の直前が冠詞 a / an（大文字で始まるものを含む）なら、置き換えた後の語の最初の文字が
  母音（a・e・i・o・u）かどうかに合わせて a / an にする。トークン番号で差し込む語（1トークンの
  未知語）では、冠詞を元のまま残す。先頭スペースのない a / an のトークンは、語の途中（前の
  トークンが英字で終わる。例: " pec|an| pie" の "an"）なら冠詞とみなさない。
- 置き換えはトークン列の中で行う。語の前後はトークンの区切り（先頭スペース付きの語は、それだけで
  1つの区切りになる）なので、文字列で置き換えてトークン化し直したのと同じトークン列になる。
- 置き換えた後、用例の長さに切り直す。置き換えた語を全て含み、始まりと次のトークンが語の途中に
  ならない窓のうち、置き換えた語が最も真ん中に来るものにする。
- 置き換えが0件、置き換えた後に元の語が残る、または切り直せないときは、ReplacementError にする。
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

VOWELS = "aeiou"


class ReplacementError(Exception):
    """用例の語を、規則どおりに置き換えられない。"""


@dataclass(frozen=True)
class Replacement:
    """置き換える語（元の語）と、置き換えた後の語。"""

    source_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    # 置き換えた後の語の表記（冠詞を合わせるのに使う。1トークンの未知語は管理用の名前）
    target_text: str
    # 冠詞を、置き換えた後の語に合わせるか（トークン番号で差し込む語では合わせない）
    adjust_articles: bool


@dataclass
class ReplacedPassage:
    token_ids: list[int]
    # 置き換えた語の、切り直した後の位置（最初のトークン）
    positions: list[int]
    # 置き換えごとの（元の形, 置き換えた後の形）。冠詞を替えたときは冠詞を含める
    records: list[tuple[str, str]]


class PassageReplacer:
    def __init__(self, tokenizer: WordTokenizer, tokens_per_passage: int) -> None:
        self._tokenizer = tokenizer
        self._length = tokens_per_passage
        self._boundaries = tokenizer.word_boundaries()
        # 冠詞のトークン（a・an の、先頭スペースの有無と、大文字で始まるかの組み合わせ）
        self._articles: dict[int, tuple[str, bool, bool]] = {}
        for article in ("a", "an"):
            for capitalized in (False, True):
                for leading_space in (False, True):
                    text = (" " if leading_space else "") + (
                        article.capitalize() if capitalized else article
                    )
                    (token_id,) = tokenizer.encode([text])
                    self._articles[token_id] = (article, capitalized, leading_space)

    def replace(
        self, token_ids: Sequence[int], positions: Sequence[int], replacement: Replacement
    ) -> ReplacedPassage:
        """出現の位置（最初のトークン）の語を置き換え、用例の長さに切り直す。"""
        if not positions:
            raise ReplacementError("置き換える出現がありません。")
        source = list(replacement.source_ids)
        pieces: list[list[int]] = []
        new_positions = []
        records = []
        start = 0
        length = 0
        for position in sorted(positions):
            if list(token_ids[position : position + len(source)]) != source:
                raise ReplacementError(f"位置 {position} に元の語がありません。")
            before = list(token_ids[start:position])
            original = self._tokenizer.decode(source)
            replaced = self._tokenizer.decode(list(replacement.target_ids))
            if self._has_article(replacement, before):
                article = self._article_for(before[-1], replacement.target_text)
                original = self._tokenizer.decode([before[-1]]) + original
                replaced = self._tokenizer.decode([article]) + replaced
                before[-1] = article
            pieces.append(before)
            length += len(before)
            new_positions.append(length)
            pieces.append(list(replacement.target_ids))
            length += len(replacement.target_ids)
            records.append((original.strip(), replaced.strip()))
            start = position + len(source)
        pieces.append(list(token_ids[start:]))
        replaced_ids = [token for piece in pieces for token in piece]
        window_start = self._window_start(replaced_ids, new_positions, len(replacement.target_ids))
        window = replaced_ids[window_start : window_start + self._length]
        if self._contains(window, source):
            raise ReplacementError("置き換えた後の用例に、元の語が残っています。")
        return ReplacedPassage(
            window, [position - window_start for position in new_positions], records
        )

    def _has_article(self, replacement: Replacement, before: list[int]) -> bool:
        """置き換える語の直前（before の最後）が、合わせる冠詞か。語の一部の a / an は除く。"""
        if not replacement.adjust_articles or not before or before[-1] not in self._articles:
            return False
        return not self._boundaries.inside_word(before, len(before) - 1)

    def _article_for(self, token_id: int, target_text: str) -> int:
        """置き換えた後の語の最初の文字に合わせた冠詞のトークン（大文字と先頭スペースは元のまま）。"""
        _, capitalized, leading_space = self._articles[token_id]
        wanted = "an" if target_text[:1].lower() in VOWELS else "a"
        for candidate, (article, other_capitalized, other_space) in self._articles.items():
            if (article, other_capitalized, other_space) == (wanted, capitalized, leading_space):
                return candidate
        raise ReplacementError(f"冠詞 {wanted} のトークンがありません。")

    def _window_start(self, token_ids: list[int], positions: list[int], word_length: int) -> int:
        """置き換えた語を全て含む、語の区切りで始まり語の区切りで終わる窓のうち、最も真ん中のもの。"""
        first = positions[0]
        last = positions[-1] + word_length
        lowest = max(0, last - self._length)
        highest = min(first, len(token_ids) - self._length)
        middle = (first + last - self._length) / 2
        starts = [
            start
            for start in range(lowest, highest + 1)
            if not self._boundaries.inside_word(token_ids, start)
            and not self._boundaries.inside_word(token_ids, start + self._length)
        ]
        if not starts:
            raise ReplacementError("置き換えた語を含むように、用例の長さに切り直せません。")
        return min(starts, key=lambda start: (abs(start - middle), start))

    def _contains(self, token_ids: list[int], word_ids: list[int]) -> bool:
        """トークン列に、語（直後が語の続きでないもの）があるか。"""
        array = np.array(token_ids)
        for position in np.flatnonzero(array == word_ids[0]):
            end = position + len(word_ids)
            if token_ids[position:end] == word_ids and not self._boundaries.inside_word(
                token_ids, end
            ):
                return True
        return False
