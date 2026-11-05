"""語のトークン化。OLMo 2 の3モデルは、同じトークナイザ（dolma2-tokenizer）を使う。

- 語の形 → トークン列（先頭スペースの有無を区別する）
- 標準とは違う区切り（例: pasta を pas|ta に分ける）や、トークン番号を差し込んだ入力のトークン列
- 埋め込みにはあるが、トークナイザが使っていないトークン番号（1トークンの未知語に割り当てる）

文字列は文字どおりにトークン化する（文中の "<|endoftext|>" などの文字列を、特殊トークンにしない）。
特殊トークンは、トークン番号で差し込む。ただし、OLMo 2 が事前学習データで個人情報の置き換えに使った
"|||EMAIL_ADDRESS|||" などの追加トークンは、事前学習のときと同じく1トークンになる。
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase

from lexical_acquisition_mechanisms.settings import ModelSource


@dataclass(frozen=True)
class WordBoundaries:
    """トークン番号ごとの、英字で始まるか・英字で終わるか。

    2つのトークンの間は、前のトークンが英字で終わり、後のトークンが英字で始まるときだけ、語の途中に
    なる（例: " cat" と "fish" の間。改行の後の "The" や、文書の先頭の "We" の前は、語の途中では
    ない）。使われていないトークン番号は、どちらも偽にする。
    """

    starts_with_letter: np.ndarray
    ends_with_letter: np.ndarray

    def inside_word(self, token_ids: Sequence[int], index: int) -> bool:
        """token_ids[index - 1] と token_ids[index] の間が、語の途中か（列の両端は途中でない）。"""
        return 0 < index < len(token_ids) and bool(
            self.ends_with_letter[token_ids[index - 1]]
            and self.starts_with_letter[token_ids[index]]
        )


class WordTokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, number_of_embedding_rows: int) -> None:
        if number_of_embedding_rows < len(tokenizer):
            raise ValueError(
                f"埋め込みの行数 {number_of_embedding_rows} が、"
                f"トークナイザの語彙数 {len(tokenizer)} より少ない。"
            )
        self._tokenizer = tokenizer
        self.number_of_embedding_rows = number_of_embedding_rows

    @classmethod
    def from_model(cls, source: ModelSource) -> "WordTokenizer":
        """モデルの取得元から、トークナイザと埋め込みの行数（設定の vocab_size）を読む。"""
        tokenizer = AutoTokenizer.from_pretrained(source.repository, revision=source.revision)
        configuration = AutoConfig.from_pretrained(source.repository, revision=source.revision)
        return cls(tokenizer, configuration.vocab_size)

    @property
    def end_of_text_id(self) -> int:
        """文書の区切りを表す特殊トークン <|endoftext|> の番号。"""
        return self._tokenizer.eos_token_id

    def word_token_ids(self, word: str, leading_space: bool = True) -> list[int]:
        """語のトークン列。leading_space なら、文中の形（例: " pasta"）にする。"""
        if not word or word != word.strip():
            raise ValueError(f"語 {word!r} は、空でなく、前後に空白のない文字列にしてください。")
        return self.encode([" " + word if leading_space else word])

    def split_word_token_ids(self, pieces: Sequence[str]) -> list[int]:
        """標準とは違う区切りでの語のトークン列。各部分（例: " pas", "ta"）を1トークンにする。"""
        token_ids = []
        for piece in pieces:
            piece_ids = self.encode([piece])
            if len(piece_ids) != 1:
                raise ValueError(
                    f"区切った {piece!r} が1トークンになりません（{len(piece_ids)} トークン）。"
                )
            token_ids.append(piece_ids[0])
        return token_ids

    def encode(self, parts: Sequence[str | Sequence[int]]) -> list[int]:
        """文字列はトークン化し、トークン番号の列はそのまま、順に並べる。

        文字列ごとにトークン化するので、全体をまとめてトークン化した結果と一致させるには、語の前の
        空白を後ろの部分に含めて区切る（例: "Do you know about the" と " pasta"）。
        """
        token_ids, _ = self.encode_with_spans(parts)
        return token_ids

    def encode_with_spans(
        self, parts: Sequence[str | Sequence[int]]
    ) -> tuple[list[int], list[range]]:
        """encode と同じトークン列と、各部分が占める位置の範囲。

        例えば、語の最後のトークンの位置は、語の部分の範囲の最後（spans[i][-1]）になる。
        """
        token_ids: list[int] = []
        spans: list[range] = []
        for part in parts:
            if isinstance(part, str):
                encoding = self._tokenizer(
                    part, add_special_tokens=False, split_special_tokens=True
                )
                part_ids = list(encoding["input_ids"])
            else:
                part_ids = list(part)
                self.check_token_ids(part_ids)
            spans.append(range(len(token_ids), len(token_ids) + len(part_ids)))
            token_ids.extend(part_ids)
        return token_ids, spans

    def decode(self, token_ids: Sequence[int]) -> str:
        """トークン列を文字列に戻す。

        トークナイザが使っていない番号（1トークンの未知語）は、" <新しいトークン 100278>" と書く
        （先頭スペース付きの語の代わりなので、前に空白を付ける）。
        """
        self.check_token_ids(token_ids)
        pieces: list[str] = []
        known_ids: list[int] = []
        for token_id in token_ids:
            if token_id < len(self._tokenizer):
                known_ids.append(token_id)
                continue
            if known_ids:
                pieces.append(self._tokenizer.decode(known_ids))
                known_ids = []
            pieces.append(f" <新しいトークン {token_id}>")
        if known_ids:
            pieces.append(self._tokenizer.decode(known_ids))
        return "".join(pieces)

    def word_boundaries(self) -> WordBoundaries:
        """トークン番号ごとの、英字で始まるか・英字で終わるか。"""
        starts = np.zeros(self.number_of_embedding_rows, dtype=bool)
        ends = np.zeros(self.number_of_embedding_rows, dtype=bool)
        for token_id in range(len(self._tokenizer)):
            text = self._tokenizer.decode([token_id])
            starts[token_id] = bool(text) and text[0].isascii() and text[0].isalpha()
            ends[token_id] = bool(text) and text[-1].isascii() and text[-1].isalpha()
        return WordBoundaries(starts, ends)

    def unused_token_ids(self, count: int) -> list[int]:
        """埋め込みにはあるが、トークナイザが使っていないトークン番号を、小さい順に返す。"""
        unused = range(len(self._tokenizer), self.number_of_embedding_rows)
        if not 0 <= count <= len(unused):
            raise ValueError(
                f"使われていないトークン番号は {len(unused)} 個です（{count} 個は返せません）。"
            )
        return list(unused[:count])

    def check_token_ids(self, token_ids: Sequence[int]) -> None:
        for token_id in token_ids:
            if not 0 <= token_id < self.number_of_embedding_rows:
                raise ValueError(
                    f"トークン番号 {token_id} は、埋め込みの行"
                    f"（0〜{self.number_of_embedding_rows - 1}）にありません。"
                )
