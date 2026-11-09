"""語を文脈の直後に置いたときの、全層の内部表現（残差ストリーム）。

文脈（中立文脈の文。空なら文脈なし）の後に語のトークン列（先頭スペース付きの形。標準とは違う区切りや、
新しいトークン番号でもよい）を置き、語の最後のトークンの位置の、埋め込みと各層の出力を取り出す。
"""

from collections.abc import Sequence

import numpy as np

from lexical_acquisition_mechanisms.language_model import LanguageModel


def word_representations(
    model: LanguageModel, words: Sequence[Sequence[int]], contexts: Sequence[str]
) -> np.ndarray:
    """形は（語, 文脈, 層の数 + 1, 隠れ次元）。0番目の層は埋め込み。"""
    inputs = []
    positions = []
    for token_ids in words:
        for context in contexts:
            parts = [context, list(token_ids)] if context else [list(token_ids)]
            ids, spans = model.tokenizer.encode_with_spans(parts)
            inputs.append(ids)
            positions.append(spans[-1][-1])
    states = model.residual_streams(inputs, positions)
    return states.reshape(len(words), len(contexts), *states.shape[1:])
