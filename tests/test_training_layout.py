"""用例を差し込む位置と、差し込んだトークン列のテスト。"""

from pathlib import Path

import numpy as np
import pytest

from lexical_acquisition_mechanisms.training_layout import (
    insert_passages,
    load_base_tokens,
    plan_insertions,
)

END = 0
SEQUENCE_LENGTH = 16
PASSAGE_LENGTH = 4


def base_tokens() -> np.ndarray:
    """6つの文書（長さ 5〜9 の、1 から数える数の列と、区切り）。"""
    documents = [list(range(1, length)) + [END] for length in (6, 9, 5, 8, 7, 9)]
    return np.array([token for document in documents for token in document], dtype=np.uint32)


def test_passages_sit_at_boundaries_inside_sequences() -> None:
    base = base_tokens()

    layout = plan_insertions(base, END, 3, PASSAGE_LENGTH, SEQUENCE_LENGTH)

    total = len(base) + 3 * PASSAGE_LENGTH
    assert layout.used_length == total // SEQUENCE_LENGTH * SEQUENCE_LENGTH
    for index, (base_position, start) in enumerate(
        zip(layout.base_positions, layout.output_starts, strict=True)
    ):
        # 文書の区切りの直後に置き、系列の区切りをまたがない
        assert base[base_position - 1] == END
        assert start == base_position + index * PASSAGE_LENGTH
        assert start % SEQUENCE_LENGTH + PASSAGE_LENGTH <= SEQUENCE_LENGTH
        assert start + PASSAGE_LENGTH <= layout.used_length
    assert list(layout.output_starts) == sorted(layout.output_starts)

    passages = [[100 + index] * (PASSAGE_LENGTH - 1) for index in range(3)]
    tokens = insert_passages(base, layout, passages, END)

    assert len(tokens) == layout.used_length
    for index, start in enumerate(layout.output_starts):
        assert list(tokens[start : start + PASSAGE_LENGTH]) == [100 + index] * 3 + [END]
    # 用例を除くと、基データの先頭（学習に使う長さの分）と同じ
    inserted = np.zeros(len(tokens), dtype=bool)
    for start in layout.output_starts:
        inserted[start : start + PASSAGE_LENGTH] = True
    assert list(tokens[~inserted]) == list(base[: (~inserted).sum()])


def test_passages_of_the_wrong_length_are_rejected() -> None:
    base = base_tokens()
    layout = plan_insertions(base, END, 1, PASSAGE_LENGTH, SEQUENCE_LENGTH)

    with pytest.raises(ValueError, match="3 トークンではありません"):
        insert_passages(base, layout, [[1, 2]], END)


def test_only_the_first_base_sequences_are_used_when_a_number_is_set(tmp_path: Path) -> None:
    path = tmp_path / "base_sequences.npy"
    np.save(path, np.arange(12, dtype=np.uint32).reshape(3, 4))

    assert load_base_tokens(path, 2).tolist() == list(range(8))
    assert load_base_tokens(path, None).tolist() == list(range(12))
    with pytest.raises(ValueError, match="超えています"):
        load_base_tokens(path, 4)
