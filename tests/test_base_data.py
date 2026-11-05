"""基データのダウンロードのテスト。学習設定とデータのファイルは、手元で作った小さなものを返す。"""

import csv
from pathlib import Path

import httpx
import numpy as np
import pytest
import yaml

from lexical_acquisition_mechanisms import base_data
from lexical_acquisition_mechanisms.base_data import (
    build_base_data,
    count_word_occurrences,
    has_long_repetition,
    output_paths,
    training_order,
)
from lexical_acquisition_mechanisms.settings import BaseDataSettings

SEQUENCE_LENGTH = 8
CONFIGURATION_URL = "https://configs.test/OLMo2-7B-stage1.yaml"


def file_tokens() -> dict[str, np.ndarray]:
    """3つのファイル。b は端数（3トークン）があり、学習設定に2回書かれる。

    c は同じトークンの繰り返し（除かれる）。
    """
    return {
        "http://data.test/a.npy": np.arange(100, 100 + 3 * SEQUENCE_LENGTH, dtype="<u4"),
        "http://data.test/b.npy": np.arange(200, 200 + 2 * SEQUENCE_LENGTH + 3, dtype="<u4"),
        "http://data.test/c.npy": np.full(SEQUENCE_LENGTH, 7, dtype="<u4"),
    }


PATHS = [
    "http://data.test/a.npy",
    "http://data.test/b.npy",
    "http://data.test/b.npy",
    "http://data.test/c.npy",
]


def handle(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if url == CONFIGURATION_URL:
        configuration = {
            "seed": 7,
            "model": {"max_sequence_length": SEQUENCE_LENGTH},
            "data": {
                "paths": PATHS,
                "instance_filter": {
                    "repetition_max_period": 13,
                    "repetition_min_period": 1,
                    "repetition_max_count": 2,
                },
            },
        }
        return httpx.Response(200, text=yaml.safe_dump(configuration))
    content = file_tokens()[url].tobytes()
    if request.method == "HEAD":
        return httpx.Response(200, headers={"Content-Length": str(len(content))})
    first, last = (int(value) for value in request.headers["Range"][6:].split("-"))
    return httpx.Response(206, content=content[first : last + 1])


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> BaseDataSettings:
    monkeypatch.setattr(base_data, "EXPECTED_NUMBER_OF_SEQUENCES", range(8, 9))
    return BaseDataSettings(
        pretraining_configuration_url=CONFIGURATION_URL,
        number_of_base_sequences=5,
        number_of_validation_sequences=2,
    )


def expected_sequences() -> list[np.ndarray]:
    """学習設定の順の系列（a の3つ、b の2つ、もう一度 b の2つ、c の1つ）。"""
    tokens = file_tokens()
    sequences = []
    for path in PATHS:
        count = len(tokens[path]) // SEQUENCE_LENGTH
        sequences += [
            tokens[path][index * SEQUENCE_LENGTH : (index + 1) * SEQUENCE_LENGTH]
            for index in range(count)
        ]
    return sequences


def test_sequences_come_in_training_order_without_repetitive_ones(
    tmp_path: Path, settings: BaseDataSettings
) -> None:
    paths = output_paths(tmp_path)

    build_base_data(settings, paths, transport=httpx.MockTransport(handle))

    order = training_order(8, seed=7)
    sequences = expected_sequences()
    expected_base = [sequences[number] for number in order[:5] if number != 7]
    expected_validation = [sequences[number] for number in order[5:7] if number != 7]
    np.testing.assert_array_equal(np.load(paths["base"]), np.array(expected_base).reshape(-1, 8))
    np.testing.assert_array_equal(
        np.load(paths["validation"]), np.array(expected_validation).reshape(-1, 8)
    )
    with paths["sources"].open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [int(row["sequence_number"]) for row in rows] == list(order[:7])
    assert [row["removed"] for row in rows] == ["1" if number == 7 else "0" for number in order[:7]]
    assert {row["part"] for row in rows[:5]} == {"base"}


def test_an_unexpected_number_of_sequences_stops_the_download(
    tmp_path: Path, settings: BaseDataSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(base_data, "EXPECTED_NUMBER_OF_SEQUENCES", range(9, 10))

    with pytest.raises(RuntimeError, match="ファイルが変わっています"):
        build_base_data(settings, output_paths(tmp_path), transport=httpx.MockTransport(handle))


def test_training_order_matches_the_shuffle_of_the_pretraining_code() -> None:
    expected = np.arange(1000, dtype=np.uint32)
    np.random.Generator(np.random.PCG64(6198)).shuffle(expected)

    np.testing.assert_array_equal(training_order(1000, 6198), expected)


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (np.arange(4096), False),
        (np.array([5] * 40 + list(range(100))), True),
        (np.array([5] * 31 + list(range(100))), False),
        # 行の途中から始まる、周期3の繰り返し（33回）
        (np.array([900, 901] + [1, 2, 3] * 33 + list(range(10, 60))), True),
        (np.array([900, 901] + [1, 2, 3] * 20 + list(range(10, 60))), False),
    ],
)
def test_long_repetitions_are_found(tokens: np.ndarray, expected: bool) -> None:
    assert has_long_repetition(tokens, 13, 1, 32) is expected


def test_word_occurrences_are_counted_within_each_sequence() -> None:
    sequences = np.array([[5, 6, 7, 5, 6], [5, 9, 5, 6, 8]], dtype=np.uint32)
    # 7 と 9 は、直前の語の続きになるトークン
    continues_word = np.isin(np.arange(10), [7, 9])

    counts = count_word_occurrences(
        sequences, {"five-six": [5, 6], "five": [5], "six-five": [6, 5]}, continues_word
    )

    # [5, 6] は、1行目の先頭（後に 7 が続く）を除く2回。系列の最後で終わるものは数える
    assert counts["five-six"] == 2
    # [5] は、2行目の先頭（後に 9 が続く）を除く3回
    assert counts["five"] == 3
    # 1行目の最後と2行目の先頭にまたがる [6, 5] は数えない
    assert counts["six-five"] == 0
