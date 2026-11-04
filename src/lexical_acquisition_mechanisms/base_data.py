"""基データ: OLMo 2 の事前学習（ステージ1）の学習データを、学習順に並べた先頭の部分。

学習順は、OLMo（github.com/allenai/OLMo）の学習の実装どおりに再現する。
1. 学習設定の data.paths（1,122個。重なっているものも別に数える）の順に、各ファイルを
   4,096 トークンずつの系列に分ける（端数は捨てる）。ファイルの大きさは HTTP の HEAD で調べる。
2. 系列の番号 0〜N-1 を np.random.Generator(PCG64(seed)).shuffle で並べ替えた順が、学習順になる。
3. 学習順の先頭から、基データと、その直後の検証データの系列を、HTTP の範囲指定で取り出す。
繰り返しの多い系列（OLMo の instance_filter に当たるもの）は、OLMo 2 が損失を 0 にして学習に
使っていないので除く。
"""

import argparse
import asyncio
import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import yaml

from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.settings import BaseDataSettings, Settings

COMMAND = "base-data download"
OUTPUT_DIRECTORY = Path("data/base")
TOKEN_BYTES = 4
# 公開されている学習の状態（7B の最後の step と、1B の2周目に入った位置）から求めた、
# 系列の数 N の範囲。ファイルが作り直されて大きさが変わっていないかを確かめる
EXPECTED_NUMBER_OF_SEQUENCES = range(950_933_504, 950_934_016)
MAXIMUM_CONCURRENT_REQUESTS = 32
NUMBER_OF_ATTEMPTS = 5


@dataclass(frozen=True)
class PretrainingData:
    """学習設定のうち、学習順を決めるもの。"""

    paths: list[str]
    seed: int
    sequence_length: int
    repetition_maximum_period: int
    repetition_minimum_period: int
    repetition_maximum_count: int


def output_paths(directory: Path = OUTPUT_DIRECTORY) -> dict[str, Path]:
    return {
        "base": directory / "base_sequences.npy",
        "validation": directory / "validation_sequences.npy",
        "sources": directory / "sequence_sources.csv",
    }


def download(settings: Settings, arguments: argparse.Namespace) -> None:
    paths = output_paths()
    step = Step(
        command=COMMAND,
        inputs={},
        outputs=list(paths.values()),
        settings=settings.base_data.model_dump(),
    )
    run_unless_unchanged(
        step, lambda: build_base_data(settings.base_data, paths), arguments.recompute
    )


def build_base_data(
    settings: BaseDataSettings,
    paths: dict[str, Path],
    transport: httpx.AsyncBaseTransport | None = None,
) -> None:
    asyncio.run(_build_base_data(settings, paths, transport))


async def _build_base_data(
    settings: BaseDataSettings,
    paths: dict[str, Path],
    transport: httpx.AsyncBaseTransport | None,
) -> None:
    limits = httpx.Limits(max_connections=MAXIMUM_CONCURRENT_REQUESTS)
    async with httpx.AsyncClient(
        transport=transport, timeout=120, limits=limits, follow_redirects=True
    ) as client:
        response = await client.get(settings.pretraining_configuration_url)
        response.raise_for_status()
        data = read_pretraining_data(response.text)
        print(f"学習データのファイル {len(set(data.paths))} 個の大きさを調べます。")
        sizes = await _file_sizes(client, sorted(set(data.paths)))
        starts = sequence_starts(data, sizes)
        number_of_sequences = int(starts[-1])
        if number_of_sequences not in EXPECTED_NUMBER_OF_SEQUENCES:
            expected = EXPECTED_NUMBER_OF_SEQUENCES
            raise RuntimeError(
                f"系列の数 {number_of_sequences} が、学習のときの数（{expected.start}〜"
                f"{expected.stop - 1}）と合いません。ファイルが変わっています。"
            )
        print(f"系列 {number_of_sequences} 個の学習順を再現します。")
        order = training_order(number_of_sequences, data.seed)
        end_of_base = settings.number_of_base_sequences
        end_of_validation = end_of_base + settings.number_of_validation_sequences
        sequence_numbers = order[:end_of_validation]
        print(f"系列 {len(sequence_numbers)} 個を取り出します。")
        sequences = await _fetch_sequences(client, data, starts, sequence_numbers)

    removed = [
        has_long_repetition(
            sequence,
            data.repetition_maximum_period,
            data.repetition_minimum_period,
            data.repetition_maximum_count,
        )
        for sequence in sequences
    ]
    kept = ~np.array(removed)
    paths["base"].parent.mkdir(parents=True, exist_ok=True)
    np.save(paths["base"], sequences[:end_of_base][kept[:end_of_base]])
    np.save(paths["validation"], sequences[end_of_base:][kept[end_of_base:]])
    _write_sources(paths["sources"], data, starts, sequence_numbers, removed, end_of_base)
    print(f"繰り返しの多い系列 {sum(removed)} 個を除きました。")


def count_word_occurrences(
    sequences: np.ndarray, words: Mapping[str, Sequence[int]], continues_word: np.ndarray
) -> dict[str, int]:
    """語ごとに、トークン列が系列（行）の中に現れる回数。

    直後のトークンが語の続き（continues_word[トークン番号] が真）のものは数えない
    （例: " cat" の後に "fish" が続く " catfish"）。系列の最後で終わるものは数える。
    """
    flat = sequences.reshape(-1)
    length = sequences.shape[1]
    counts = {}
    for word, token_ids in words.items():
        starts = np.flatnonzero(flat == token_ids[0])
        starts = starts[starts % length + len(token_ids) <= length]
        for offset, token_id in enumerate(token_ids[1:], start=1):
            starts = starts[flat[starts + offset] == token_id]
        ends = starts + len(token_ids)
        inside = ends % length != 0
        complete = np.ones(len(starts), dtype=bool)
        complete[inside] = ~continues_word[flat[ends[inside]]]
        counts[word] = int(complete.sum())
    return counts


def read_pretraining_data(configuration_text: str) -> PretrainingData:
    configuration = yaml.safe_load(configuration_text)
    data = configuration["data"]
    repetition = data["instance_filter"]
    return PretrainingData(
        paths=list(data["paths"]),
        seed=int(configuration["seed"]),
        sequence_length=int(configuration["model"]["max_sequence_length"]),
        repetition_maximum_period=int(repetition["repetition_max_period"]),
        repetition_minimum_period=int(repetition["repetition_min_period"]),
        repetition_maximum_count=int(repetition["repetition_max_count"]),
    )


def sequence_starts(data: PretrainingData, sizes: dict[str, int]) -> np.ndarray:
    """data.paths の各ファイルの、最初の系列の番号（最後の要素は系列の数 N）。"""
    counts = [sizes[path] // (TOKEN_BYTES * data.sequence_length) for path in data.paths]
    return np.concatenate([[0], np.cumsum(counts, dtype=np.int64)])


def training_order(number_of_sequences: int, seed: int) -> np.ndarray:
    """学習順に並べた系列の番号（OLMo の IterableDataset と同じ並べ替え）。"""
    order = np.arange(number_of_sequences, dtype=np.uint32)
    np.random.Generator(np.random.PCG64(seed)).shuffle(order)
    return order


def has_long_repetition(
    tokens: np.ndarray, maximum_period: int, minimum_period: int, maximum_count: int
) -> bool:
    """周期 minimum_period〜maximum_period のトークン列が maximum_count 回以上続く部分があるか。

    OLMo の find_periodic_sequences（olmo/data/util.py）と同じ判定。系列を周期ごとの行に並べ、
    前の行と同じ行が続くところを探す（行の途中から始まる繰り返しは、前後の行の一致で補う）。
    """
    array = tokens.astype(np.int64)
    for period in range(minimum_period, min(maximum_period, len(array) // 3) + 1):
        padded = np.pad(array, (0, period - len(array) % period), constant_values=-1)
        rows = padded.reshape(-1, period)
        equal_to_previous = rows == np.roll(rows, shift=1, axis=0)
        (repeated_rows,) = np.where(equal_to_previous.all(axis=1))
        if len(repeated_rows) == 0:
            continue
        for run in np.split(repeated_rows, np.where(np.diff(repeated_rows) != 1)[0] + 1):
            first_row, last_row = int(run[0]), int(run[-1])
            start_offset = _start_of_last_true_run(equal_to_previous[first_row - 1])
            start_offset = period - start_offset if start_offset > 0 else 0
            end_offset = _end_of_first_true_run(equal_to_previous[last_row + 1])
            start = (first_row - 1) * period - start_offset
            end = (last_row + 1) * period + end_offset
            times = (end - start) // period
            if times > 2 and times >= maximum_count:
                return True
    return False


def _end_of_first_true_run(values: np.ndarray) -> int:
    if not values[0]:
        return 0
    cumulative = np.cumsum(values)
    if cumulative[-1] == len(values):
        return len(values)
    return int(np.where(cumulative[:-1] == cumulative[1:])[0][0]) + 1


def _start_of_last_true_run(values: np.ndarray) -> int:
    reverse = _end_of_first_true_run(values[::-1])
    return len(values) - reverse if reverse > 0 else -1


async def _file_sizes(client: httpx.AsyncClient, urls: list[str]) -> dict[str, int]:
    limit = asyncio.Semaphore(MAXIMUM_CONCURRENT_REQUESTS)

    async def size_of(url: str) -> tuple[str, int]:
        async with limit:
            response = await _request(client, "HEAD", url)
            return url, int(response.headers["Content-Length"])

    return dict(await asyncio.gather(*(size_of(url) for url in urls)))


async def _fetch_sequences(
    client: httpx.AsyncClient,
    data: PretrainingData,
    starts: np.ndarray,
    sequence_numbers: np.ndarray,
) -> np.ndarray:
    limit = asyncio.Semaphore(MAXIMUM_CONCURRENT_REQUESTS)
    sequence_bytes = TOKEN_BYTES * data.sequence_length
    sequences = np.zeros((len(sequence_numbers), data.sequence_length), dtype=np.uint32)

    async def fetch(position: int) -> None:
        file_index, sequence_in_file = _locate(starts, int(sequence_numbers[position]))
        first_byte = sequence_in_file * sequence_bytes
        headers = {"Range": f"bytes={first_byte}-{first_byte + sequence_bytes - 1}"}
        async with limit:
            response = await _request(client, "GET", data.paths[file_index], headers)
        if len(response.content) != sequence_bytes:
            raise RuntimeError(
                f"{data.paths[file_index]} から {sequence_bytes} バイトを読めません。"
            )
        sequences[position] = np.frombuffer(response.content, dtype="<u4")

    await asyncio.gather(*(fetch(position) for position in range(len(sequence_numbers))))
    return sequences


async def _request(
    client: httpx.AsyncClient, method: str, url: str, headers: dict[str, str] | None = None
) -> httpx.Response:
    for attempt in range(NUMBER_OF_ATTEMPTS):
        if attempt > 0:
            await asyncio.sleep(2**attempt)
        try:
            response = await client.request(method, url, headers=headers)
        except httpx.TransportError:
            continue
        if response.status_code < 500:
            response.raise_for_status()
            return response
    raise RuntimeError(f"{url} を {NUMBER_OF_ATTEMPTS} 回試しても取得できませんでした。")


def _locate(starts: np.ndarray, sequence_number: int) -> tuple[int, int]:
    """系列の番号を、data.paths の何番目のファイルの何番目の系列かに直す。"""
    file_index = int(np.searchsorted(starts, sequence_number, side="right")) - 1
    return file_index, sequence_number - int(starts[file_index])


def _write_sources(
    path: Path,
    data: PretrainingData,
    starts: np.ndarray,
    sequence_numbers: np.ndarray,
    removed: list[bool],
    end_of_base: int,
) -> None:
    """取り出した系列の出どころ（学習順の位置・系列の番号・ファイル・ファイルの中の位置）を書く。"""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["part", "training_order", "sequence_number", "file", "sequence_in_file", "removed"]
        )
        for position, number in enumerate(sequence_numbers.tolist()):
            file_index, sequence_in_file = _locate(starts, number)
            part = "base" if position < end_of_base else "validation"
            row = [part, position, number, data.paths[file_index], sequence_in_file]
            writer.writerow([*row, int(removed[position])])
