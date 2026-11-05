"""Pile から、選んだ語を含む用例の候補を切り出す（passages collect）。

1. Pile の全ての断片で、語（words finalize が書いた語のうち、未知語を除く全て）の標準の形の出現を
   探す。数え方は Pile の見本（pile_word_usages）と同じで、文書ごとに、前に数えた出現から用例1つ分の
   文字数以上離れた出現を、1つの文書から取る数の上限まで取る。
2. 出現ごとに、位置と seed から決まる鍵を付け、語ごとに鍵の小さいものから、必要な数 ×
   passages.collection_factor × OCCURRENCE_MARGIN 個を残す（無作為に選んだのと同じになり、断片を
   読む順によらない）。窓の除外と重複の除去で減る分を見込み、判定に回す候補が、必要な数 ×
   collection_factor 個ほどになるようにする。断片ごとには、Pile 全体での用例の見積もりから決めた
   鍵の上限より小さい出現だけを残す（上限は、残す数の passages.key_limit_margin 倍の出現が入る値。
   足りなければ止まる）。
3. 残した出現の前後の文字列を、data/passages/occurrences.parquet に書く。
4. 前後の文字列から、語を含む用例の長さの窓を切り出して確かめ（passage_windows）、使える窓を
   data/passages/candidates.parquet に、語ごとの数を data/passages/collection_summary.csv に書く。
断片ごとの結果は data/passages/collection_shards/ に残し、止まった後に実行し直すと、続きから読む
（全てを書き終えたら消す）。
"""

import argparse
import heapq
import json
import math
import shutil
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import NamedTuple

import pyarrow as arrow
import pyarrow.parquet as parquet

from lexical_acquisition_mechanisms import pile, word_finalization
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.inflected_forms import inflected_forms
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.passage_windows import WindowSettings, cut_windows
from lexical_acquisition_mechanisms.pile import PileDocument
from lexical_acquisition_mechanisms.pile_word_usages import (
    characters_per_passage,
    occurrence_key,
    standard_form_occurrences,
)
from lexical_acquisition_mechanisms.settings import Settings
from lexical_acquisition_mechanisms.word_selection import SelectedWord

COMMAND = "passages collect"
OUTPUT_DIRECTORY = Path("data/passages")
OCCURRENCES_PATH = OUTPUT_DIRECTORY / "occurrences.parquet"
CANDIDATES_PATH = OUTPUT_DIRECTORY / "candidates.parquet"
SUMMARY_PATH = OUTPUT_DIRECTORY / "collection_summary.csv"
SHARD_DIRECTORY = OUTPUT_DIRECTORY / "collection_shards"
MANAGED_ROLES = ("noun", "adjective", "target")
# 出現の前後に残す文字数の、用例のトークン数に対する倍率（1トークンが7文字までなら、
# 出現がどこにあっても、前後に用例1つ分のトークンが入る）
CONTEXT_CHARACTERS_PER_TOKEN = 7
# 残す出現の数の、判定に回す候補の数（必要な数 × collection_factor）に対する倍率（窓の除外と、
# 次の段階の重複の除去で減る分を見込む）
OCCURRENCE_MARGIN = 1.5
NUMBER_OF_PROCESSES = 24


SHARD_SCHEMA = arrow.schema(
    [
        ("word", arrow.string()),
        ("key", arrow.float64()),
        ("shard", arrow.int32()),
        ("row", arrow.int32()),
        ("start", arrow.int64()),
        ("context_start", arrow.int64()),
        ("context", arrow.string()),
    ]
)


CANDIDATE_SCHEMA = arrow.schema(
    [
        ("word", arrow.string()),
        ("role", arrow.string()),
        ("shard", arrow.int32()),
        ("row", arrow.int32()),
        ("start", arrow.int64()),
        ("key", arrow.float64()),
        ("token_ids", arrow.list_(arrow.int32())),
        ("occurrence_positions", arrow.list_(arrow.int16())),
    ]
)


@dataclass
class CollectionSummary:
    word: str
    role: str
    # 判定に合格する必要がある数と、残した出現・使える窓の数
    needed: int
    occurrences: int
    candidates: int
    # 窓を使わなかった理由と、その数
    rejected: str
    # 残した出現が、残す数（kept_occurrences）に届いたか（届いていれば、Pile にはさらに出現が
    # ありうる。届かなければ、Pile の出現を全て残した）
    reached_limit: bool


class Occurrence(NamedTuple):
    key: float
    shard: int
    row: int
    # 出現（先頭スペースの後の、語の最初の文字）の、文書の中の位置
    start: int
    # 残した前後の文字列と、その文書の中の開始位置
    context_start: int
    context: str


def needed_passages(word: SelectedWord, settings: Settings) -> int:
    """判定に合格する必要がある用例の数。形容詞は、K を露出回数の最大にしても足りる数にする。"""
    evaluation = settings.passages.evaluation_passages_per_word
    if word.role in {"noun", "adjective"}:
        return max(settings.unknown_word_acquisition.exposure_levels) + evaluation
    if word.role == "target":
        return evaluation
    return settings.evaluation.passages_per_natural_word


def kept_occurrences(word: SelectedWord, settings: Settings) -> int:
    """語ごとに残す出現の数（必要な数 × passages.collection_factor × OCCURRENCE_MARGIN）。"""
    factor = settings.passages.collection_factor * OCCURRENCE_MARGIN
    return math.ceil(needed_passages(word, settings) * factor)


def context_characters(settings: Settings) -> int:
    """出現の前後に残す文字数（窓を切り出すときも、同じ値で文書の端かを判断する）。"""
    return settings.passages.tokens_per_passage * CONTEXT_CHARACTERS_PER_TOKEN


def collect_passages(settings: Settings, arguments: argparse.Namespace) -> None:
    record_settings = {
        "passages": settings.passages.model_dump(
            include={
                "tokens_per_passage",
                "maximum_passages_per_document",
                "evaluation_passages_per_word",
                "collection_factor",
                "key_limit_margin",
            }
        ),
        "unknown_word_acquisition": settings.unknown_word_acquisition.model_dump(
            include={"exposure_levels"}
        ),
        "evaluation": settings.evaluation.model_dump(include={"passages_per_natural_word"}),
        "seed": settings.words.seed,
        "tokenizer": next(iter(settings.models.values())).model_dump(),
    }
    step = Step(
        COMMAND,
        inputs={word_finalization.OUTPUT_PATH: word_finalization.COMMAND},
        outputs=[OCCURRENCES_PATH, CANDIDATES_PATH, SUMMARY_PATH],
        settings=record_settings,
    )
    run_unless_unchanged(
        step,
        lambda: build_passage_candidates(settings, word_finalization.OUTPUT_PATH, OUTPUT_DIRECTORY),
        arguments.recompute,
    )


def build_passage_candidates(
    settings: Settings,
    words_path: Path,
    output_directory: Path,
    number_of_processes: int = NUMBER_OF_PROCESSES,
    download_directory: Path = pile.DOWNLOAD_DIRECTORY,
    download: Callable[[str, Path], Path] = pile.download_shard,
) -> None:
    words = [word for word in read_rows(words_path, SelectedWord) if word.role != "unknown"]
    build_occurrences(
        settings, words, output_directory, number_of_processes, download_directory, download
    )
    occurrences = parquet.read_table(output_directory / OCCURRENCES_PATH.name).to_pylist()
    forms = {word.word: inflected_forms(word.word, word.role) for word in words}
    window_settings = WindowSettings(
        tokens_per_passage=settings.passages.tokens_per_passage,
        context_characters=context_characters(settings),
        seed=settings.words.seed,
        other_forms=forms,
        managed_forms={
            form: word.word
            for word in words
            if word.role in MANAGED_ROLES
            for form in (word.word, *forms[word.word])
        },
    )
    source = next(iter(settings.models.values()))
    windows = cut_windows(occurrences, source, window_settings, number_of_processes)
    _write_candidates(settings, words, occurrences, windows, output_directory)
    shutil.rmtree(output_directory / SHARD_DIRECTORY.name)


def _write_candidates(
    settings: Settings,
    words: Sequence[SelectedWord],
    occurrences: Sequence[dict],
    windows: Sequence,
    output_directory: Path,
) -> None:
    rows = []
    rejected: dict[str, Counter[str]] = {word.word: Counter() for word in words}
    for occurrence, window in zip(occurrences, windows, strict=True):
        if isinstance(window, str):
            rejected[occurrence["word"]][window] += 1
            continue
        rows.append(
            {
                **{name: occurrence[name] for name in ("word", "role", "shard", "row", "start")},
                "key": occurrence["key"],
                "token_ids": window.token_ids,
                "occurrence_positions": window.occurrence_positions,
            }
        )
    table = arrow.Table.from_pylist(rows, schema=CANDIDATE_SCHEMA)
    parquet.write_table(table, output_directory / CANDIDATES_PATH.name)
    found = Counter(occurrence["word"] for occurrence in occurrences)
    kept = Counter(row["word"] for row in rows)
    summary = [
        CollectionSummary(
            word.word,
            word.role,
            needed_passages(word, settings),
            found[word.word],
            kept[word.word],
            ", ".join(f"{reason}: {number}" for reason, number in rejected[word.word].items()),
            found[word.word] >= kept_occurrences(word, settings),
        )
        for word in words
    ]
    write_rows(output_directory / SUMMARY_PATH.name, CollectionSummary, summary)


def build_occurrences(
    settings: Settings,
    words: Sequence[SelectedWord],
    output_directory: Path,
    number_of_processes: int,
    download_directory: Path,
    download: Callable[[str, Path], Path],
) -> None:
    """語ごとに、鍵の小さい出現を残し、前後の文字列を書く。"""
    kept_numbers = {word.word: kept_occurrences(word, settings) for word in words}
    margin = settings.passages.key_limit_margin
    key_limits = {
        word.word: min(1.0, margin * kept_numbers[word.word] / max(word.estimated_pile_passages, 1))
        for word in words
    }
    scan = partial(
        scan_documents_for_passages,
        key_limits=key_limits,
        seed=settings.words.seed,
        characters_per_passage=characters_per_passage(settings.passages),
        maximum_passages_per_document=settings.passages.maximum_passages_per_document,
        context_characters=context_characters(settings),
    )
    shard_directory = _prepare_shard_directory(
        output_directory / SHARD_DIRECTORY.name, scan.keywords
    )
    heaps: dict[str, list[tuple[float, Occurrence]]] = {word.word: [] for word in words}
    results = _shard_results(
        scan, shard_directory, number_of_processes, download_directory, download
    )
    for number, occurrences in enumerate(results, start=1):
        for word, found in occurrences.items():
            heap = heaps[word]
            for occurrence in found:
                if len(heap) < kept_numbers[word]:
                    heapq.heappush(heap, (-occurrence.key, occurrence))
                elif occurrence.key < -heap[0][0]:
                    heapq.heapreplace(heap, (-occurrence.key, occurrence))
        if number % 50 == 0:
            print(f"Pile の断片を {number} / {pile.NUMBER_OF_SHARDS} 個読みました。")
    short = [
        word.word
        for word in words
        if len(heaps[word.word]) < kept_numbers[word.word] and key_limits[word.word] < 1
    ]
    if short:
        raise CommandStopped(
            "Pile での用例の見積もりより出現が少なく、残す数に届かなかった語があります"
            f"（{short}）。settings.yaml の passages.key_limit_margin を大きくしてください。",
            rerun_from=COMMAND,
        )
    roles = {word.word: word.role for word in words}
    rows = [
        {"word": word, "role": roles[word], **occurrence._asdict()}
        for word, heap in heaps.items()
        for _, occurrence in sorted(heap, key=lambda pair: -pair[0])
    ]
    parquet.write_table(arrow.Table.from_pylist(rows), output_directory / OCCURRENCES_PATH.name)


def scan_documents_for_passages(
    documents: Iterator[PileDocument],
    key_limits: Mapping[str, float],
    seed: int,
    characters_per_passage: int,
    maximum_passages_per_document: int,
    context_characters: int,
) -> dict[str, list[Occurrence]]:
    """1つの断片の文書で、用例として数える出現のうち、鍵が上限より小さいものを残す。"""
    found: dict[str, list[Occurrence]] = {}
    for document in documents:
        text = document.text
        occurrences = standard_form_occurrences(
            text, key_limits, characters_per_passage, maximum_passages_per_document
        )
        for match, counted in occurrences:
            key = occurrence_key(seed, document.shard, document.row, match.start())
            if not counted or key >= key_limits[match.group()]:
                continue
            context_start = max(0, match.start() - context_characters)
            context = text[context_start : match.end() + context_characters]
            found.setdefault(match.group(), []).append(
                Occurrence(key, document.shard, document.row, match.start(), context_start, context)
            )
    return found


def _prepare_shard_directory(directory: Path, conditions: Mapping[str, object]) -> Path:
    """断片ごとの結果を置くディレクトリ。前回と条件（語ごとの鍵の上限など）が違えば、中身を消す。"""
    text = json.dumps(conditions, sort_keys=True, ensure_ascii=False)
    conditions_path = directory / "conditions.json"
    if directory.exists() and (
        not conditions_path.exists() or conditions_path.read_text(encoding="utf-8") != text
    ):
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    conditions_path.write_text(text, encoding="utf-8")
    return directory


def _shard_results(
    scan: Callable[[Iterator[PileDocument]], dict[str, list[Occurrence]]],
    directory: Path,
    number_of_processes: int,
    download_directory: Path,
    download: Callable[[str, Path], Path],
) -> Iterator[dict[str, list[Occurrence]]]:
    """断片ごとの結果。前回までに読んだ断片は保存した結果を返し、残りの断片を読んで保存する。"""
    missing = []
    for shard in range(pile.NUMBER_OF_SHARDS):
        path = directory / f"shard-{shard:05d}.parquet"
        if path.exists():
            yield _read_shard_result(path)
        else:
            missing.append(shard)
    results = pile.scan_shards(missing, scan, number_of_processes, download_directory, download)
    for shard, result in zip(missing, results, strict=True):
        rows = [
            {"word": word, **occurrence._asdict()}
            for word, occurrences in result.items()
            for occurrence in occurrences
        ]
        path = directory / f"shard-{shard:05d}.parquet"
        temporary_path = path.with_suffix(".writing")
        parquet.write_table(arrow.Table.from_pylist(rows, schema=SHARD_SCHEMA), temporary_path)
        temporary_path.replace(path)
        yield result


def _read_shard_result(path: Path) -> dict[str, list[Occurrence]]:
    result: dict[str, list[Occurrence]] = {}
    for row in parquet.read_table(path).to_pylist():
        word = row.pop("word")
        result.setdefault(word, []).append(Occurrence(**row))
    return result
