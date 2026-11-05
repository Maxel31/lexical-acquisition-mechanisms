"""ほぼ同じ用例の候補を除き、元の文書の単位で学習用と評価用に分ける
（passages remove-duplicates-and-split）。

1. 語ごとに、ほぼ同じ候補（トークン列の8割以上が一致する。difflib の一致率）を除く。鍵の小さい
   候補から順に残す。似た組は MinHash（3トークンの並びの集合から作る短い署名で、似た集合ほど
   一致しやすい）で探し、トークン列の一致率で確かめる（違いが散らばった組も見つかるように、並びを
   短くし、並びの集合ではなくトークン列を比べる）。
2. 元の文書（Pile の断片と行）ごとに、全ての語で共通の分け方を1つ決める（同じ文書の候補は、どの語
   でも、学習用と評価用の両方には入らない）。ターゲット語か自然な語の候補を含む文書（これらの語は
   全て評価用）は評価用にする。ほかの文書は、seed と文書から決まる鍵が、評価用の用例の割合より
   小さければ評価用にする。
3. 評価用の候補のうち、基データと32トークン以上続けて同じ並びを含むものを除く。
結果を data/passages/split_candidates.parquet に、語ごとの数を data/passages/split_summary.csv に
書く。
"""

import argparse
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pyarrow as arrow
import pyarrow.compute as compute
import pyarrow.parquet as parquet

from lexical_acquisition_mechanisms import base_data, passage_collection
from lexical_acquisition_mechanisms.csv_tables import write_rows
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.pile_word_usages import occurrence_key
from lexical_acquisition_mechanisms.settings import Settings

COMMAND = "passages remove-duplicates-and-split"
OUTPUT_DIRECTORY = Path("data/passages")
SPLIT_PATH = OUTPUT_DIRECTORY / "split_candidates.parquet"
SUMMARY_PATH = OUTPUT_DIRECTORY / "split_summary.csv"
SHINGLE_LENGTH = 3
DUPLICATE_SHARE = 0.8
NUMBER_OF_HASHES = 64
ROWS_PER_BAND = 4
NEARBY_MEMBERS = 50
BASE_DATA_OVERLAP_LENGTH = 32
# 学習用の用例も使う役割（ほかの役割の語は、全て評価用）
TRAINING_ROLES = ("noun", "adjective")
# 文書の順を決める鍵の seed に足す数（出現や窓の位置を決めた鍵と別の数にするため）
DOCUMENT_SEED_OFFSET = 2
HASH_MULTIPLIER = np.uint64(1099511628211)


@dataclass
class SplitSummary:
    word: str
    role: str
    candidates: int
    duplicates: int
    overlapping_base_data: int
    training: int
    evaluation: int


def remove_duplicates_and_split(settings: Settings, arguments: argparse.Namespace) -> None:
    base_path = base_data.output_paths()["base"]
    step = Step(
        COMMAND,
        inputs={
            passage_collection.CANDIDATES_PATH: passage_collection.COMMAND,
            base_path: base_data.COMMAND,
        },
        outputs=[SPLIT_PATH, SUMMARY_PATH],
        settings={
            "passages": settings.passages.model_dump(include={"evaluation_passages_per_word"}),
            "unknown_word_acquisition": settings.unknown_word_acquisition.model_dump(
                include={"exposure_levels"}
            ),
            "seed": settings.words.seed,
        },
    )
    run_unless_unchanged(
        step,
        lambda: build_split(
            settings, passage_collection.CANDIDATES_PATH, base_path, SPLIT_PATH, SUMMARY_PATH
        ),
        arguments.recompute,
    )


def build_split(
    settings: Settings,
    candidates_path: Path,
    base_sequences_path: Path,
    split_path: Path,
    summary_path: Path,
) -> None:
    table = parquet.read_table(candidates_path)
    evaluation_share = settings.passages.evaluation_passages_per_word / (
        max(settings.unknown_word_acquisition.exposure_levels)
        + settings.passages.evaluation_passages_per_word
    )
    # ターゲット語と自然な語の候補を含む文書（全て評価用にする）
    evaluation_documents = {
        (row["shard"], row["row"])
        for row in table.select(["role", "shard", "row"]).to_pylist()
        if row["role"] not in TRAINING_ROLES
    }
    kept_tables = []
    summaries = []
    for word in sorted(set(table.column("word").to_pylist())):
        rows = table.filter(compute.equal(table.column("word"), word)).sort_by("key")
        windows = np.array(rows.column("token_ids").to_pylist(), dtype=np.int64)
        unique = _unique_indices(windows)
        rows = rows.take(unique)
        role = rows.column("role")[0].as_py()
        share = evaluation_share if role in TRAINING_ROLES else 1.0
        splits = _document_splits(rows, evaluation_documents, share, settings.words.seed)
        kept_tables.append(rows.append_column("split", arrow.array(splits)))
        duplicates = len(windows) - len(unique)
        summaries.append(SplitSummary(word, role, len(windows), duplicates, 0, 0, 0))
    candidates = arrow.concat_tables(kept_tables)
    overlapping = _overlaps_base_data(candidates, np.load(base_sequences_path, mmap_mode="r"))
    removed = Counter(np.array(candidates.column("word").to_pylist())[overlapping].tolist())
    split = candidates.filter(arrow.array(~overlapping))
    counts = Counter(
        zip(split.column("word").to_pylist(), split.column("split").to_pylist(), strict=True)
    )
    for summary in summaries:
        summary.overlapping_base_data = removed[summary.word]
        summary.training = counts[(summary.word, "training")]
        summary.evaluation = counts[(summary.word, "evaluation")]
    parquet.write_table(split, split_path)
    write_rows(summary_path, SplitSummary, summaries)


def _shingle_hashes(windows: np.ndarray, length: int) -> np.ndarray:
    """窓ごとの、length トークンの並びのハッシュ（形は 窓 × 並びの数）。"""
    count = windows.shape[1] - length + 1
    hashes = np.zeros((windows.shape[0], count), dtype=np.uint64)
    tokens = windows.astype(np.uint64)
    with np.errstate(over="ignore"):
        for offset in range(length):
            hashes = hashes * HASH_MULTIPLIER + tokens[:, offset : offset + count]
    return hashes


def _unique_indices(windows: np.ndarray) -> list[int]:
    """ほぼ同じ窓を除いた、残す窓の番号（前の窓から順に残す）。"""
    shingles = _shingle_hashes(windows, SHINGLE_LENGTH)
    generator = np.random.default_rng(0)
    multipliers = generator.integers(1, 2**63, NUMBER_OF_HASHES, dtype=np.uint64) | np.uint64(1)
    with np.errstate(over="ignore"):
        signatures = np.stack(
            [(shingles * multiplier).min(axis=1) for multiplier in multipliers], axis=1
        )
    similar: dict[int, set[int]] = {}
    for band in range(0, NUMBER_OF_HASHES, ROWS_PER_BAND):
        groups: dict[bytes, list[int]] = {}
        for index, row in enumerate(signatures[:, band : band + ROWS_PER_BAND]):
            groups.setdefault(row.tobytes(), []).append(index)
        for members in groups.values():
            # 大きな組で比べる数が増えすぎないように、先頭と、直前の数個とだけ比べる
            for position, index in enumerate(members[1:], start=1):
                earlier = [members[0], *members[max(1, position - NEARBY_MEMBERS) : position]]
                similar.setdefault(index, set()).update(earlier)
    tokens = windows.tolist()
    kept: list[int] = []
    kept_set: set[int] = set()
    for index in range(len(windows)):
        duplicate = any(
            _similar_enough(tokens[index], tokens[other])
            for other in similar.get(index, ())
            if other in kept_set
        )
        if not duplicate:
            kept.append(index)
            kept_set.add(index)
    return kept


def _similar_enough(first: list[int], second: list[int]) -> bool:
    """トークン列の一致率（2 × 一致したトークンの数 ÷ 2つの長さの和）が DUPLICATE_SHARE 以上か。"""
    matcher = SequenceMatcher(None, first, second, autojunk=False)
    return (
        matcher.real_quick_ratio() >= DUPLICATE_SHARE
        and matcher.quick_ratio() >= DUPLICATE_SHARE
        and matcher.ratio() >= DUPLICATE_SHARE
    )


def _document_splits(
    rows: arrow.Table, evaluation_documents: set[tuple[int, int]], share: float, seed: int
) -> list[str]:
    """候補ごとの分け方（文書ごとに、全ての語で同じになる）。

    evaluation_documents の文書と、文書の鍵が share より小さい文書を評価用にする。
    """
    documents = zip(rows.column("shard").to_pylist(), rows.column("row").to_pylist(), strict=True)
    return [
        "evaluation"
        if document in evaluation_documents
        or occurrence_key(seed + DOCUMENT_SEED_OFFSET, *document, 0) < share
        else "training"
        for document in documents
    ]


def token_rows(column: arrow.ChunkedArray, indices: np.ndarray) -> np.ndarray:
    """リストの列（各行の長さが同じ）の、indices（小さい順）の行を2次元の配列にする。

    リストの列の take は、かたまりをつなげてから取るので、列の全体の要素の数が 2^31 を超えると
    オフセットがあふれる。かたまりごとに取り出す。
    """
    if np.any(np.diff(indices) < 0):
        raise ValueError("indices は小さい順に並べてください。")
    parts = []
    start = 0
    for chunk in column.chunks:
        stop = start + len(chunk)
        mine = indices[(indices >= start) & (indices < stop)] - start
        if len(mine):
            taken = chunk.take(arrow.array(mine))
            values = taken.flatten().to_numpy(zero_copy_only=False)
            parts.append(values.reshape(len(mine), -1).astype(np.int64))
        start = stop
    return np.concatenate(parts) if parts else np.zeros((0, 0), dtype=np.int64)


def _overlaps_base_data(split: arrow.Table, base_sequences: np.ndarray) -> np.ndarray:
    """評価用の候補のうち、基データと BASE_DATA_OVERLAP_LENGTH トークン続けて同じ並びを含むもの。"""
    overlapping = np.zeros(len(split), dtype=bool)
    evaluation = np.flatnonzero(np.array(split.column("split").to_pylist()) == "evaluation")
    if len(evaluation) == 0:
        return overlapping
    windows = token_rows(split.column("token_ids"), evaluation)
    window_hashes = _shingle_hashes(windows, BASE_DATA_OVERLAP_LENGTH)
    sorted_hashes = np.unique(window_hashes)
    found = np.zeros(len(sorted_hashes), dtype=bool)
    for start in range(0, base_sequences.shape[0], 10_000):
        chunk = np.asarray(base_sequences[start : start + 10_000], dtype=np.int64)
        hashes = _shingle_hashes(chunk, BASE_DATA_OVERLAP_LENGTH).reshape(-1)
        positions = np.searchsorted(sorted_hashes, hashes).clip(max=len(sorted_hashes) - 1)
        found[positions[sorted_hashes[positions] == hashes]] = True
    matched = found[np.searchsorted(sorted_hashes, window_hashes)]
    overlapping[evaluation[matched.any(axis=1)]] = True
    return overlapping
