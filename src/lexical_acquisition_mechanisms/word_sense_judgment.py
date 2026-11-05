"""用例の候補の中で、語が意図した語義で使われているかを、LLM が出現ごとに判定する
（passages judge-word-sense）。

- 名詞（ターゲット語と自然な名詞を含む）は、WordNet の語義の説明の名詞として、形容詞（自然な形容詞を
  含む）は、その語義の形容詞として使われているかを判定する。名前・題名・慣用句などの一部は不合格。
- 用例は、全ての出現が合格したときに合格とする。
- 語と分け方（学習用・評価用）の組ごとに、鍵の小さい候補から、合格した用例が必要な数に届くまで
  判定する。合格する割合を見込んで少しずつ問い合わせ、足りなければ続きを問い合わせる。
- 実行し直したとき（例: K を変えたとき）、前回と同じ指示文と LLM で、語の語義の説明も同じなら、
  前回判定した候補（窓のトークンと出現の位置も同じもの）は前回の結果（答えが得られずに不合格に
  したものを含む）を使い、問い合わせない。
  答えの得られなかった問い合わせは保存されないので、問い合わせ直すと結果が変わり、学習に使った用例と
  ずれうるため。--recompute を付けたときは使わない。
- 出現の位置がない候補は、問い合わせずに不合格にする。
- 問い合わせる前に、組ごとの候補の数を確かめる。合格する必要がある数に届かない語があれば止まる
  （passage_shortage）。合格する見込み（候補の数 × usage_share）が、必要な数 ×
  passages.collection_factor に届かない語は、表示して続ける。
結果を data/passages/judgments.parquet に、組ごとの数を data/passages/judgment_summary.csv に書く。
"""

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet

from lexical_acquisition_mechanisms import (
    passage_collection,
    passage_shortage,
    passage_split,
    word_finalization,
    word_selection,
)
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.llm_client import (
    ANSWERS_PATH,
    LlmClient,
    LlmRequest,
    PromptTemplate,
    prompt_record_settings,
)
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.passage_shortage import needed_passing_passages
from lexical_acquisition_mechanisms.settings import Settings
from lexical_acquisition_mechanisms.word_selection import SelectedWord
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

COMMAND = "passages judge-word-sense"
OUTPUT_DIRECTORY = Path("data/passages")
JUDGMENTS_PATH = OUTPUT_DIRECTORY / "judgments.parquet"
SUMMARY_PATH = OUTPUT_DIRECTORY / "judgment_summary.csv"
NOUN_PROMPT = "judge-noun-sense"
ADJECTIVE_PROMPT = "judge-adjective-sense"
SCHEMA = {
    "type": "object",
    "properties": {"intended": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["intended", "reason"],
    "additionalProperties": False,
}
# 1回に問い合わせる候補の数 = 不足の数 × REQUEST_MARGIN ÷ 見込みの合格の割合
# （MINIMUM_REQUESTS 以上）
REQUEST_MARGIN = 1.2
MINIMUM_PASS_RATE = 0.2
MINIMUM_REQUESTS = 20
# judgments.parquet の、判定に使った指示文と LLM を書くメタデータの鍵
JUDGMENT_SETTINGS_KEY = b"judgment_settings"
# judgments.parquet の、判定に使った語ごとの語義の説明を書くメタデータの鍵
JUDGMENT_DEFINITIONS_KEY = b"judgment_definitions"


@dataclass
class JudgmentSummary:
    word: str
    role: str
    split: str
    # 合格する必要がある数と、判定した数・合格した数
    needed: int
    judged: int
    passed: int


def judge_word_sense(settings: Settings, arguments: argparse.Namespace) -> None:
    prompts = {
        "noun": PromptTemplate.load(NOUN_PROMPT),
        "adjective": PromptTemplate.load(ADJECTIVE_PROMPT),
    }
    record_settings = {
        **prompt_record_settings(settings.llm, list(prompts.values())),
        "passages": settings.passages.model_dump(include={"evaluation_passages_per_word"}),
        "unknown_word_acquisition": settings.unknown_word_acquisition.model_dump(
            include={"exposure_levels"}
        ),
        "sense_addition": settings.sense_addition.model_dump(include={"passages_per_added_sense"}),
        "evaluation": settings.evaluation.model_dump(include={"passages_per_natural_word"}),
    }
    step = Step(
        COMMAND,
        inputs={
            word_finalization.OUTPUT_PATH: word_finalization.COMMAND,
            passage_collection.SUMMARY_PATH: passage_collection.COMMAND,
            passage_split.SPLIT_PATH: passage_split.COMMAND,
        },
        outputs=[JUDGMENTS_PATH, SUMMARY_PATH],
        settings=record_settings,
    )
    client = LlmClient(settings.llm, ANSWERS_PATH, rerun_from=COMMAND)
    run_unless_unchanged(
        step,
        lambda: build_judgments(
            settings,
            prompts,
            client,
            word_finalization.OUTPUT_PATH,
            passage_split.SPLIT_PATH,
            (passage_collection.SUMMARY_PATH, word_selection.EXCLUDED_WORDS_PATH),
            (JUDGMENTS_PATH, SUMMARY_PATH),
            reuse=not arguments.recompute,
        ),
        arguments.recompute,
    )


def build_judgments(
    settings: Settings,
    prompts: Mapping[str, PromptTemplate],
    client: LlmClient,
    words_path: Path,
    split_path: Path,
    shortage_paths: tuple[Path, Path],
    output_paths: tuple[Path, Path],
    reuse: bool = True,
) -> None:
    """shortage_paths は、collect の語ごとの数（collection_summary.csv）と、外す語の CSV。

    reuse なら、前回の判定の結果を使う（指示文・LLM・語の語義の説明が同じ候補だけ）。
    """
    judgments_path, summary_path = output_paths
    words = {word.word: word for word in read_rows(words_path, SelectedWord)}
    tokenizer = WordTokenizer.from_model(next(iter(settings.models.values())))
    # 候補は多い（数百万件）ので、判定する候補だけを、その都度 Python の値にする
    table = parquet.read_table(split_path)
    schema = table.schema
    table = contiguous_with_large_lists(table)
    remaining = _indices_by_group(table)
    needed = {
        group: needed_passing_passages(words[group[0]].role, group[1], settings)
        for group in remaining
    }
    _check_candidates(settings, words, remaining, shortage_paths)
    judged: Counter[tuple[str, str]] = Counter()
    passed: Counter[tuple[str, str]] = Counter()
    judged_indices: list[int] = []
    accepted_values: list[bool] = []
    reason_lists: list[list[str]] = []
    signature = json.dumps(
        prompt_record_settings(settings.llm, list(prompts.values())), sort_keys=True
    )
    definitions = {word.word: word.definition for word in words.values()}
    previous = _previous_judgments(judgments_path, signature, definitions) if reuse else {}
    while batch := _next_batch(remaining, needed, judged, passed, words):
        candidates = table.take([index for _, index in batch]).to_pylist()
        requests = []
        owners = []
        for number, candidate in enumerate(candidates):
            if _candidate_key(candidate) in previous:
                continue
            word = words[candidate["word"]]
            for position in candidate["occurrence_positions"]:
                requests.append(_request(word, candidate, position, prompts, tokenizer))
                owners.append(number)
        verdicts: list[list[dict[str, Any] | None]] = [[] for _ in batch]
        for owner, answer in zip(owners, client.ask_all(requests), strict=True):
            verdicts[owner].append(answer)
        for (group, index), candidate, answers in zip(batch, candidates, verdicts, strict=True):
            if _candidate_key(candidate) in previous:
                accepted, reasons = previous[_candidate_key(candidate)]
            else:
                accepted = bool(answers) and all(
                    answer is not None and answer["intended"] for answer in answers
                )
                reasons = [answer["reason"] if answer else "判定できなかった" for answer in answers]
            judged[group] += 1
            passed[group] += accepted
            judged_indices.append(index)
            accepted_values.append(accepted)
            reason_lists.append(reasons)
    # 判定した候補だけなら、元の型（list）に戻せる
    results = table.take(judged_indices).cast(schema)
    results = results.append_column("passed", arrow.array(accepted_values, arrow.bool_()))
    reasons = arrow.array(reason_lists, arrow.list_(arrow.string()))
    results = results.append_column("reasons", reasons)
    metadata = {
        **(results.schema.metadata or {}),
        JUDGMENT_SETTINGS_KEY: signature.encode(),
        JUDGMENT_DEFINITIONS_KEY: json.dumps(definitions, sort_keys=True).encode(),
    }
    parquet.write_table(results.replace_schema_metadata(metadata), judgments_path)
    summaries = [
        JudgmentSummary(
            word,
            words[word].role,
            split,
            needed[(word, split)],
            judged[(word, split)],
            passed[(word, split)],
        )
        for word, split in sorted(remaining)
    ]
    write_rows(summary_path, JudgmentSummary, summaries)


def _check_candidates(
    settings: Settings,
    words: Mapping[str, SelectedWord],
    remaining: Mapping[tuple[str, str], list[int]],
    shortage_paths: tuple[Path, Path],
) -> None:
    """問い合わせる前に、組ごとの候補の数が、合格する必要がある数に届くかを確かめる。"""
    managed = [word for word in words.values() if word.role != "unknown"]
    counts = {group: len(indices) for group, indices in remaining.items()}
    if shortages := passage_shortage.find_shortages(managed, counts, settings):
        passage_shortage.stop_for_shortage(shortages, "判定に回す候補", *shortage_paths)
    factor = settings.passages.collection_factor
    few = []
    for word in managed:
        share = max(word.usage_share, passage_collection.MINIMUM_USAGE_SHARE)
        for split in passage_shortage.SPLITS:
            needed = passage_shortage.needed_passing_passages(word.role, split, settings)
            found = counts.get((word.word, split), 0)
            if needed and found * share < needed * factor:
                few.append(passage_shortage.Shortage(word.word, word.role, split, found, needed))
    if few:
        print(
            "合格する見込み（判定に回す候補の数 × usage_share）が、合格する必要がある数の"
            f" {factor} 倍に届かない語があります（{passage_shortage.listed(few)}）。合格する割合が"
            "見込みより低いと、passages finalize で足りなくなります。"
        )


def contiguous_with_large_lists(table: arrow.Table) -> arrow.Table:
    """リストの列を large_list にし、かたまりを1つにつなげた表。

    リストの列の take は、かたまりをつなげてから取る。候補（数百万件 × 用例の長さ）では、要素の数が
    2^31 を超えて list のオフセットがあふれ、問い合わせのたびにつなぎ直すのも遅いので、先に1回だけ
    つなげる。
    """
    fields = [
        arrow.field(field.name, arrow.large_list(field.type.value_type), field.nullable)
        if arrow.types.is_list(field.type)
        else field
        for field in table.schema
    ]
    return table.cast(arrow.schema(fields, table.schema.metadata)).combine_chunks()


def _candidate_key(candidate: Mapping[str, Any]) -> tuple[str, int, int, int, bytes]:
    """候補の鍵: 語・断片・行・始まりと、窓の中身（トークンと出現の位置）の要約。"""
    window = _window_digest(
        np.asarray(candidate["token_ids"], dtype=np.int64),
        np.asarray(candidate["occurrence_positions"], dtype=np.int64),
    )
    return candidate["word"], candidate["shard"], candidate["row"], candidate["start"], window


def _window_digest(token_ids: np.ndarray, positions: np.ndarray) -> bytes:
    """窓のトークンと出現の位置の要約（設定を変えて切り出し直すと、同じ始まりでも窓が変わりうる）。"""
    data = token_ids.astype(np.int64).tobytes() + b"|" + positions.astype(np.int64).tobytes()
    return hashlib.blake2b(data, digest_size=16).digest()


def _list_values(column: arrow.ChunkedArray) -> list[np.ndarray]:
    """リストの列の、行ごとの値（Python の値に直さずに取り出す）。"""
    combined = column.combine_chunks()
    values = combined.values.to_numpy(zero_copy_only=False)
    offsets = combined.offsets.to_numpy()
    return [values[offsets[index] : offsets[index + 1]] for index in range(len(combined))]


def _previous_judgments(
    path: Path, signature: str, definitions: Mapping[str, str]
) -> dict[tuple[str, int, int, int, bytes], tuple[bool, list[str]]]:
    """前回の判定の結果（候補ごとの合否と理由）。

    指示文か LLM が違えば、使わない（空）。語の語義の説明（definitions）が前回と違う語の候補も
    使わない。
    """
    if not path.exists():
        return {}
    table = parquet.read_table(path)
    metadata = table.schema.metadata or {}
    if metadata.get(JUDGMENT_SETTINGS_KEY) != signature.encode():
        return {}
    judged_with = json.loads(metadata.get(JUDGMENT_DEFINITIONS_KEY, b"{}"))
    rows = table.select(["word", "shard", "row", "start", "passed", "reasons"]).to_pylist()
    windows = zip(
        _list_values(table.column("token_ids")),
        _list_values(table.column("occurrence_positions")),
        strict=True,
    )
    return {
        (row["word"], row["shard"], row["row"], row["start"], _window_digest(*window)): (
            row["passed"],
            row["reasons"],
        )
        for row, window in zip(rows, windows, strict=True)
        if row["word"] in judged_with and judged_with[row["word"]] == definitions.get(row["word"])
    }


def _indices_by_group(table: arrow.Table) -> dict[tuple[str, str], list[int]]:
    """語と分け方の組ごとの、候補の行の番号（鍵の小さい順）。"""
    words = np.array(table.column("word").to_pylist())
    splits = np.array(table.column("split").to_pylist())
    keys = table.column("key").to_numpy()
    groups = {}
    for word, split in sorted(set(zip(words, splits, strict=True))):
        indices = np.flatnonzero((words == word) & (splits == split))
        groups[(str(word), str(split))] = indices[np.argsort(keys[indices])].tolist()
    return groups


def _next_batch(
    remaining: dict[tuple[str, str], list[int]],
    needed: Mapping[tuple[str, str], int],
    judged: Mapping[tuple[str, str], int],
    passed: Mapping[tuple[str, str], int],
    words: Mapping[str, SelectedWord],
) -> list[tuple[tuple[str, str], int]]:
    """組ごとに、合格の不足を埋めるのに見込まれる数の候補（行の番号）を、鍵の小さい順に取り出す。"""
    batch = []
    for group, indices in remaining.items():
        deficit = needed[group] - passed[group]
        if deficit <= 0 or not indices:
            continue
        rate = passed[group] / judged[group] if judged[group] else words[group[0]].usage_share
        count = max(
            MINIMUM_REQUESTS, math.ceil(deficit * REQUEST_MARGIN / max(rate, MINIMUM_PASS_RATE))
        )
        batch += [(group, index) for index in indices[:count]]
        del indices[:count]
    return batch


def _request(
    word: SelectedWord,
    candidate: Mapping[str, Any],
    position: int,
    prompts: Mapping[str, PromptTemplate],
    tokenizer: WordTokenizer,
) -> LlmRequest:
    """出現を [[ ]] で囲んだ用例で、語が意図した語義で使われているかを問う。"""
    token_ids = candidate["token_ids"]
    end = position + word.token_count
    passage = (
        tokenizer.decode(token_ids[:position])
        + f" [[{word.word}]]"
        + tokenizer.decode(token_ids[end:])
    )
    part_of_speech = "adjective" if word.role.endswith("adjective") else "noun"
    values = {"word": word.word, "definition": word.definition, "passage": passage}
    return LlmRequest(prompts[part_of_speech], values, SCHEMA)
