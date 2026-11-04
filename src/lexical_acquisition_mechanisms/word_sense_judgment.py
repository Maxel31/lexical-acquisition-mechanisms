"""用例の候補の中で、語が意図した語義で使われているかを、LLM が出現ごとに判定する
（passages judge-word-sense）。

- 名詞（ターゲット語と自然な名詞を含む）は、WordNet の語義の説明の名詞として、形容詞（自然な形容詞を
  含む）は、その語義の形容詞として使われているかを判定する。名前・題名・慣用句などの一部は不合格。
- 用例は、全ての出現が合格したときに合格とする。
- 語と分け方（学習用・評価用）の組ごとに、鍵の小さい候補から、合格した用例が必要な数に届くまで
  判定する。合格する割合を見込んで少しずつ問い合わせ、足りなければ続きを問い合わせる。
- 実行し直したとき（例: K を変えたとき）、前回と同じ指示文と LLM なら、前回判定した候補は前回の結果
  （答えが得られずに不合格にしたものを含む）を使い、問い合わせない。答えの得られなかった問い合わせは
  保存されないので、問い合わせ直すと結果が変わり、学習に使った用例とずれうるため。
- 出現の位置がない候補は、問い合わせずに不合格にする。
結果を data/passages/judgments.parquet に、組ごとの数を data/passages/judgment_summary.csv に書く。
"""

import argparse
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

from lexical_acquisition_mechanisms import passage_split, word_finalization
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.llm_client import (
    ANSWERS_PATH,
    LlmClient,
    LlmRequest,
    PromptTemplate,
    prompt_record_settings,
)
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
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


@dataclass
class JudgmentSummary:
    word: str
    role: str
    split: str
    # 合格する必要がある数と、判定した数・合格した数
    needed: int
    judged: int
    passed: int


def needed_passing_passages(role: str, split: str, settings: Settings) -> int:
    """語の役割と分け方ごとの、合格する必要がある用例の数（ターゲット語と自然な語は評価用だけ）。"""
    if split == "training":
        if role == "noun":
            return max(settings.unknown_word_acquisition.exposure_levels)
        if role == "adjective":
            return settings.sense_addition.passages_per_added_sense
        return 0
    if role.startswith("natural-"):
        return settings.evaluation.passages_per_natural_word
    return settings.passages.evaluation_passages_per_word


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
            (JUDGMENTS_PATH, SUMMARY_PATH),
        ),
        arguments.recompute,
    )


def build_judgments(
    settings: Settings,
    prompts: Mapping[str, PromptTemplate],
    client: LlmClient,
    words_path: Path,
    split_path: Path,
    output_paths: tuple[Path, Path],
) -> None:
    judgments_path, summary_path = output_paths
    words = {word.word: word for word in read_rows(words_path, SelectedWord)}
    tokenizer = WordTokenizer.from_model(next(iter(settings.models.values())))
    # 候補は多い（数百万件）ので、判定する候補だけを、その都度 Python の値にする
    table = parquet.read_table(split_path)
    remaining = _indices_by_group(table)
    needed = {
        group: needed_passing_passages(words[group[0]].role, group[1], settings)
        for group in remaining
    }
    judged: Counter[tuple[str, str]] = Counter()
    passed: Counter[tuple[str, str]] = Counter()
    judged_indices: list[int] = []
    accepted_values: list[bool] = []
    reason_lists: list[list[str]] = []
    signature = json.dumps(
        prompt_record_settings(settings.llm, list(prompts.values())), sort_keys=True
    )
    previous = _previous_judgments(judgments_path, signature)
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
    results = table.take(judged_indices)
    results = results.append_column("passed", arrow.array(accepted_values, arrow.bool_()))
    reasons = arrow.array(reason_lists, arrow.list_(arrow.string()))
    results = results.append_column("reasons", reasons)
    metadata = {**(results.schema.metadata or {}), JUDGMENT_SETTINGS_KEY: signature.encode()}
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


def _candidate_key(candidate: Mapping[str, Any]) -> tuple[str, int, int, int]:
    return candidate["word"], candidate["shard"], candidate["row"], candidate["start"]


def _previous_judgments(
    path: Path, signature: str
) -> dict[tuple[str, int, int, int], tuple[bool, list[str]]]:
    """前回の判定の結果（候補ごとの合否と理由）。指示文か LLM が違えば、使わない（空）。"""
    if not path.exists():
        return {}
    table = parquet.read_table(path)
    if (table.schema.metadata or {}).get(JUDGMENT_SETTINGS_KEY) != signature.encode():
        return {}
    rows = table.select(["word", "shard", "row", "start", "passed", "reasons"]).to_pylist()
    return {_candidate_key(row): (row["passed"], row["reasons"]) for row in rows}


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
