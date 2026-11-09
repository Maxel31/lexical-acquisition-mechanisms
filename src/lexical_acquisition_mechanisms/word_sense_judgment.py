"""用例の候補の中で、語が意図した語義で使われているかを、LLM が出現ごとに判定する
（passages judge-word-sense）。

- 名詞（ターゲット語と自然な名詞を含む）は、WordNet の語義の説明の名詞として、形容詞（自然な形容詞を
  含む）は、その語義の形容詞として使われているかを判定する。名前・題名・慣用句などの一部は不合格。
- 用例は、全ての出現が合格したときに合格とする。
- 語と分け方（学習用・評価用）の組ごとに、鍵の小さい候補から、合格した用例が必要な数に届くまで
  判定する。合格する割合を見込んで少しずつ問い合わせ、足りなければ続きを問い合わせる。
結果を data/passages/judgments.parquet に、組ごとの数を data/passages/judgment_summary.csv に書く。
"""

import argparse
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    """語の役割と分け方ごとの、合格する必要がある用例の数。"""
    if role.startswith("natural-"):
        return settings.evaluation.sentences_per_natural_word
    if split == "evaluation":
        return settings.passages.evaluation_passages_per_word
    if role == "noun":
        return max(settings.unknown_word_acquisition.exposure_levels)
    if role == "adjective":
        return settings.sense_addition.passages_per_added_sense
    return 0


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
        "evaluation": settings.evaluation.model_dump(include={"sentences_per_natural_word"}),
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
    remaining: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for candidate in sorted(parquet.read_table(split_path).to_pylist(), key=lambda c: c["key"]):
        remaining.setdefault((candidate["word"], candidate["split"]), []).append(candidate)
    needed = {
        group: needed_passing_passages(words[group[0]].role, group[1], settings)
        for group in remaining
    }
    judged: Counter[tuple[str, str]] = Counter()
    passed: Counter[tuple[str, str]] = Counter()
    results: list[dict[str, Any]] = []
    while batch := _next_batch(remaining, needed, judged, passed, words):
        requests = []
        owners = []
        for index, (_, candidate) in enumerate(batch):
            word = words[candidate["word"]]
            for position in candidate["occurrence_positions"]:
                requests.append(_request(word, candidate, position, prompts, tokenizer))
                owners.append(index)
        verdicts: list[list[dict[str, Any] | None]] = [[] for _ in batch]
        for owner, answer in zip(owners, client.ask_all(requests), strict=True):
            verdicts[owner].append(answer)
        for (group, candidate), answers in zip(batch, verdicts, strict=True):
            accepted = all(answer is not None and answer["intended"] for answer in answers)
            judged[group] += 1
            passed[group] += accepted
            reasons = [answer["reason"] if answer else "判定できなかった" for answer in answers]
            results.append({**candidate, "passed": accepted, "reasons": reasons})
    schema = parquet.read_schema(split_path)
    schema = schema.append(arrow.field("passed", arrow.bool_()))
    schema = schema.append(arrow.field("reasons", arrow.list_(arrow.string())))
    parquet.write_table(arrow.Table.from_pylist(results, schema=schema), judgments_path)
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


def _next_batch(
    remaining: dict[tuple[str, str], list[dict[str, Any]]],
    needed: Mapping[tuple[str, str], int],
    judged: Mapping[tuple[str, str], int],
    passed: Mapping[tuple[str, str], int],
    words: Mapping[str, SelectedWord],
) -> list[tuple[tuple[str, str], dict[str, Any]]]:
    """組ごとに、合格の不足を埋めるのに見込まれる数の候補を、鍵の小さい順に取り出す。"""
    batch = []
    for group, candidates in remaining.items():
        deficit = needed[group] - passed[group]
        if deficit <= 0 or not candidates:
            continue
        rate = passed[group] / judged[group] if judged[group] else words[group[0]].usage_share
        count = max(
            MINIMUM_REQUESTS, math.ceil(deficit * REQUEST_MARGIN / max(rate, MINIMUM_PASS_RATE))
        )
        batch += [(group, candidate) for candidate in candidates[:count]]
        del candidates[:count]
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
