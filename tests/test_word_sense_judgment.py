"""用例の語義の判定のテスト。LLM は、決まった答えを返す部品で置き換える。"""

import json
from pathlib import Path

import httpx
import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

from lexical_acquisition_mechanisms import word_sense_judgment
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.llm_client import LlmClient, PromptTemplate
from lexical_acquisition_mechanisms.passage_collection import CANDIDATE_SCHEMA, CollectionSummary
from lexical_acquisition_mechanisms.settings import ModelSource, Settings, load_settings
from lexical_acquisition_mechanisms.word_selection import ExcludedWord, SelectedWord
from lexical_acquisition_mechanisms.word_sense_judgment import (
    ADJECTIVE_PROMPT,
    NOUN_PROMPT,
    JudgmentSummary,
    build_judgments,
)
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# 学習用の pasta の候補（鍵の順）。"[[pasta]] brand" と判定させる出現は不合格
PASTA_TEXTS = [
    "We ate pasta tonight.",
    "A pasta brand was sold.",
    "The pasta brand is new.",
    "Fresh pasta is good.",
    "Some pasta for lunch.",
    "My pasta brand and our pasta dish.",
    "Hot pasta again.",
    "Cold pasta later.",
    "More pasta please.",
]
asked_prompts: list[str] = []


def answer_like_the_llm(request: httpx.Request) -> httpx.Response:
    content = json.loads(request.content)["messages"][0]["content"]
    asked_prompts.append("adjective" if "(an adjective)" in content else "noun")
    passage = content.split("Passage:")[1]
    answer = {"intended": "]] brand" not in passage, "reason": "stub"}
    choice = {"message": {"content": json.dumps(answer)}, "finish_reason": "stop"}
    return httpx.Response(200, json={"choices": [choice]})


@pytest.fixture
def settings(tiny_model_source: ModelSource, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """名詞は、学習用 3 件と評価用 1 件、自然な語は 1 件の合格が必要。1回に1件から問い合わせる。"""
    monkeypatch.setattr(word_sense_judgment, "MINIMUM_REQUESTS", 1)
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    passages = settings.passages.model_copy(update={"evaluation_passages_per_word": 1})
    exposure = settings.unknown_word_acquisition.model_copy(update={"exposure_levels": [0, 3]})
    evaluation = settings.evaluation.model_copy(update={"passages_per_natural_word": 1})
    return settings.model_copy(
        update={
            "models": {"tiny": tiny_model_source},
            "passages": passages,
            "unknown_word_acquisition": exposure,
            "evaluation": evaluation,
        }
    )


def candidate(
    tokenizer: WordTokenizer, word: str, split: str, key: float, text: str
) -> dict[str, object]:
    token_ids = tokenizer.encode([text])
    word_id = tokenizer.word_token_ids(word)[0]
    positions = [index for index, token in enumerate(token_ids) if token == word_id]
    return {
        "word": word,
        "role": "",
        "shard": 0,
        "row": int(key * 100),
        "start": 0,
        "key": key,
        "token_ids": token_ids,
        "occurrence_positions": positions,
        "split": split,
    }


def shortage_paths(tmp_path: Path, words: list[SelectedWord]) -> tuple[Path, Path]:
    """collect の語ごとの数（どの語も、Pile の出現を全て残した）と、外す語の CSV のパス。"""
    summary = [CollectionSummary(word.word, word.role, 0, 0, 0, "", False) for word in words]
    write_rows(tmp_path / "collection_summary.csv", CollectionSummary, summary)
    return tmp_path / "collection_summary.csv", tmp_path / "excluded_words.csv"


def test_candidates_are_judged_until_enough_pass(
    tmp_path: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    tokenizer = WordTokenizer.from_model(settings.models["tiny"])
    words = [
        SelectedWord("pasta", "noun", token_count=1, definition="a dish", usage_share=0.9),
        SelectedWord("quiet", "natural-adjective", token_count=1, definition="calm", usage_share=1),
    ]
    write_rows(tmp_path / "words.csv", SelectedWord, words)
    candidates = [
        candidate(tokenizer, "pasta", "training", index / 10, text)
        for index, text in enumerate(PASTA_TEXTS)
    ]
    candidates.append(candidate(tokenizer, "pasta", "evaluation", 0.95, "Good pasta here."))
    candidates.append(candidate(tokenizer, "quiet", "evaluation", 0.5, "It was quiet here."))
    schema = CANDIDATE_SCHEMA.append(arrow.field("split", arrow.string()))
    table = arrow.Table.from_pylist(candidates, schema=schema)
    parquet.write_table(table, tmp_path / "split.parquet")
    prompts = {
        part_of_speech: PromptTemplate.load(name, REPOSITORY_ROOT / "llm_prompts")
        for part_of_speech, name in [("noun", NOUN_PROMPT), ("adjective", ADJECTIVE_PROMPT)]
    }
    client = LlmClient(
        settings.llm,
        tmp_path / "answers.sqlite3",
        rerun_from="passages judge-word-sense",
        transport=httpx.MockTransport(answer_like_the_llm),
    )
    outputs = (tmp_path / "judgments.parquet", tmp_path / "summary.csv")

    build_judgments(
        settings,
        prompts,
        client,
        tmp_path / "words.csv",
        tmp_path / "split.parquet",
        shortage_paths(tmp_path, words),
        outputs,
    )

    # 評価用は、合格する見込み（候補の数 × usage_share）が必要な数の 2.5 倍に届かないので表示する。
    # 学習用（9件 × 0.9 = 8.1 件。必要な数 3 件の 2.5 倍は 7.5 件）は表示しない
    out = capsys.readouterr().out
    assert "pasta（evaluation 1 / 1 件）, quiet（evaluation 1 / 1 件）" in out
    assert "pasta（training" not in out
    summary = {(row.word, row.split): row for row in read_rows(outputs[1], JudgmentSummary)}
    # 1回目は 3 × 1.2 ÷ 0.9 → 4 件（合格 2）、2回目は 1 × 1.2 ÷ 0.5 → 3 件
    # （"pasta brand" と "pasta dish" の2つの出現がある6番目は、1つが不合格なので不合格）
    training = summary[("pasta", "training")]
    assert (training.needed, training.judged, training.passed) == (3, 7, 4)
    assert summary[("quiet", "evaluation")].passed == 1
    assert summary[("pasta", "evaluation")].passed == 1
    judgments = parquet.read_table(outputs[0]).to_pylist()
    assert {round(row["key"], 2) for row in judgments if row["split"] == "training"} == {
        round(index / 10, 2) for index in range(7)
    }
    sixth = next(row for row in judgments if row["word"] == "pasta" and round(row["key"], 2) == 0.5)
    assert (sixth["passed"], len(sixth["reasons"])) == (False, 2)
    assert asked_prompts.count("adjective") == 1


def test_rejudging_reuses_previous_results_with_the_same_prompts(
    tmp_path: Path, settings: Settings
) -> None:
    tokenizer = WordTokenizer.from_model(settings.models["tiny"])
    words = [SelectedWord("pasta", "noun", token_count=1, definition="a dish", usage_share=0.9)]
    write_rows(tmp_path / "words.csv", SelectedWord, words)
    candidates = [
        candidate(tokenizer, "pasta", "training", index / 10, text)
        for index, text in enumerate(PASTA_TEXTS)
    ]
    candidates.append(candidate(tokenizer, "pasta", "evaluation", 0.95, "Good pasta here."))
    schema = CANDIDATE_SCHEMA.append(arrow.field("split", arrow.string()))
    parquet.write_table(
        arrow.Table.from_pylist(candidates, schema=schema), tmp_path / "split.parquet"
    )
    prompts = {
        part_of_speech: PromptTemplate.load(name, REPOSITORY_ROOT / "llm_prompts")
        for part_of_speech, name in [("noun", NOUN_PROMPT), ("adjective", ADJECTIVE_PROMPT)]
    }
    outputs = (tmp_path / "judgments.parquet", tmp_path / "summary.csv")
    requested: list[str] = []

    def judge(answers_path: Path, unfinished: str = "", reuse: bool = True) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            passage = json.loads(request.content)["messages"][0]["content"].split("Passage:")[1]
            requested.append(passage)
            if unfinished and unfinished in passage:
                choice = {"message": {"content": "{"}, "finish_reason": "length"}
            else:
                answer = {"intended": "]] brand" not in passage, "reason": "stub"}
                choice = {"message": {"content": json.dumps(answer)}, "finish_reason": "stop"}
            return httpx.Response(200, json={"choices": [choice]})

        client = LlmClient(
            settings.llm,
            answers_path,
            rerun_from="passages judge-word-sense",
            transport=httpx.MockTransport(respond),
        )
        build_judgments(
            settings,
            prompts,
            client,
            tmp_path / "words.csv",
            tmp_path / "split.parquet",
            shortage_paths(tmp_path, words),
            outputs,
            reuse=reuse,
        )

    # 1回目: "We ate pasta" の答えが長さの上限で切れ、判定できずに不合格になる
    judge(tmp_path / "first.sqlite3", unfinished="We ate [[pasta]]")
    first = {row["key"]: row["passed"] for row in parquet.read_table(outputs[0]).to_pylist()}
    assert first[0.0] is False
    # 2回目（同じ指示文。答えの保存は空）: 前回判定した候補は問い合わせず、結果も同じ
    requested.clear()
    judge(tmp_path / "second.sqlite3")
    second = {row["key"]: row["passed"] for row in parquet.read_table(outputs[0]).to_pylist()}
    assert second == first
    assert requested == []
    # --recompute（reuse=False）なら、前回の結果は使わない
    judge(tmp_path / "recompute.sqlite3", reuse=False)
    assert requested
    # 同じ断片・行・始まりでも、窓の中身が変わった候補は、問い合わせ直す
    requested.clear()
    changed = list(candidates)
    changed[1] = candidate(tokenizer, "pasta", "training", 0.1, "A pasta brand was made.")
    parquet.write_table(arrow.Table.from_pylist(changed, schema=schema), tmp_path / "split.parquet")
    judge(tmp_path / "window.sqlite3")
    assert requested and all("was made" in passage for passage in requested)
    # 語の語義の説明が変わると、前回の結果は使わない
    requested.clear()
    judge(tmp_path / "same-definition.sqlite3")
    assert requested == []
    words[0] = SelectedWord(
        "pasta", "noun", token_count=1, definition="a food made from dough", usage_share=0.9
    )
    write_rows(tmp_path / "words.csv", SelectedWord, words)
    judge(tmp_path / "new-definition.sqlite3")
    assert requested
    # 指示文を直すと、前回の結果は使わない
    requested.clear()
    prompts["noun"] = PromptTemplate(prompts["noun"].name, prompts["noun"].text + "\n")
    judge(tmp_path / "third.sqlite3")
    assert requested


def test_too_few_candidates_stop_before_asking(tmp_path: Path, settings: Settings) -> None:
    tokenizer = WordTokenizer.from_model(settings.models["tiny"])
    words = [SelectedWord("pasta", "noun", token_count=1, definition="a dish", usage_share=0.9)]
    write_rows(tmp_path / "words.csv", SelectedWord, words)
    # 学習用は 3 件の合格が必要だが、候補が 2 件しかない。評価用の候補はない
    candidates = [
        candidate(tokenizer, "pasta", "training", index / 10, text)
        for index, text in enumerate(PASTA_TEXTS[:2])
    ]
    schema = CANDIDATE_SCHEMA.append(arrow.field("split", arrow.string()))
    parquet.write_table(
        arrow.Table.from_pylist(candidates, schema=schema), tmp_path / "split.parquet"
    )
    requested: list[httpx.Request] = []
    client = LlmClient(
        settings.llm,
        tmp_path / "answers.sqlite3",
        rerun_from="passages judge-word-sense",
        transport=httpx.MockTransport(lambda request: requested.append(request) or None),
    )
    prompts = {"noun": PromptTemplate.load(NOUN_PROMPT, REPOSITORY_ROOT / "llm_prompts")}

    with pytest.raises(
        CommandStopped, match="pasta（training 2 / 3 件）, pasta（evaluation 0 / 1 件）"
    ) as stopped:
        build_judgments(
            settings,
            prompts,
            client,
            tmp_path / "words.csv",
            tmp_path / "split.parquet",
            shortage_paths(tmp_path, words),
            (tmp_path / "judgments.parquet", tmp_path / "summary.csv"),
        )

    assert stopped.value.rerun_from == "words select"
    assert requested == []
    excluded = read_rows(tmp_path / "excluded_words.csv", ExcludedWord)
    assert [(row.word, row.role) for row in excluded] == [("pasta", "noun")]


def test_candidates_are_combined_into_one_chunk_with_large_lists() -> None:
    # 候補が多いと list のオフセットがあふれるので、large_list にして1つのかたまりにつなげる
    table = arrow.concat_tables(
        [
            arrow.table(
                {"word": ["pasta"], "token_ids": arrow.array([[1, 2]], arrow.list_(arrow.int32()))}
            )
        ]
        * 3
    )

    combined = word_sense_judgment.contiguous_with_large_lists(table)

    assert combined.column("token_ids").num_chunks == 1
    assert combined.schema.field("token_ids").type == arrow.large_list(arrow.int32())
    assert combined.cast(table.schema).equals(table.combine_chunks())
