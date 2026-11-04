"""用例の語義の判定のテスト。LLM は、決まった答えを返す部品で置き換える。"""

import json
from pathlib import Path

import httpx
import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

from lexical_acquisition_mechanisms import word_sense_judgment
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.llm_client import LlmClient, PromptTemplate
from lexical_acquisition_mechanisms.passage_collection import CANDIDATE_SCHEMA
from lexical_acquisition_mechanisms.settings import ModelSource, Settings, load_settings
from lexical_acquisition_mechanisms.word_selection import SelectedWord
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
    evaluation = settings.evaluation.model_copy(update={"sentences_per_natural_word": 1})
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


def test_candidates_are_judged_until_enough_pass(tmp_path: Path, settings: Settings) -> None:
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
        settings, prompts, client, tmp_path / "words.csv", tmp_path / "split.parquet", outputs
    )

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
