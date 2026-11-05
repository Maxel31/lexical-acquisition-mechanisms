"""最小対の候補を作る処理のテスト。LLM は、決まった答えを返す部品で置き換える。"""

import json
from pathlib import Path

import httpx
import pytest

from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.llm_client import LlmClient, PromptTemplate
from lexical_acquisition_mechanisms.minimal_pairs import (
    PROMPT,
    MinimalPairCandidate,
    build_minimal_pair_candidates,
    candidates_per_sense_pair,
)
from lexical_acquisition_mechanisms.sense_addition_assignments import SenseAdditionAssignment
from lexical_acquisition_mechanisms.settings import ModelSource, load_settings
from lexical_acquisition_mechanisms.word_selection import SelectedWord

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PAIRS = [
    {"first": "She cooked the ___ for dinner.", "second": "She parked the ___ for dinner."},
    # 同じ候補、片方の文だけが同じ候補（2つ）、空欄がない、手がかりが空欄の後、語の数が違う
    {"first": "She cooked the ___ for dinner.", "second": "She parked the ___ for dinner."},
    {"first": "She cooked the ___ for dinner.", "second": "She washed the ___ for dinner."},
    {"first": "She boiled the ___ for dinner.", "second": "She parked the ___ for dinner."},
    {"first": "She cooked dinner.", "second": "She parked dinner."},
    {"first": "The ___ was cooked.", "second": "The ___ was parked."},
    {"first": "She cooked the ___ .", "second": "She carefully parked the ___ ."},
    {"first": "He ate the ___ quickly.", "second": "He drove the ___ quickly."},
    # 空欄の外に donor（pasta）・ターゲット語の別の形（cars）がある、空欄の下線が4本
    {"first": "She cooked the ___ with pasta.", "second": "She parked the ___ with pasta."},
    {"first": "She cooked the ___ near cars.", "second": "She parked the ___ near cars."},
    {"first": "She baked the ____ today.", "second": "She washed the ____ today."},
]


def answer_like_the_llm(request: httpx.Request) -> httpx.Response:
    content = json.loads(request.content)["messages"][0]["content"]
    pairs = (
        PAIRS
        if "an adjective" not in content
        else [{"first": "It was so ___.", "second": "It was the ___."}]
    )
    choice = {"message": {"content": json.dumps({"pairs": pairs})}, "finish_reason": "stop"}
    return httpx.Response(200, json={"choices": [choice]})


def test_valid_pairs_are_kept(tmp_path: Path, tiny_model_source: ModelSource) -> None:
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    settings = settings.model_copy(update={"models": {"tiny": tiny_model_source}})
    words = [
        SelectedWord("car", "target", category="vehicle", token_count=1, definition="a vehicle"),
        SelectedWord("pasta", "noun", category="food", definition="a dish"),
        SelectedWord("quiet", "adjective", definition="calm"),
    ]
    write_rows(tmp_path / "words.csv", SelectedWord, words)
    assignments = [
        SenseAdditionAssignment(
            "other-category-4", "car", "vehicle", 1, "pasta", "noun", "food", 1
        ),
        SenseAdditionAssignment(
            "other-category-1", "car", "vehicle", 1, "pasta", "noun", "food", 1
        ),
        SenseAdditionAssignment("adjective-4", "car", "vehicle", 1, "quiet", "adjective", "", 1),
    ]
    write_rows(tmp_path / "assignments.csv", SenseAdditionAssignment, assignments)
    client = LlmClient(
        settings.llm,
        tmp_path / "answers.sqlite3",
        rerun_from="evaluation-data generate-minimal-pairs",
        transport=httpx.MockTransport(answer_like_the_llm),
    )

    build_minimal_pair_candidates(
        settings,
        PromptTemplate.load(PROMPT, REPOSITORY_ROOT / "llm_prompts"),
        client,
        tmp_path / "words.csv",
        tmp_path / "assignments.csv",
        tmp_path / "candidates.csv",
    )

    candidates = read_rows(tmp_path / "candidates.csv", MinimalPairCandidate)
    nouns = [row for row in candidates if row.donor == "pasta"]
    assert [(row.donor_cue, row.target_cue) for row in nouns] == [
        ("cooked", "parked"),
        ("ate", "drove"),
    ]
    assert {row.kind for row in nouns} == {"other-category"}
    (adjective,) = [row for row in candidates if row.donor == "quiet"]
    assert (adjective.kind, adjective.donor_cue, adjective.target_cue) == ("adjective", "so", "the")


def test_missing_answers_stop_instead_of_dropping_sense_pairs(
    tmp_path: Path, tiny_model_source: ModelSource
) -> None:
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    settings = settings.model_copy(update={"models": {"tiny": tiny_model_source}})
    words = [
        SelectedWord("car", "target", category="vehicle", token_count=1, definition="a vehicle"),
        SelectedWord("quiet", "adjective", definition="calm"),
    ]
    write_rows(tmp_path / "words.csv", SelectedWord, words)
    assignment = SenseAdditionAssignment(
        "adjective-4", "car", "vehicle", 1, "quiet", "adjective", "", 1
    )
    write_rows(tmp_path / "assignments.csv", SenseAdditionAssignment, [assignment])

    def unfinished(request: httpx.Request) -> httpx.Response:
        # 長さの上限で切れた答え（形に合わない答えは保存せず、None になる）
        choice = {"message": {"content": '{"pairs": ['}, "finish_reason": "length"}
        return httpx.Response(200, json={"choices": [choice]})

    client = LlmClient(
        settings.llm,
        tmp_path / "answers.sqlite3",
        rerun_from="evaluation-data generate-minimal-pairs",
        transport=httpx.MockTransport(unfinished),
    )

    with pytest.raises(CommandStopped, match="最小対の答えが 1 件"):
        build_minimal_pair_candidates(
            settings,
            PromptTemplate.load(PROMPT, REPOSITORY_ROOT / "llm_prompts"),
            client,
            tmp_path / "words.csv",
            tmp_path / "assignments.csv",
            tmp_path / "candidates.csv",
        )


def test_candidates_follow_the_number_of_minimal_pairs() -> None:
    # 最小対の数を増やすと、LLM に作らせる候補も増える（README の手作業のとおりに作り直せる）
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    evaluation = settings.evaluation.model_copy(
        update={"minimal_pairs_per_sense_pair": 30, "minimal_pair_candidate_factor": 1.5}
    )

    assert candidates_per_sense_pair(settings.model_copy(update={"evaluation": evaluation})) == 45


def test_sense_pairs_without_usable_candidates_stop(
    tmp_path: Path, tiny_model_source: ModelSource
) -> None:
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    settings = settings.model_copy(update={"models": {"tiny": tiny_model_source}})
    words = [
        SelectedWord("car", "target", category="vehicle", token_count=1, definition="a vehicle"),
        SelectedWord("quiet", "adjective", definition="calm"),
    ]
    write_rows(tmp_path / "words.csv", SelectedWord, words)
    assignment = SenseAdditionAssignment(
        "adjective-4", "car", "vehicle", 1, "quiet", "adjective", "", 1
    )
    write_rows(tmp_path / "assignments.csv", SenseAdditionAssignment, [assignment])

    def without_blanks(request: httpx.Request) -> httpx.Response:
        # 空欄のない文の組しか返さない
        pairs = [{"first": "It was so nice.", "second": "It was the nice."}]
        choice = {"message": {"content": json.dumps({"pairs": pairs})}, "finish_reason": "stop"}
        return httpx.Response(200, json={"choices": [choice]})

    client = LlmClient(
        settings.llm,
        tmp_path / "answers.sqlite3",
        rerun_from="evaluation-data generate-minimal-pairs",
        transport=httpx.MockTransport(without_blanks),
    )

    with pytest.raises(CommandStopped, match="候補が1つもない語義の組があります（1 組") as stopped:
        build_minimal_pair_candidates(
            settings,
            PromptTemplate.load(PROMPT, REPOSITORY_ROOT / "llm_prompts"),
            client,
            tmp_path / "words.csv",
            tmp_path / "assignments.csv",
            tmp_path / "candidates.csv",
        )
    assert stopped.value.rerun_from == "evaluation-data generate-minimal-pairs"
