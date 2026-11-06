"""最小対の候補を作る処理のテスト。LLM は、決まった答えを返す部品で置き換える。"""

import argparse
import json
from pathlib import Path

import httpx
import pytest

from lexical_acquisition_mechanisms import minimal_pairs
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.llm_client import LlmClient, PromptTemplate
from lexical_acquisition_mechanisms.minimal_pairs import (
    PROMPT,
    MinimalPairCandidate,
    build_minimal_pair_candidates,
    candidates_per_sense_pair,
    swapped_cue_sentences,
)
from lexical_acquisition_mechanisms.sense_addition_assignments import SenseAdditionAssignment
from lexical_acquisition_mechanisms.settings import ModelSource, load_settings
from lexical_acquisition_mechanisms.word_selection import SelectedWord

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PAIRS = [
    {"sentence": "She cooked the ___ for dinner.", "first_cue": "cooked", "second_cue": "parked"},
    # 同じ候補、片方の文だけが同じ候補（2つ）、空欄がない、手がかりが空欄の後、手がかりが文にない、
    # 語の数が違う
    {"sentence": "She cooked the ___ for dinner.", "first_cue": "cooked", "second_cue": "parked"},
    {"sentence": "She cooked the ___ for dinner.", "first_cue": "cooked", "second_cue": "washed"},
    {"sentence": "She boiled the ___ for dinner.", "first_cue": "boiled", "second_cue": "parked"},
    {"sentence": "She cooked dinner.", "first_cue": "cooked", "second_cue": "parked"},
    {"sentence": "The ___ was cooked.", "first_cue": "cooked", "second_cue": "parked"},
    {"sentence": "She ate the ___.", "first_cue": "cooked", "second_cue": "parked"},
    {"sentence": "She cooked the ___ .", "first_cue": "cooked", "second_cue": "carefully parked"},
    {"sentence": "He ate the ___ quickly.", "first_cue": "ate", "second_cue": "drove"},
    # 空欄の外に donor（pasta）・ターゲット語の別の形（cars）がある、空欄の下線が4本、空欄の後に
    # 英字が続く
    {"sentence": "She cooked the ___ with pasta.", "first_cue": "cooked", "second_cue": "parked"},
    {"sentence": "She cooked the ___ near cars.", "first_cue": "cooked", "second_cue": "parked"},
    {"sentence": "She baked the ____ today.", "first_cue": "baked", "second_cue": "washed"},
    {"sentence": "She baked the ___s today.", "first_cue": "baked", "second_cue": "washed"},
]


def answer_like_the_llm(request: httpx.Request) -> httpx.Response:
    content = json.loads(request.content)["messages"][0]["content"]
    pairs = (
        PAIRS
        if "an adjective" not in content
        else [{"sentence": "It was so ___.", "first_cue": "so", "second_cue": "the"}]
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


def test_sense_pairs_short_of_candidates_are_asked_again_with_other_numbers(
    tmp_path: Path, tiny_model_source: ModelSource
) -> None:
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    settings = settings.model_copy(update={"models": {"tiny": tiny_model_source}})
    words = [
        SelectedWord("car", "target", category="vehicle", token_count=1, definition="a vehicle"),
        SelectedWord("pasta", "noun", category="food", definition="a dish"),
    ]
    write_rows(tmp_path / "words.csv", SelectedWord, words)
    assignment = SenseAdditionAssignment(
        "other-category-4", "car", "vehicle", 1, "pasta", "noun", "food", 1
    )
    write_rows(tmp_path / "assignments.csv", SenseAdditionAssignment, [assignment])
    number = candidates_per_sense_pair(settings)
    asked_numbers = []

    def short_at_first(request: httpx.Request) -> httpx.Response:
        content = json.loads(request.content)["messages"][0]["content"]
        asked = int(content.rsplit("Write ", 1)[1].split()[0])
        asked_numbers.append(asked)
        # 最初は、2つの手がかりが同じ語の1組だけを返す（使えない）
        pairs = (
            [{"sentence": "She moved the ___.", "first_cue": "moved", "second_cue": "moved"}]
            if asked == number
            else PAIRS
        )
        choice = {"message": {"content": json.dumps({"pairs": pairs})}, "finish_reason": "stop"}
        return httpx.Response(200, json={"choices": [choice]})

    client = LlmClient(
        settings.llm,
        tmp_path / "answers.sqlite3",
        rerun_from="evaluation-data generate-minimal-pairs",
        transport=httpx.MockTransport(short_at_first),
    )

    build_minimal_pair_candidates(
        settings,
        PromptTemplate.load(PROMPT, REPOSITORY_ROOT / "llm_prompts"),
        client,
        tmp_path / "words.csv",
        tmp_path / "assignments.csv",
        tmp_path / "candidates.csv",
    )

    # 使える候補が最小対の数に届かない間、頼む数を変えて3回まで問い直し、重ならない候補だけを足す
    step = minimal_pairs.ASK_AGAIN_STEP
    assert asked_numbers == [number, number + step, number + 2 * step, number + 3 * step]
    candidates = read_rows(tmp_path / "candidates.csv", MinimalPairCandidate)
    assert [(row.donor_cue, row.target_cue) for row in candidates] == [
        ("cooked", "parked"),
        ("ate", "drove"),
    ]


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
        pairs = [{"sentence": "It was so nice.", "first_cue": "so", "second_cue": "the"}]
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


def test_the_target_sentence_swaps_only_the_cue_word() -> None:
    assert swapped_cue_sentences("She cooked the ___ for dinner.", "cooked", "parked") == (
        "She cooked the ___ for dinner.",
        "She parked the ___ for dinner.",
    )
    # 記号の付いた手がかりも置き換える。手がかりが空欄の後・2回・文にないときは使わない
    assert swapped_cue_sentences("Quickly, ___ arrived.", "Quickly", "Slowly") == (
        "Quickly, ___ arrived.",
        "Slowly, ___ arrived.",
    )
    assert swapped_cue_sentences("The ___ was so big.", "so", "the") is None
    assert swapped_cue_sentences("It was so so ___.", "so", "the") is None
    assert swapped_cue_sentences("It was ___.", "so", "the") is None


def test_record_has_the_cue_instructions(monkeypatch: pytest.MonkeyPatch) -> None:
    # 指示文の .md を変えずに、手がかりの指示・答えの形・空欄や記号の決まりだけを直しても、作り直す
    steps = []
    monkeypatch.setattr(
        minimal_pairs, "run_unless_unchanged", lambda step, compute, recompute: steps.append(step)
    )
    minimal_pairs.generate_minimal_pairs(
        load_settings(REPOSITORY_ROOT / "settings.yaml"), argparse.Namespace(recompute=False)
    )
    assert steps[0].settings["cue_instructions"] == minimal_pairs.CUE_INSTRUCTIONS
    assert steps[0].settings["cue_punctuation"] == minimal_pairs.PUNCTUATION
    assert steps[0].settings["schema"] == minimal_pairs.SCHEMA
    assert steps[0].settings["blank_pattern"] == minimal_pairs.BLANK_PATTERN.pattern
    assert steps[0].settings["maximum_tokens_per_pair"] == minimal_pairs.MAXIMUM_TOKENS_PER_PAIR
