"""2つ目の語義の追加の評価データを作る処理のテスト。LLM は決まった答えを返す部品で、モデルの
手がかりの効果は決まった値で置き換える。"""

import json
import re
from pathlib import Path

import httpx
import numpy as np
import pyarrow.parquet as parquet
import pytest

from lexical_acquisition_mechanisms import minimal_pair_judgment, minimal_pairs
from lexical_acquisition_mechanisms import second_sense_evaluation_data as evaluation_data
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.llm_client import LlmClient, PromptTemplate
from lexical_acquisition_mechanisms.minimal_pairs import MinimalPairCandidate, SensePair
from lexical_acquisition_mechanisms.pretrained_model_checks import CheckedMinimalPair
from lexical_acquisition_mechanisms.second_sense_assignments import SecondSenseAssignment
from lexical_acquisition_mechanisms.second_sense_evaluation_data import (
    SecondSensePaths,
    second_sense_pairs,
    write_second_sense_minimal_pairs,
    write_second_sense_questions,
)
from lexical_acquisition_mechanisms.settings import ModelSource, load_settings
from lexical_acquisition_mechanisms.task_questions import TaskExample
from lexical_acquisition_mechanisms.word_selection import SelectedWord

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORDS = [
    SelectedWord("cola", "noun", category="food", token_count=1, definition="a soft drink"),
    SelectedWord("car", "noun", category="vehicle", token_count=1, definition="a vehicle"),
    SelectedWord("tea", "noun", category="food", token_count=1, definition="a hot drink"),
    SelectedWord("flaid", "unknown", token_count=2),
]


def assignment(run: str, word: str, second: str, kind: str, exposure: int) -> SecondSenseAssignment:
    return SecondSenseAssignment(run, word, 2, "cola", "food", 3000, kind, second, "", exposure)


def write_inputs(directory: Path) -> SecondSensePaths:
    paths = SecondSensePaths(
        words=directory / "words.csv",
        assignments=directory / "assignments.csv",
        examples=directory / "examples.csv",
        candidates=directory / "out" / "candidates.csv",
        judged=directory / "out" / "judged.csv",
        judgments=directory / "out" / "judgments.csv",
        checked=directory / "out" / "checked.csv",
        questions=directory / "out" / "questions.parquet",
    )
    write_rows(paths.words, SelectedWord, WORDS)
    write_rows(
        paths.assignments,
        SecondSenseAssignment,
        [
            assignment("run-1", "flaid", "car", "other-category", 100),
            # 同じ組が別の run にもある
            assignment("run-2", "blick", "car", "other-category", 1000),
            assignment("run-2", "dax", "tea", "same-category", 10),
        ],
    )
    examples = [
        TaskExample("sense", f"word{number}", f"I saw the {{word}} {number}.", "a|b", "a")
        for number in range(5)
    ]
    write_rows(paths.examples, TaskExample, examples)
    return paths


def test_sense_pairs_are_the_pairs_of_the_first_and_second_nouns(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path)

    pairs = second_sense_pairs(paths.words, paths.assignments)

    assert pairs == [
        SensePair("cola", "food", 1, "car", "noun", "other-category"),
        SensePair("cola", "food", 1, "tea", "noun", "same-category"),
    ]


def answer_like_the_llm(request: httpx.Request) -> httpx.Response:
    content = json.loads(request.content)["messages"][0]["content"]
    if content.startswith("We are checking"):
        # 判定: car の組は 2 文とも合格、tea の組は Y の文が不合格
        fits = re.search(r"\btea\b", content) is None
        answer = {"reason": "", "sentence_1_fits": fits, "sentence_2_fits": True}
    else:
        cue = "brewed" if re.search(r"\btea\b", content) else "drove"
        answer = {
            "pairs": [{"sentence": f"We {cue} the ___.", "first_cue": cue, "second_cue": "drank"}]
        }
    choice = {"message": {"content": json.dumps(answer)}, "finish_reason": "stop"}
    return httpx.Response(200, json={"choices": [choice]})


def test_minimal_pairs_are_made_for_each_pair_and_judged(
    tmp_path: Path, tiny_model_source: ModelSource
) -> None:
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    settings = settings.model_copy(update={"models": {"tiny": tiny_model_source}})
    paths = write_inputs(tmp_path)
    prompts = tuple(
        PromptTemplate.load(name, REPOSITORY_ROOT / "llm_prompts")
        for name in (minimal_pairs.PROMPT, minimal_pair_judgment.PROMPT)
    )
    client = LlmClient(
        settings.llm,
        tmp_path / "answers.sqlite3",
        rerun_from=evaluation_data.GENERATE_COMMAND,
        transport=httpx.MockTransport(answer_like_the_llm),
    )

    write_second_sense_minimal_pairs(settings, prompts, client, paths)

    candidates = read_rows(paths.candidates, MinimalPairCandidate)
    assert [
        (row.target, row.donor, row.kind, row.donor_cue, row.target_cue) for row in candidates
    ] == [
        ("cola", "car", "other-category", "drove", "drank"),
        ("cola", "tea", "same-category", "brewed", "drank"),
    ]
    judged = read_rows(paths.judged, MinimalPairCandidate)
    assert [(row.target, row.donor) for row in judged] == [("cola", "car")]


def candidate(donor: str, cue: str) -> MinimalPairCandidate:
    kind = "other-category" if donor == "car" else "same-category"
    return MinimalPairCandidate(
        "cola",
        "food",
        1,
        donor,
        "noun",
        kind,
        f"We {cue} the ___.",
        "We drank the ___.",
        cue,
        "drank",
    )


def test_questions_use_the_checked_pairs_and_the_same_examples_in_both_sentences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    evaluation = settings.evaluation.model_copy(
        update={"minimal_pairs_per_sense_pair": 2, "number_of_task_examples": 2}
    )
    settings = settings.model_copy(
        update={"evaluation": evaluation, "models": {"a": ModelSource(repository="a")}}
    )
    paths = write_inputs(tmp_path)
    candidates = [
        candidate("car", "drove"),
        candidate("car", "parked"),  # 手がかりが逆の向きに動かす
        candidate("car", "washed"),
        candidate("car", "fixed"),  # 組ごとの数を超える
        candidate("tea", "brewed"),
    ]
    paths.judged.parent.mkdir(parents=True)
    write_rows(paths.judged, MinimalPairCandidate, candidates)
    effects = np.array([[1.0, 2.0], [1.0, -0.5], [0.5, 0.5], [1.0, 1.0], [2.0, 1.0]])
    monkeypatch.setattr(evaluation_data, "cue_effects", lambda model, rows, words: effects)

    write_second_sense_questions(settings, lambda source: None, paths)

    checked = read_rows(paths.checked, CheckedMinimalPair)
    assert [(row.donor, row.donor_cue, row.target_cue_effect) for row in checked] == [
        ("car", "drove", 2.0),
        ("car", "washed", 0.5),
        ("tea", "brewed", 1.0),
    ]
    questions = parquet.read_table(paths.questions).to_pylist()
    assert len(questions) == 6
    for first, second in zip(questions[0::2], questions[1::2], strict=True):
        assert (first["cue"], second["cue"]) == ("first", "second")
        assert first["answer"] == first["first_noun"] == "cola"
        assert second["answer"] == second["second_noun"]
        assert first["choices"] == second["choices"]
        assert sorted(first["choices"]) == sorted(["cola", second["second_noun"]])
        # 2つの問題の違いは、最後の問題の文だけ（例も選択肢の順番も同じ）
        head, last = first["prompt_template"].rsplit("Sentence: ", 1)
        assert second["prompt_template"].rsplit("Sentence: ", 1)[0] == head
        assert last.startswith("We drank the {word}.")


def test_questions_stop_when_a_pair_has_no_checked_minimal_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    settings = settings.model_copy(update={"models": {"a": ModelSource(repository="a")}})
    paths = write_inputs(tmp_path)
    paths.judged.parent.mkdir(parents=True)
    # tea の組には、LLM の判定に合格した候補がない
    write_rows(paths.judged, MinimalPairCandidate, [candidate("car", "drove")])
    monkeypatch.setattr(
        evaluation_data, "cue_effects", lambda model, rows, words: np.ones((len(rows), 2))
    )

    with pytest.raises(CommandStopped, match="cola/tea"):
        write_second_sense_questions(settings, lambda source: None, paths)
