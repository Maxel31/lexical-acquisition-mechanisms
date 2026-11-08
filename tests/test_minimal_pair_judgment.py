"""最小対の候補を LLM が判定する処理のテスト。LLM は、決まった答えを返す部品で置き換える。"""

import json
from pathlib import Path

import httpx

from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.llm_client import LlmClient, PromptTemplate
from lexical_acquisition_mechanisms.minimal_pair_judgment import (
    PROMPT,
    MinimalPairJudgment,
    judgment_request,
    write_judged_minimal_pairs,
)
from lexical_acquisition_mechanisms.minimal_pairs import MinimalPairCandidate
from lexical_acquisition_mechanisms.settings import load_settings
from lexical_acquisition_mechanisms.word_selection import SelectedWord

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORDS = [
    SelectedWord("cola", "target", category="food", token_count=1, definition="a soft drink"),
    SelectedWord("meat", "noun", category="food", definition="the flesh of animals"),
    SelectedWord("arcane", "adjective", definition="known only to a few"),
]


def candidate(donor: str, donor_sentence: str, target_sentence: str) -> MinimalPairCandidate:
    role, kind = ("adjective", "adjective") if donor == "arcane" else ("noun", "same-category")
    cues = donor_sentence.split()[1], target_sentence.split()[1]
    return MinimalPairCandidate(
        "cola", "food", 1, donor, role, kind, donor_sentence, target_sentence, *cues
    )


CANDIDATES = [
    candidate("meat", "We fried the ___.", "We drank the ___."),
    # 手がかりがどちらの語にも合う（LLM は donor の文を不合格にする）
    candidate("meat", "We spotted the ___.", "We sipped the ___."),
    # 答えが得られない
    candidate("meat", "We grilled the ___.", "We poured the ___."),
    candidate("arcane", "It was so ___.", "It was the ___."),
]


def answer_like_the_llm(request: httpx.Request) -> httpx.Response:
    content = json.loads(request.content)["messages"][0]["content"]
    if "grilled" in content:
        # 長さの上限で切れた答え（形に合わない答えは保存せず、None になる）
        choice = {"message": {"content": '{"reason": "'}, "finish_reason": "length"}
        return httpx.Response(200, json={"choices": [choice]})
    answer = {
        "reason": "checked",
        # 指示文の例にも spotted があるので、候補の文で見分ける
        "sentence_1_fits": "We spotted" not in content,
        "sentence_2_fits": True,
    }
    choice = {"message": {"content": json.dumps(answer)}, "finish_reason": "stop"}
    return httpx.Response(200, json={"choices": [choice]})


def test_only_candidates_whose_two_sentences_pass_are_kept(tmp_path: Path) -> None:
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    write_rows(tmp_path / "words.csv", SelectedWord, WORDS)
    write_rows(tmp_path / "candidates.csv", MinimalPairCandidate, CANDIDATES)
    client = LlmClient(
        settings.llm,
        tmp_path / "answers.sqlite3",
        rerun_from="evaluation-data judge-minimal-pairs",
        transport=httpx.MockTransport(answer_like_the_llm),
    )

    write_judged_minimal_pairs(
        PromptTemplate.load(PROMPT, REPOSITORY_ROOT / "llm_prompts"),
        client,
        tmp_path / "words.csv",
        tmp_path / "candidates.csv",
        (tmp_path / "judged.csv", tmp_path / "judgments.csv"),
    )

    judged = read_rows(tmp_path / "judged.csv", MinimalPairCandidate)
    assert [row.donor_sentence for row in judged] == ["We fried the ___.", "It was so ___."]
    judgments = read_rows(tmp_path / "judgments.csv", MinimalPairJudgment)
    assert [
        (row.donor_sentence_fits, row.target_sentence_fits, row.passed) for row in judgments
    ] == [
        (True, True, True),
        (False, True, False),
        (False, False, False),
        (True, True, True),
    ]
    assert judgments[2].reason == ""


def test_request_fills_the_blank_with_each_word() -> None:
    words = {word.word: word for word in WORDS}
    prompt = PromptTemplate.load(PROMPT, REPOSITORY_ROOT / "llm_prompts")

    text = prompt.render(judgment_request(prompt, CANDIDATES[0], words).values)

    for sentence in [
        "Sentence 1 with word A: We fried the meat.",
        "Sentence 1 with word B: We fried the cola.",
        "Sentence 2 with word B: We drank the cola.",
        "Sentence 2 with word A: We drank the meat.",
    ]:
        assert sentence in text
    assert "Both words are nouns." in text
    adjective = prompt.render(judgment_request(prompt, CANDIDATES[3], words).values)
    assert "Word A is an adjective" in adjective
