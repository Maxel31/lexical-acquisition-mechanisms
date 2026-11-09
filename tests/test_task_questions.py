"""課題の問題を作る処理のテスト。LLM は、決まった答えを返す部品で置き換える。"""

import json
from pathlib import Path

import httpx
import pyarrow.parquet as parquet

from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.llm_client import LlmClient, PromptTemplate
from lexical_acquisition_mechanisms.natural_word_sentences import NaturalWordSentence
from lexical_acquisition_mechanisms.settings import ModelSource, load_settings
from lexical_acquisition_mechanisms.task_questions import (
    PROMPT,
    Paraphrase,
    TaskExample,
    write_task_questions,
)
from lexical_acquisition_mechanisms.word_selection import SelectedWord

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# 言い換えの候補（語そのもの・大文字・2語・管理対象の語は使わない）
PARAPHRASES = {
    "soup": ["soup", "Meal", "two words", "pasta", "broth"],
    "cat": ["kitty"],
    "dog": ["stallion", "puppy"],
    "car": ["automobile"],
}


def answer_like_the_llm(request: httpx.Request) -> httpx.Response:
    content = json.loads(request.content)["messages"][0]["content"]
    word = content.split('"')[1]
    answer = {"paraphrases": PARAPHRASES.get(word, ["thing"])}
    choice = {"message": {"content": json.dumps(answer)}, "finish_reason": "stop"}
    return httpx.Response(200, json={"choices": [choice]})


def test_examples_and_natural_word_questions(
    tmp_path: Path, tiny_model_source: ModelSource
) -> None:
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    evaluation = settings.evaluation.model_copy(update={"number_of_task_examples": 1})
    settings = settings.model_copy(
        update={"models": {"tiny": tiny_model_source}, "evaluation": evaluation}
    )
    natural = [
        ("soup", "food"),
        ("cat", "animal"),
        ("dog", "animal"),
        ("bread", "food"),
        ("horse", "animal"),
        ("cheese", "food"),
    ]
    words = [
        SelectedWord(
            word, "natural-noun", category=category, token_count=1, part_of_speech_probe=True
        )
        for word, category in natural
    ]
    words += [
        SelectedWord("quiet", "natural-adjective", token_count=1, part_of_speech_probe=True),
        SelectedWord("car", "target", category="vehicle", token_count=1, definition="a vehicle"),
        SelectedWord("pasta", "noun", category="food", token_count=1),
    ]
    write_rows(tmp_path / "words.csv", SelectedWord, words)
    sentences = [
        NaturalWordSentence(word, "", "", f"We saw the {word} {when}.", 11)
        for word in ("soup", "cat", "dog", "bread", "horse", "cheese", "quiet")
        for when in ("today", "again")
    ]
    write_rows(tmp_path / "sentences.csv", NaturalWordSentence, sentences)
    client = LlmClient(
        settings.llm,
        tmp_path / "answers.sqlite3",
        rerun_from="evaluation-data build-task-questions",
        transport=httpx.MockTransport(answer_like_the_llm),
    )
    outputs = (
        tmp_path / "paraphrases.csv",
        tmp_path / "examples.csv",
        tmp_path / "questions.parquet",
    )

    write_task_questions(
        settings,
        PromptTemplate.load(PROMPT, REPOSITORY_ROOT / "llm_prompts"),
        client,
        (tmp_path / "words.csv", tmp_path / "sentences.csv"),
        outputs,
    )

    paraphrases = {row.word: row.paraphrase for row in read_rows(outputs[0], Paraphrase)}
    assert paraphrases == {
        "soup": "broth",
        "cat": "kitty",
        "dog": "puppy",
        "bread": "thing",
        "horse": "thing",
        "cheese": "thing",
        "car": "automobile",
    }
    examples = read_rows(outputs[1], TaskExample)
    # 語ごとに最初の文を例にする（課題1・カテゴリは名詞6語、品詞は7語）
    assert sorted({(row.task, row.word) for row in examples}) == sorted(
        [("sense", word) for word, _ in natural]
        + [("category", word) for word, _ in natural]
        + [("part-of-speech", word) for word, _ in natural]
        + [("part-of-speech", "quiet")]
    )
    soup = next(row for row in examples if (row.task, row.word) == ("sense", "soup"))
    assert soup.answer == "broth" and "broth" in soup.choices.split("|")
    # 別のカテゴリの自然な名詞だけを、ほかの選択肢にする
    assert set(soup.choices.split("|")) - {"broth"} == {"cat", "dog", "horse"}
    questions = parquet.read_table(outputs[2]).to_pylist()
    assert len(questions) == 6 + 6 + 7
    for question in questions:
        assert question["answer"] in question["choices"]
        assert question["prompt_template"].count("Sentence:") == 2
        assert "{word}" in question["prompt_template"].split("\n\n")[-1]
