"""意味の近さの評定のテスト。LLM は、決まった答えを返す部品で置き換える。"""

import csv
import json
from pathlib import Path

import httpx

from lexical_acquisition_mechanisms.csv_tables import write_rows
from lexical_acquisition_mechanisms.llm_client import LlmClient, PromptTemplate
from lexical_acquisition_mechanisms.settings import load_settings
from lexical_acquisition_mechanisms.similarity_ratings import PROMPT, build_similarity_ratings
from lexical_acquisition_mechanisms.word_selection import SelectedWord

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORDS = [
    SelectedWord("pasta", "noun", category="food", definition="a dish of dough"),
    SelectedWord("hog", "noun", category="animal", definition="a domestic swine"),
    SelectedWord("bagel", "target", category="food", definition="a ring-shaped roll"),
    SelectedWord("quiet", "adjective", definition="free of noise"),
]


def answer_like_the_llm(request: httpx.Request) -> httpx.Response:
    """bagel と pasta の組だけ 5、ほかは 1 と答える。"""
    content = json.loads(request.content)["messages"][0]["content"]
    similarity = 5 if '"pasta"' in content else 1
    choice = {"message": {"content": json.dumps({"similarity": similarity})}}
    return httpx.Response(200, json={"choices": [choice | {"finish_reason": "stop"}]})


def test_every_target_and_noun_pair_is_rated(tmp_path: Path) -> None:
    selected_words_path = tmp_path / "selected_words.csv"
    write_rows(selected_words_path, SelectedWord, WORDS)
    client = LlmClient(
        load_settings(REPOSITORY_ROOT / "settings.yaml").llm,
        tmp_path / "answers.sqlite3",
        rerun_from="words rate-similarity-with-llm",
        transport=httpx.MockTransport(answer_like_the_llm),
    )
    output_path = tmp_path / "similarity_ratings.csv"

    build_similarity_ratings(
        PromptTemplate.load(PROMPT, REPOSITORY_ROOT / "llm_prompts"),
        client,
        selected_words_path,
        output_path,
    )

    with output_path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [(row["target"], row["noun"], row["llm_similarity"]) for row in rows] == [
        ("bagel", "pasta", "5"),
        ("bagel", "hog", "1"),
    ]
    assert rows[1]["noun_category"] == "animal"
