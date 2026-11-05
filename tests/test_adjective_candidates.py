"""形容詞の候補を作る処理のテスト。

LLM と Pile のダウンロードは、決まった答えを返す部品で置き換える。
"""

import csv
import json
from functools import partial
from pathlib import Path

import httpx
import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

from lexical_acquisition_mechanisms import pile
from lexical_acquisition_mechanisms.adjective_candidates import (
    ASSOCIATION_PROMPT,
    LISTING_PROMPT,
    USAGE_PROMPT,
    build_adjective_candidates,
)
from lexical_acquisition_mechanisms.llm_client import LlmClient, PromptTemplate
from lexical_acquisition_mechanisms.pile_word_usages import sample_word_usages
from lexical_acquisition_mechanisms.settings import ModelSource, Settings, load_settings

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LISTED_ADJECTIVES = ["quiet", "delicious", "well-known", "pasta"]
TEXTS = ["It was quiet here.", "The quiet was broken.", "A delicious meal."]
FILE_NAMES = ["data/train-00000-of-00001.parquet"]


def write_local_shard(file_name: str, directory: Path) -> Path:
    directory.mkdir(parents=True)
    path = directory / "shard.parquet"
    parquet.write_table(arrow.table({"text": TEXTS}), path)
    return path


def answer_like_the_llm(request: httpx.Request) -> httpx.Response:
    """挙げる問いには LISTED_ADJECTIVES、使われ方には「The [[」で始まる見本だけ名詞と答える。

    結びつきの評定は、delicious だけ food を 6、ほかはすべて 1 と答える。
    """
    content = json.loads(request.content)["messages"][0]["content"]
    if content.startswith("You are helping to build word lists"):
        answer = {"adjectives": [{"word": word, "meaning": "x"} for word in LISTED_ADJECTIVES]}
    elif content.startswith("How strongly"):
        word = content.split('"')[1]
        names = ["food", "animal", "vehicle", "clothing", "furniture", "building"]
        answer = {name: 1 for name in names} | ({"food": 6} if word == "delicious" else {})
    else:
        answer = {"adjective": "The [[" not in content, "reason": "stub"}
    choice = {"message": {"content": json.dumps(answer)}, "finish_reason": "stop"}
    return httpx.Response(200, json={"choices": [choice]})


@pytest.fixture
def settings(tiny_model_source: ModelSource, wordnet_directory: Path) -> Settings:
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    words = settings.words.model_copy(
        update={
            "listing_kinds": ["common"],
            "adjectives_per_listing": 4,
            "pile_sample_shards": 1,
            "usage_examples_per_word": 5,
        }
    )
    return settings.model_copy(update={"models": {"tiny": tiny_model_source}, "words": words})


def test_adjectives_are_checked_for_usage_and_category_association(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pile, "shard_file_names", lambda: FILE_NAMES)
    names = (LISTING_PROMPT, USAGE_PROMPT, ASSOCIATION_PROMPT)
    prompts = [PromptTemplate.load(name, REPOSITORY_ROOT / "llm_prompts") for name in names]
    client = LlmClient(
        settings.llm,
        tmp_path / "answers.sqlite3",
        rerun_from="words generate-adjective-candidates",
        transport=httpx.MockTransport(answer_like_the_llm),
    )
    sample_usages = partial(
        sample_word_usages,
        number_of_processes=1,
        download_directory=tmp_path / "pile",
        download=write_local_shard,
    )
    output_path = tmp_path / "adjective_candidates.csv"

    build_adjective_candidates(settings, prompts, client, output_path, sample_usages)

    with output_path.open(encoding="utf-8") as file:
        rows = {row["word"]: row for row in csv.DictReader(file)}
    assert rows["quiet"]["rejected_reason"] == ""
    assert rows["quiet"]["synset"] == "quiet.a.01"
    assert rows["quiet"]["pile_sample_count"] == "2"
    assert float(rows["quiet"]["adjective_usage_share"]) == pytest.approx(0.5)
    assert rows["quiet"]["largest_category_association"] == "1"
    assert rows["delicious"]["most_associated_category"] == "food"
    assert rows["delicious"]["largest_category_association"] == "6"
    assert rows["well-known"]["rejected_reason"] == "1語の小文字ではない"
    assert rows["pasta"]["rejected_reason"] == "WordNet で、形容詞の語義がない"
