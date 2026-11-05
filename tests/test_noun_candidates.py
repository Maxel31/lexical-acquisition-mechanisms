"""名詞の候補を作る処理のテスト。

LLM と Pile のダウンロードは、決まった答えを返す部品で置き換える。
"""

import csv
import json
import re
from functools import partial
from pathlib import Path

import httpx
import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

from lexical_acquisition_mechanisms import pile
from lexical_acquisition_mechanisms.llm_client import LlmClient, PromptTemplate
from lexical_acquisition_mechanisms.noun_candidates import (
    CATEGORY_PROMPT,
    LISTING_PROMPT,
    USAGE_PROMPT,
    build_noun_candidates,
)
from lexical_acquisition_mechanisms.pile_word_usages import sample_word_usages
from lexical_acquisition_mechanisms.settings import ModelSource, Settings, load_settings

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# 複数形で挙がった語は、そのカテゴリの語義がある単数形にする（giraffes は見出しになく、shoes は
# 「立場」の語義だけで見出しにある）
LISTED_NOUNS = {
    "food": ["pasta", "chicken", "ice cream", "car", "food"],
    "animal": ["giraffes", "chicken"],
    "clothing": ["shoes"],
}
TEXTS_BY_SHARD = [
    ["We ate pasta tonight.", "A pasta brand was sold."],
    ["Fresh pasta is good.", "The giraffe ran."],
]
FILE_NAMES = ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"]


def write_local_shard(file_name: str, directory: Path) -> Path:
    directory.mkdir(parents=True)
    path = directory / "shard.parquet"
    texts = TEXTS_BY_SHARD[FILE_NAMES.index(file_name)]
    parquet.write_table(arrow.table({"text": texts}), path)
    return path


def answer_like_the_llm(request: httpx.Request) -> httpx.Response:
    """名詞を挙げる問いには LISTED_NOUNS を返す。

    WordNet から足した語のカテゴリの問いには、horse と armadillo は animal、ほかは none と答える。
    使われ方の問いには、「brand」があれば名前の一部、「Fresh」があれば比喩（同じ語義）、
    それ以外は字義どおりと答える。
    """
    content = json.loads(request.content)["messages"][0]["content"]
    if "What does the English noun" in content:
        word = re.search(r'noun "(\w+)"', content).group(1)
        category = "animal" if word in {"horse", "armadillo"} else "none"
        answer = {"reason": "stub", "category": category}
    elif content.startswith("You are helping to build word lists"):
        category = re.search(r"\(category: (\w+)\)", content).group(1)
        nouns = [{"word": word, "meaning": "x"} for word in LISTED_NOUNS.get(category, [])]
        answer = {"nouns": nouns}
    else:
        passage = content.split("Passage:")[1]
        usage = "name" if "brand" in passage else "figurative" if "Fresh" in passage else "intended"
        answer = {"usage": usage, "reason": "stub"}
    choice = {"message": {"content": json.dumps(answer)}, "finish_reason": "stop"}
    return httpx.Response(200, json={"choices": [choice]})


@pytest.fixture
def settings(tiny_model_source: ModelSource, wordnet_directory: Path) -> Settings:
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    words = settings.words.model_copy(
        update={
            "listing_kinds": ["common"],
            "listing_letter_groups": ["a, b"],
            "nouns_per_listing": 5,
            "pile_sample_shards": 2,
            "usage_examples_per_word": 3,
        }
    )
    return settings.model_copy(update={"models": {"tiny": tiny_model_source}, "words": words})


def test_candidates_are_checked_with_wordnet_and_pile_usages(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pile, "shard_file_names", lambda: FILE_NAMES)
    prompts = [
        PromptTemplate.load(name, REPOSITORY_ROOT / "llm_prompts")
        for name in (LISTING_PROMPT, USAGE_PROMPT, CATEGORY_PROMPT)
    ]
    client = LlmClient(
        settings.llm,
        tmp_path / "answers.sqlite3",
        rerun_from="words generate-noun-candidates",
        transport=httpx.MockTransport(answer_like_the_llm),
    )
    sample_usages = partial(
        sample_word_usages,
        number_of_processes=1,
        download_directory=tmp_path / "pile",
        download=write_local_shard,
    )
    output_path = tmp_path / "noun_candidates.csv"

    build_noun_candidates(settings, prompts, client, output_path, sample_usages)

    with output_path.open(encoding="utf-8") as file:
        rows = {row["word"]: row for row in csv.DictReader(file)}
    assert rows["pasta"]["rejected_reason"] == ""
    assert rows["pasta"]["synset"] == "pasta.n.01"
    assert rows["pasta"]["token_count"] == "1"
    assert rows["pasta"]["pile_sample_count"] == "3"
    assert rows["pasta"]["estimated_pile_count"] == str(round(3 * 1650 / 2))
    assert rows["pasta"]["estimated_pile_passages"] == str(round(3 * 1650 / 2))
    assert rows["pasta"]["judged_usages"] == "3"
    # 字義どおりは3件中1件、同じ語義（比喩を含む）は2件
    assert float(rows["pasta"]["intended_usage_share"]) == pytest.approx(1 / 3)
    assert float(rows["pasta"]["same_meaning_usage_share"]) == pytest.approx(2 / 3)
    assert rows["giraffe"]["category"] == "animal"
    assert rows["giraffe"]["rejected_reason"] == ""
    assert rows["shoe"]["category"] == "clothing"
    assert rows["shoe"]["rejected_reason"] == ""
    assert "shoes" not in rows
    assert rows["chicken"]["rejected_reason"] == "複数のカテゴリで挙がった"
    assert rows["ice cream"]["rejected_reason"] == "1語の小文字ではない"
    assert rows["car"]["rejected_reason"] == "WordNet で、このカテゴリの語義がない"
    assert rows["food"]["rejected_reason"] == "カテゴリの名前（課題2の選択肢）"
    # LLM が挙げなかった3トークンまでの語は WordNet から足し、LLM が答えたカテゴリが合う語だけを残す
    assert (rows["horse"]["source"], rows["horse"]["rejected_reason"]) == ("wordnet", "")
    assert rows["horse"]["category"] == "animal"
    assert (rows["armadillo"]["source"], rows["armadillo"]["rejected_reason"]) == ("wordnet", "")
    assert int(rows["armadillo"]["token_count"]) > 1
    assert rows["comma"]["rejected_reason"] == "LLM が答えたカテゴリが違う（none）"
    assert rows["pasta"]["source"] == "llm-listing"
