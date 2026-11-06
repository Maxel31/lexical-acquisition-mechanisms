"""形容詞の候補を作る（words generate-adjective-candidates）。

1. LLM が、6カテゴリのどれとも結びつきが強くない形容詞を挙げる（挙げる語の種類を変えて、何回か）。
2. WordNet に形容詞の語義があり、先頭スペース付きで1トークンの語だけを残す。
3. Pile の先頭の断片で、標準の形の出現回数を数え、見本を選ぶ。
4. LLM が見本ごとに、形容詞として使われているかを判定する。
5. LLM が、各カテゴリとの結びつきの強さ（その種類のものに特有か。1: ほかのものと同じくらい使う〜
   7: ほとんどその種類のものに使う）を 1〜7 で評定する。短い理由を先に書かせる。
形容詞72語と自然な形容詞は、words select がこの候補から選ぶ。
"""

import argparse
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from lexical_acquisition_mechanisms.csv_tables import write_rows
from lexical_acquisition_mechanisms.llm_client import (
    ANSWERS_PATH,
    LlmClient,
    LlmRequest,
    PromptTemplate,
    prompt_record_settings,
    require_answers,
    with_maximum_items,
)
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.pile_word_usages import WordUsages, sample_word_usages
from lexical_acquisition_mechanisms.settings import Settings
from lexical_acquisition_mechanisms.usage_shares import usage_shares
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer
from lexical_acquisition_mechanisms.wordnet_senses import (
    adjective_senses,
    noun_senses,
    other_parts_of_speech,
)

COMMAND = "words generate-adjective-candidates"
OUTPUT_PATH = Path("data/words/adjective_candidates.csv")
LISTING_PROMPT = "list-adjectives"
USAGE_PROMPT = "judge-adjective-usage"
ASSOCIATION_PROMPT = "rate-category-association"
WORD_FORM = re.compile(r"^[a-z]+$")
LISTING_SCHEMA = {
    "type": "object",
    "properties": {
        "adjectives": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"word": {"type": "string"}, "meaning": {"type": "string"}},
                "required": ["word", "meaning"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["adjectives"],
    "additionalProperties": False,
}
# 理由を先に書かせる（品詞を書いてから決めるほうが、名詞の使い方（a government official）を
# 見分けやすい）
USAGE_SCHEMA = {
    "type": "object",
    "properties": {"reason": {"type": "string"}, "adjective": {"type": "boolean"}},
    "required": ["reason", "adjective"],
    "additionalProperties": False,
}


@dataclass
class AdjectiveCandidate:
    word: str
    # LLM が挙げたときの、意図した意味の説明
    llm_meaning: str
    # 候補から外した理由（外していなければ空）
    rejected_reason: str = ""
    # 最もよく使われる WordNet の形容詞の語義と、その説明
    synset: str = ""
    definition: str = ""
    number_of_noun_senses: int = 0
    # 名詞と形容詞のほかに WordNet で持つ品詞（v: 動詞, r: 副詞）
    other_parts_of_speech: str = ""
    token_count: int = 0
    pile_sample_count: int = 0
    estimated_pile_count: int = 0
    # Pile 全体で、用例として切り出せる文章の数の見積もり
    estimated_pile_passages: int = 0
    # LLM が判定できた見本の数と、そのうち形容詞として使われていた割合
    judged_usages: int = 0
    adjective_usage_share: float = 0.0
    # 各カテゴリとの結びつきの評定（1〜7）のうち最も強いものと、そのカテゴリ
    largest_category_association: int = 0
    most_associated_category: str = ""


def generate_adjective_candidates(settings: Settings, arguments: argparse.Namespace) -> None:
    names = (LISTING_PROMPT, USAGE_PROMPT, ASSOCIATION_PROMPT)
    prompts = [PromptTemplate.load(name) for name in names]
    record_settings = {
        "words": settings.words.model_dump(exclude={"nouns_per_listing"}),
        "passages": settings.passages.model_dump(
            include={"tokens_per_passage", "maximum_passages_per_document"}
        ),
        "tokenizer": next(iter(settings.models.values())).model_dump(),
        **prompt_record_settings(settings.llm, prompts),
    }
    step = Step(COMMAND, inputs={}, outputs=[OUTPUT_PATH], settings=record_settings)
    client = LlmClient(settings.llm, ANSWERS_PATH, rerun_from=COMMAND)
    run_unless_unchanged(
        step,
        lambda: build_adjective_candidates(settings, prompts, client, OUTPUT_PATH),
        arguments.recompute,
    )


def build_adjective_candidates(
    settings: Settings,
    prompts: Sequence[PromptTemplate],
    client: LlmClient,
    output_path: Path,
    sample_usages: Callable[..., dict[str, WordUsages]] = sample_word_usages,
) -> None:
    listing_prompt, usage_prompt, association_prompt = prompts
    words = settings.words
    category_list = ", ".join(category.description for category in words.categories)
    requests = [
        LlmRequest(
            listing_prompt,
            {
                "number": str(words.adjectives_per_listing),
                "kind": kind,
                "letters": letters,
                "categories": category_list,
            },
            with_maximum_items(LISTING_SCHEMA, "adjectives", words.adjectives_per_listing),
            maximum_tokens=60 * words.adjectives_per_listing,
        )
        for kind in words.listing_kinds
        for letters in words.listing_letter_groups
    ]
    candidates: dict[str, AdjectiveCandidate] = {}
    answers = client.ask_all(requests)
    require_answers(answers, "形容詞の一覧", COMMAND)
    for answer in answers:
        for adjective in (answer or {"adjectives": []})["adjectives"]:
            word = adjective["word"].strip()
            candidates.setdefault(word, AdjectiveCandidate(word, adjective["meaning"]))
    tokenizer = WordTokenizer.from_model(next(iter(settings.models.values())))
    for candidate in candidates.values():
        _check(candidate, settings, tokenizer)
    kept = [candidate for candidate in candidates.values() if not candidate.rejected_reason]
    usages = sample_usages(
        [candidate.word for candidate in kept],
        words.pile_sample_shards,
        words.usage_examples_per_word,
        words.seed,
        settings.passages,
    )

    def make_request(word: str, passage: str) -> LlmRequest:
        return LlmRequest(usage_prompt, {"word": word, "passage": passage}, USAGE_SCHEMA)

    shares = usage_shares(
        [candidate.word for candidate in kept],
        usages,
        words.pile_sample_shards,
        make_request,
        {"adjective": lambda answer: answer["adjective"]},
        client,
    )
    for candidate in kept:
        share = shares[candidate.word]
        candidate.pile_sample_count = share.pile_sample_count
        candidate.estimated_pile_count = share.estimated_pile_count
        candidate.estimated_pile_passages = share.estimated_pile_passages
        candidate.judged_usages = share.judged_usages
        candidate.adjective_usage_share = share.shares["adjective"]
    _rate_associations(kept, settings, association_prompt, client)
    write_rows(
        output_path,
        AdjectiveCandidate,
        sorted(candidates.values(), key=lambda candidate: candidate.word),
    )


def _check(candidate: AdjectiveCandidate, settings: Settings, tokenizer: WordTokenizer) -> None:
    if not WORD_FORM.match(candidate.word):
        candidate.rejected_reason = "1語の小文字ではない"
        return
    senses = adjective_senses(candidate.word)
    candidate.number_of_noun_senses = len(noun_senses(candidate.word, settings.words.categories))
    candidate.other_parts_of_speech = "".join(other_parts_of_speech(candidate.word))
    candidate.token_count = len(tokenizer.word_token_ids(candidate.word))
    if not senses:
        candidate.rejected_reason = "WordNet で、形容詞の語義がない"
    elif candidate.token_count != 1:
        candidate.rejected_reason = "先頭スペース付きで1トークンではない"
    else:
        candidate.synset = senses[0].synset
        candidate.definition = senses[0].definition


def _rate_associations(
    candidates: Sequence[AdjectiveCandidate],
    settings: Settings,
    prompt: PromptTemplate,
    client: LlmClient,
) -> None:
    categories = settings.words.categories
    category_lines = "\n".join(
        f"- {category.name}: {category.description}" for category in categories
    )
    names = [category.name for category in categories]
    schema = {
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
            **{name: {"type": "integer", "enum": list(range(1, 8))} for name in names},
        },
        "required": ["reason", *names],
        "additionalProperties": False,
    }
    requests = [
        LlmRequest(prompt, {"word": candidate.word, "categories": category_lines}, schema)
        for candidate in candidates
    ]
    for candidate, ratings in zip(candidates, client.ask_all(requests), strict=True):
        if ratings is None:
            candidate.rejected_reason = "カテゴリとの結びつきを評定できなかった"
            continue
        candidate.most_associated_category = max(names, key=ratings.get)
        candidate.largest_category_association = ratings[candidate.most_associated_category]
