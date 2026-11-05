"""名詞の候補を作る（words generate-noun-candidates）。

1. LLM が、カテゴリごとに名詞を挙げる（挙げる語の種類を変えて、何回か）。挙げた形に、WordNet で
   そのカテゴリの語義がなく、複数形を戻した形（例: shoes → shoe）にあれば、戻した形にする。
2. WordNet で、そのカテゴリの語義があり、ほかのカテゴリの語義がない語だけを残す。
3. Pile の先頭の断片で、標準の形（例: " pasta"）の出現回数を数え、見本を選ぶ。
4. LLM が見本ごとに、その語の使われ方を判定する（USAGES: 意図した語義（WordNet の説明）の名詞として
   字義どおり・同じ語義の比喩や慣用句や複合名詞の一部・別の語義・名前の一部・名詞でない）。
   字義どおりの割合（後で用例として使える割合）と、同じ語義の割合（比喩などを含む。設計書の
   「ほぼ単義」を、別の語義・名前・名詞でない使い方が少ないことで判断するため）を出す。
名詞72語・ターゲット語18語・自然な名詞は、words select がこの候補から選ぶ。
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
from lexical_acquisition_mechanisms.settings import CategorySettings, Settings
from lexical_acquisition_mechanisms.usage_shares import usage_shares
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer
from lexical_acquisition_mechanisms.wordnet_senses import (
    load_wordnet,
    noun_senses,
    other_parts_of_speech,
)

COMMAND = "words generate-noun-candidates"
OUTPUT_PATH = Path("data/words/noun_candidates.csv")
LISTING_PROMPT = "list-category-nouns"
USAGE_PROMPT = "judge-noun-usage"
WORD_FORM = re.compile(r"^[a-z]+$")
LISTING_SCHEMA = {
    "type": "object",
    "properties": {
        "nouns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"word": {"type": "string"}, "meaning": {"type": "string"}},
                "required": ["word", "meaning"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["nouns"],
    "additionalProperties": False,
}
# 見本での使われ方（先頭の2つは、意図した語義）
USAGES = ("intended", "figurative", "another meaning", "name", "not a noun")
USAGE_SCHEMA = {
    "type": "object",
    "properties": {"usage": {"type": "string", "enum": list(USAGES)}, "reason": {"type": "string"}},
    "required": ["usage", "reason"],
    "additionalProperties": False,
}


@dataclass
class NounCandidate:
    word: str
    category: str
    # LLM が挙げたときの、意図した意味の説明
    llm_meaning: str
    # 候補から外した理由（外していなければ空）
    rejected_reason: str = ""
    # そのカテゴリで最もよく使われる WordNet の語義と、その説明
    synset: str = ""
    definition: str = ""
    number_of_noun_senses: int = 0
    # 名詞と形容詞のほかに WordNet で持つ品詞（v: 動詞, r: 副詞）
    other_parts_of_speech: str = ""
    token_count: int = 0
    # Pile の先頭の断片での、標準の形の出現回数と、Pile 全体での見積もり
    pile_sample_count: int = 0
    estimated_pile_count: int = 0
    # Pile 全体で、用例として切り出せる文章の数の見積もり
    estimated_pile_passages: int = 0
    # LLM が判定できた見本の数と、そのうち意図した語義で、字義どおりに使われていた割合と、
    # 意図した語義で使われていた割合（比喩・慣用句・複合名詞の一部を含む）
    judged_usages: int = 0
    intended_usage_share: float = 0.0
    same_meaning_usage_share: float = 0.0


def generate_noun_candidates(settings: Settings, arguments: argparse.Namespace) -> None:
    prompts = [PromptTemplate.load(LISTING_PROMPT), PromptTemplate.load(USAGE_PROMPT)]
    record_settings = {
        "words": settings.words.model_dump(exclude={"adjectives_per_listing"}),
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
        lambda: build_noun_candidates(settings, prompts, client, OUTPUT_PATH),
        arguments.recompute,
    )


def build_noun_candidates(
    settings: Settings,
    prompts: Sequence[PromptTemplate],
    client: LlmClient,
    output_path: Path,
    sample_usages: Callable[..., dict[str, WordUsages]] = sample_word_usages,
) -> None:
    listing_prompt, usage_prompt = prompts
    words = settings.words
    requests = [
        LlmRequest(
            listing_prompt,
            {
                "number": str(words.nouns_per_listing),
                "category": category.name,
                "description": category.description,
                "kind": kind,
                "letters": letters,
            },
            with_maximum_items(LISTING_SCHEMA, "nouns", words.nouns_per_listing),
            maximum_tokens=60 * words.nouns_per_listing,
        )
        for category in words.categories
        for kind in words.listing_kinds
        for letters in words.listing_letter_groups
    ]
    answers = client.ask_all(requests)
    require_answers(answers, "名詞の一覧", COMMAND)
    candidates = _listed_candidates(requests, answers, words.categories)
    tokenizer = WordTokenizer.from_model(next(iter(settings.models.values())))
    for candidate in candidates:
        if not candidate.rejected_reason:
            _check_with_wordnet(candidate, settings)
            candidate.token_count = len(tokenizer.word_token_ids(candidate.word))
    kept = [candidate for candidate in candidates if not candidate.rejected_reason]
    usages = sample_usages(
        [candidate.word for candidate in kept],
        words.pile_sample_shards,
        words.usage_examples_per_word,
        words.seed,
        settings.passages,
    )
    _judge_usages(kept, usages, usage_prompt, client, words.pile_sample_shards)
    write_rows(output_path, NounCandidate, candidates)


def _listed_candidates(
    requests: Sequence[LlmRequest],
    answers: Sequence[dict | None],
    categories: Sequence[CategorySettings],
) -> list[NounCandidate]:
    """LLM が挙げた名詞。複数のカテゴリで挙がった語や、1語の小文字でない語は外す。"""
    by_word: dict[str, NounCandidate] = {}
    for request, answer in zip(requests, answers, strict=True):
        category = request.values["category"]
        for noun in (answer or {"nouns": []})["nouns"]:
            word = _listed_form(noun["word"].strip(), category, categories)
            candidate = by_word.setdefault(word, NounCandidate(word, category, noun["meaning"]))
            if candidate.category != category:
                candidate.rejected_reason = "複数のカテゴリで挙がった"
            elif not WORD_FORM.match(word):
                candidate.rejected_reason = "1語の小文字ではない"
    return sorted(by_word.values(), key=lambda candidate: (candidate.category, candidate.word))


def _listed_form(word: str, category: str, categories: Sequence[CategorySettings]) -> str:
    """挙がった語の、候補にする形。

    挙がった形に、WordNet でそのカテゴリの語義がなく、複数形を戻した形にあれば、戻した形にする
    （例: shoes は「立場」の語義だけで見出しにあるので、shoe にする）。どちらにもなければ、
    そのまま。
    """
    for form in (word, *_singular_forms(word)):
        if any(category in sense.categories for sense in noun_senses(form, categories)):
            return form
    return word


def _singular_forms(word: str) -> list[str]:
    """複数形を戻した形のうち、WordNet の名詞の見出しにあるもの（WordNet の規則と例外の表で
    戻す）。"""
    wordnet = load_wordnet()
    forms = [wordnet.morphy(word, wordnet.NOUN)] + [
        word.removesuffix(suffix) + replacement
        for suffix, replacement in wordnet.MORPHOLOGICAL_SUBSTITUTIONS[wordnet.NOUN]
        if word.endswith(suffix)
    ]
    return [
        form
        for form in dict.fromkeys(forms)
        if form and form != word and wordnet.lemmas(form, pos=wordnet.NOUN)
    ]


def _check_with_wordnet(candidate: NounCandidate, settings: Settings) -> None:
    senses = noun_senses(candidate.word, settings.words.categories)
    in_category = [sense for sense in senses if candidate.category in sense.categories]
    other_categories = {
        name for sense in senses for name in sense.categories if name != candidate.category
    }
    candidate.number_of_noun_senses = len(senses)
    if not in_category:
        candidate.rejected_reason = "WordNet で、このカテゴリの語義がない"
    elif other_categories:
        candidate.rejected_reason = (
            f"WordNet で、ほかのカテゴリの語義もある（{sorted(other_categories)}）"
        )
    else:
        candidate.synset = in_category[0].synset
        candidate.definition = in_category[0].definition
    candidate.other_parts_of_speech = "".join(other_parts_of_speech(candidate.word))


def _judge_usages(
    candidates: Sequence[NounCandidate],
    usages: dict[str, WordUsages],
    prompt: PromptTemplate,
    client: LlmClient,
    pile_sample_shards: int,
) -> None:
    definitions = {candidate.word: candidate.definition for candidate in candidates}

    def make_request(word: str, passage: str) -> LlmRequest:
        values = {"word": word, "definition": definitions[word], "passage": passage}
        return LlmRequest(prompt, values, USAGE_SCHEMA)

    words = [candidate.word for candidate in candidates]
    conditions = {
        "intended": lambda answer: answer["usage"] == USAGES[0],
        "same meaning": lambda answer: answer["usage"] in USAGES[:2],
    }
    shares = usage_shares(words, usages, pile_sample_shards, make_request, conditions, client)
    for candidate in candidates:
        share = shares[candidate.word]
        candidate.pile_sample_count = share.pile_sample_count
        candidate.estimated_pile_count = share.estimated_pile_count
        candidate.estimated_pile_passages = share.estimated_pile_passages
        candidate.judged_usages = share.judged_usages
        candidate.intended_usage_share = share.shares["intended"]
        candidate.same_meaning_usage_share = share.shares["same meaning"]
