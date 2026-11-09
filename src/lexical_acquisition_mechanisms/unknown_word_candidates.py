"""未知語の候補を作る（words generate-unknown-word-candidates）。

未知語は、OLMo 2 の事前学習データと Pile に一度も現れない、先頭スペース付きで2・3トークンの
文字列。綴りの近さで2種類作る。
- near: 6カテゴリに属さない実在語（形容詞・抽象名詞）の母音を、1〜2個、別の母音に替えた文字列
  （例: resonant → resonunt）。元の語は、先頭スペース付きで1〜2トークンになる、よく使われる語に
  する。
- far: ARC 非語データベースの非語のうち、同音の実在語がなく、語頭の子音の並びと語尾の部分が
  英語の語にあり、英語の語にない2文字の並びを含まないもの（発音できる非語）。
候補を無作為に並べ、infini-gram で一度も現れないことを確かめながら、トークン数と近さの組ごとに
集める。1トークンの未知語（新しいトークン番号）と、ほかの語と最初のトークンが重ならないことは、
words select で扱う。
"""

import argparse
import csv
import io
import itertools
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx

from lexical_acquisition_mechanisms.csv_tables import write_rows
from lexical_acquisition_mechanisms.infini_gram import InfiniGramClient
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.settings import Settings
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer
from lexical_acquisition_mechanisms.wordnet_senses import load_wordnet

COMMAND = "words generate-unknown-word-candidates"
OUTPUT_PATH = Path("data/words/unknown_word_candidates.csv")
TOKEN_COUNTS = (2, 3)
VOWELS = "aeiou"
# ARC 非語データベースの TSV の列（0 から数える）
ARC_NONWORD, ARC_BODY_NEIGHBOURS, ARC_WORDS_WITH_ONSET, ARC_ILLEGAL_BIGRAMS = 0, 5, 21, 23


@dataclass
class UnknownWordCandidate:
    word: str
    # 綴りの近さ（near: 実在語の母音を替えたもの、far: ARC の非語）
    spelling_closeness: str
    # near の元の実在語（far は空）
    source_word: str
    token_count: int
    # 先頭スペース付きの形のトークン番号（空白区切り）
    token_ids: str
    # infini-gram の索引のコーパスでの出現回数の合計（0 なら一度も現れない）
    occurrences: int


def generate_unknown_word_candidates(settings: Settings, arguments: argparse.Namespace) -> None:
    record_settings = {
        "unknown_words": settings.unknown_words.model_dump(),
        "words": settings.words.model_dump(include={"categories", "seed"}),
        "tokenizer": next(iter(settings.models.values())).model_dump(),
    }
    step = Step(COMMAND, inputs={}, outputs=[OUTPUT_PATH], settings=record_settings)
    run_unless_unchanged(
        step,
        lambda: build_unknown_word_candidates(settings, InfiniGramClient(COMMAND), OUTPUT_PATH),
        arguments.recompute,
    )


def build_unknown_word_candidates(
    settings: Settings,
    client: InfiniGramClient,
    output_path: Path,
    arc_transport: httpx.BaseTransport | None = None,
) -> None:
    tokenizer = WordTokenizer.from_model(next(iter(settings.models.values())))
    real_words = set(load_wordnet().all_lemma_names())
    far = _far_candidates(settings, tokenizer, real_words, arc_transport)
    checked: list[UnknownWordCandidate] = []
    for token_count in TOKEN_COUNTS:
        near = _near_candidates(settings, tokenizer, real_words)
        for candidates in (near, iter(far)):
            matching = (c for c in candidates if c.token_count == token_count)
            checked += _collect_absent(matching, settings, client)
    write_rows(output_path, UnknownWordCandidate, checked)


def _near_candidates(
    settings: Settings, tokenizer: WordTokenizer, real_words: set[str]
) -> Iterator[UnknownWordCandidate]:
    """実在語の母音を1〜2個替えた文字列を、元の語1つにつき1つ、無作為な順に返す。"""
    wordnet = load_wordnet()
    random_numbers = random.Random(settings.words.seed)
    roots = {
        wordnet.synset(root)
        for category in settings.words.categories
        for root in category.wordnet_roots
    }
    abstraction = wordnet.synset("abstraction.n.06")
    lemmas = {*wordnet.all_lemma_names(pos=wordnet.ADJ), *wordnet.all_lemma_names(pos=wordnet.NOUN)}
    sources = sorted(
        word for word in lemmas if word.isalpha() and word.islower() and len(word) >= 5
    )
    random_numbers.shuffle(sources)
    for source in sources:
        if len(tokenizer.word_token_ids(source)) > 2:
            continue
        noun_ancestors = [
            {synset} | set(synset.closure(lambda node: node.hypernyms()))
            for synset in wordnet.synsets(source, pos=wordnet.NOUN)
        ]
        if any(ancestors & roots for ancestors in noun_ancestors):
            continue
        is_adjective = bool(wordnet.synsets(source, pos=wordnet.ADJ))
        if not is_adjective and not all(abstraction in ancestors for ancestors in noun_ancestors):
            continue
        variants = sorted(set(_vowel_variants(source)) - real_words)
        if variants:
            yield _candidate(random_numbers.choice(variants), "near", source, tokenizer)


def _vowel_variants(word: str) -> Iterator[str]:
    """母音を1つまたは2つ、別の母音に替えた語。"""
    positions = [index for index, letter in enumerate(word) if letter in VOWELS]
    for number in (1, 2):
        for chosen in itertools.combinations(positions, number):
            choices = [[vowel for vowel in VOWELS if vowel != word[index]] for index in chosen]
            for replacement in itertools.product(*choices):
                letters = list(word)
                for index, vowel in zip(chosen, replacement, strict=True):
                    letters[index] = vowel
                yield "".join(letters)


def _far_candidates(
    settings: Settings,
    tokenizer: WordTokenizer,
    real_words: set[str],
    transport: httpx.BaseTransport | None,
) -> list[UnknownWordCandidate]:
    """ARC の発音できる非語を、無作為な順に返す。"""
    with httpx.Client(transport=transport, follow_redirects=True, timeout=300) as client:
        response = client.get(settings.unknown_words.arc_nonwords_url)
        response.raise_for_status()
    words = sorted(
        row[ARC_NONWORD]
        for row in csv.reader(io.StringIO(response.text), delimiter="\t")
        if int(row[ARC_WORDS_WITH_ONSET]) > 0
        and int(row[ARC_BODY_NEIGHBOURS]) > 0
        and int(row[ARC_ILLEGAL_BIGRAMS]) == 0
        and row[ARC_NONWORD].isalpha()
        and row[ARC_NONWORD] not in real_words
    )
    random.Random(settings.words.seed).shuffle(words)
    return [_candidate(word, "far", "", tokenizer) for word in words]


def _candidate(
    word: str, closeness: str, source: str, tokenizer: WordTokenizer
) -> UnknownWordCandidate:
    token_ids = tokenizer.word_token_ids(word)
    joined = " ".join(map(str, token_ids))
    return UnknownWordCandidate(word, closeness, source, len(token_ids), joined, -1)


def _collect_absent(
    candidates: Iterable[UnknownWordCandidate], settings: Settings, client: InfiniGramClient
) -> list[UnknownWordCandidate]:
    """一度も現れない候補が必要な数だけ集まるまで、候補をまとめて確かめる。確かめた候補を返す。"""
    needed = settings.unknown_words.candidates_per_group
    remaining = iter(candidates)
    checked: list[UnknownWordCandidate] = []
    while sum(candidate.occurrences == 0 for candidate in checked) < needed:
        batch = list(itertools.islice(remaining, needed))
        if not batch:
            break
        totals = dict.fromkeys((candidate.word for candidate in batch), 0)
        for index in settings.unknown_words.corpus_indexes:
            for word, count in client.occurrences(list(totals), index).items():
                totals[word] += count
        for candidate in batch:
            candidate.occurrences = totals[candidate.word]
        checked += batch
    return checked
