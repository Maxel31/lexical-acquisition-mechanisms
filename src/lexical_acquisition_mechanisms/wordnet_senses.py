"""WordNet（英語の語彙データベース）での、語の語義とカテゴリ。

名詞の語義のカテゴリは、settings.yaml の words.categories に書いた上位語（WordNet の synset）の
下にあるかで決める（例: pasta.n.01 は food.n.01 の下にあるので food）。
WordNet のデータは、初めて使うときに data/nltk_data にダウンロードする。NLTK は、ほかの利用者が
書き込めるディレクトリにはダウンロードしないので、このディレクトリは持ち主だけが書き込めるようにする。
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import nltk
from nltk.corpus.reader.wordnet import WordNetCorpusReader

from lexical_acquisition_mechanisms.settings import CategorySettings

NLTK_DATA_DIRECTORY = Path("data/nltk_data")


@dataclass(frozen=True)
class Sense:
    """語の1つの語義。"""

    # WordNet の synset の名前（例: pasta.n.01）
    synset: str
    definition: str
    # この語義が属するカテゴリ（名詞のときだけ。どれにも属さなければ空）
    categories: tuple[str, ...] = ()


def noun_senses(word: str, categories: Sequence[CategorySettings]) -> list[Sense]:
    """語の名詞の語義を、WordNet の順（よく使われる順）に返す。"""
    wordnet = load_wordnet()
    roots = {
        category.name: {wordnet.synset(root) for root in category.wordnet_roots}
        for category in categories
    }
    senses = []
    for synset in wordnet.synsets(word, pos=wordnet.NOUN):
        if word not in _lemma_names(synset):
            continue
        ancestors = {synset} | set(synset.closure(lambda node: node.hypernyms()))
        belonging = tuple(name for name, root_set in roots.items() if ancestors & root_set)
        senses.append(Sense(synset.name(), synset.definition(), belonging))
    return senses


def adjective_senses(word: str) -> list[Sense]:
    """語の形容詞の語義（WordNet の形容詞と、それに準じる形容詞）を、よく使われる順に返す。"""
    wordnet = load_wordnet()
    return [
        Sense(synset.name(), synset.definition())
        for synset in wordnet.synsets(word, pos=wordnet.ADJ)
        if word in _lemma_names(synset)
    ]


def other_parts_of_speech(word: str) -> list[str]:
    """名詞と形容詞のほかに、その語が WordNet で持つ品詞（動詞 v・副詞 r）。"""
    wordnet = load_wordnet()
    return [
        part_of_speech
        for part_of_speech in (wordnet.VERB, wordnet.ADV)
        if any(word in _lemma_names(synset) for synset in wordnet.synsets(word, pos=part_of_speech))
    ]


def hypernym_words(synset_name: str) -> set[str]:
    """語義の、全ての上位語（間接の上位語と、固有名詞の上位語を含む）の見出しの語（小文字）。"""
    wordnet = load_wordnet()
    ancestors = wordnet.synset(synset_name).closure(
        lambda node: node.hypernyms() + node.instance_hypernyms()
    )
    return {lemma.name().lower() for ancestor in ancestors for lemma in ancestor.lemmas()}


def _lemma_names(synset) -> set[str]:
    # synsets() は複数形なども元の形に戻して引くので、語そのものが見出しの語義だけを使う
    return {lemma.name() for lemma in synset.lemmas()}


@cache
def load_wordnet() -> WordNetCorpusReader:
    """WordNet を読み込む。なければ data/nltk_data にダウンロードする。"""
    directory = NLTK_DATA_DIRECTORY.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)
    if str(directory) not in nltk.data.path:
        nltk.data.path.insert(0, str(directory))
    if not (directory / "corpora" / "wordnet.zip").exists():
        if not nltk.download("wordnet", download_dir=str(directory), quiet=True):
            raise RuntimeError(f"WordNet を {directory} にダウンロードできませんでした。")
    from nltk.corpus import wordnet

    wordnet.ensure_loaded()
    return wordnet
