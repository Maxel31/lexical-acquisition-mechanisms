"""通しのテスト（test_end_to_end.py）のための、データ作成の段階の生成物を小さく作る部品。

語と用例の作成（LLM・Pile・基データのダウンロードが要る段階）の代わりに、それらのコマンドが書く
生成物を、極小のモデルのトークナイザの語彙から作った語で書き出し、処理の記録も残す（以降の
コマンドが、入力として使えるように）。
"""

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet
from transformers import AutoTokenizer

from lexical_acquisition_mechanisms import (
    adjective_candidates,
    base_data,
    hidden_state_similarity,
    neutral_contexts,
    noun_candidates,
    passage_finalization,
    similarity_ratings,
    word_finalization,
)
from lexical_acquisition_mechanisms.adjective_candidates import AdjectiveCandidate
from lexical_acquisition_mechanisms.csv_tables import write_rows
from lexical_acquisition_mechanisms.hidden_state_similarity import HiddenStateSimilarity
from lexical_acquisition_mechanisms.neutral_contexts import NeutralContext
from lexical_acquisition_mechanisms.noun_candidates import NounCandidate
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.passage_collection import CANDIDATE_SCHEMA
from lexical_acquisition_mechanisms.settings import Settings
from lexical_acquisition_mechanisms.similarity_ratings import SimilarityRating
from lexical_acquisition_mechanisms.word_selection import SelectedWord
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

# 問題文・用例の決まった文や、課題の選択肢に使う語（管理対象の語には使わない）
RESERVED = {
    "food",
    "animal",
    "vehicle",
    "clothing",
    "furniture",
    "building",
    "noun",
    "adjective",
    "the",
    "was",
    "right",
    "here",
    "today",
    "fine",
    "sentence",
    "question",
    "answer",
    "word",
    "means",
    "same",
    "kind",
    "part",
    "speech",
    "this",
    "yes",
    "know",
    "about",
    "heard",
}
CUES = ("saw", "took", "liked", "moved", "kept", "found")
SUBJECTS = ("She", "He", "They", "We", "You", "I")


class WordSource:
    """トークナイザの語彙から、指定したトークン数の、最初のトークンが互いに重ならない語を作る。"""

    def __init__(self, tokenizer: WordTokenizer, repository: str) -> None:
        vocabulary = AutoTokenizer.from_pretrained(repository).get_vocab()
        self._tokenizer = tokenizer
        self._pieces = sorted(
            token[1:]
            for token in vocabulary
            if token.startswith("Ġ")
            and token[1:].isascii()
            and token[1:].isalpha()
            and token[1:].islower()
            and 5 <= len(token) - 1 <= 8
            and token[1:] not in RESERVED
        )
        self._suffixes = sorted(
            token
            for token in vocabulary
            if token.isascii() and token.isalpha() and token.islower() and 2 <= len(token) <= 3
        )
        self._random = random.Random(0)
        self._random.shuffle(self._pieces)
        self._first_tokens: set[int] = set()

    def take(self, token_count: int) -> tuple[str, list[int]]:
        while True:
            suffixes = (self._random.choice(self._suffixes) for _ in range(token_count - 1))
            word = self._pieces.pop() + "".join(suffixes)
            ids = self._tokenizer.word_token_ids(word)
            if len(ids) == token_count and ids[0] not in self._first_tokens:
                self._first_tokens.add(ids[0])
                return word, ids


@dataclass
class World:
    """作った語（役割ごと）と、課題の言い換えの語。"""

    words: list[SelectedWord] = field(default_factory=list)
    candidate_nouns: list[NounCandidate] = field(default_factory=list)
    candidate_adjectives: list[AdjectiveCandidate] = field(default_factory=list)
    paraphrases: list[str] = field(default_factory=list)


def build_world(settings: Settings, tokenizer: WordTokenizer, repository: str) -> World:
    source = WordSource(tokenizer, repository)
    categories = [category.name for category in settings.words.categories]
    world = World()

    def add(word: str, ids: list[int], role: str, **values) -> None:
        world.words.append(
            SelectedWord(
                word,
                role,
                token_count=len(ids),
                token_ids=" ".join(map(str, ids)),
                synset=f"{word}.n.01",
                definition=f"a thing called {word}",
                usage_share=1.0,
                **values,
            )
        )

    for category in categories:
        # 名詞と形容詞は、words select と同じく1トークンの語
        for _ in range(12):
            add(*source.take(1), "noun", category=category)
        for token_count in (1, 2, 3):
            add(*source.take(token_count), "target", category=category)
        # 自然な名詞は1・2・3トークンを順に。1トークンの10語は、品詞のプローブにも使う
        for number in range(30):
            word, ids = source.take(1 + number % 3)
            add(word, ids, "natural-noun", category=category, part_of_speech_probe=len(ids) == 1)
    for _ in range(72):
        add(*source.take(1), "adjective")
    for _ in range(60):
        add(*source.take(1), "natural-adjective", part_of_speech_probe=True)
    unused = tokenizer.unused_token_ids(24)
    for number, token_id in enumerate(unused):
        add(f"<new_{number + 1:02d}>", [token_id], "unknown")
    for token_count in (2, 3):
        for closeness in ("near", "far"):
            for _ in range(12):
                add(*source.take(token_count), "unknown", spelling_closeness=closeness)
    for token_count in (1, 2, 3):
        for number in range(8):
            word, ids = source.take(token_count)
            world.candidate_nouns.append(
                NounCandidate(
                    word,
                    categories[number % 6],
                    "",
                    token_count=token_count,
                    estimated_pile_count=1000 - number,
                )
            )
            word, ids = source.take(token_count)
            world.candidate_adjectives.append(
                AdjectiveCandidate(
                    word, "", token_count=token_count, estimated_pile_count=1000 - number
                )
            )
    world.paraphrases = [source.take(1)[0] for _ in range(4)]
    return world


def write_world(settings: Settings, tokenizer: WordTokenizer, world: World) -> None:
    """データ作成の段階の生成物を、処理の記録とともに書く（作業のディレクトリで呼ぶ）。"""
    _write(
        word_finalization.COMMAND,
        [word_finalization.OUTPUT_PATH],
        lambda: write_rows(word_finalization.OUTPUT_PATH, SelectedWord, world.words),
    )
    targets = [word for word in world.words if word.role == "target"]
    nouns = [word for word in world.words if word.role == "noun"]
    pairs = [(target, noun) for target in targets for noun in nouns]
    random_numbers = random.Random(0)
    _write(
        hidden_state_similarity.COMMAND,
        [hidden_state_similarity.SIMILARITY_PATH],
        lambda: write_rows(
            hidden_state_similarity.SIMILARITY_PATH,
            HiddenStateSimilarity,
            [
                HiddenStateSimilarity(
                    target.word,
                    target.category,
                    noun.word,
                    noun.category,
                    model,
                    random_numbers.uniform(0, 1),
                )
                for target, noun in pairs
                for model in settings.models
            ],
        ),
    )
    _write(
        similarity_ratings.COMMAND,
        [similarity_ratings.OUTPUT_PATH],
        lambda: write_rows(
            similarity_ratings.OUTPUT_PATH,
            SimilarityRating,
            [
                SimilarityRating(
                    target.word,
                    target.category,
                    noun.word,
                    noun.category,
                    random_numbers.randint(1, 7),
                )
                for target, noun in pairs
            ],
        ),
    )
    _write(
        noun_candidates.COMMAND,
        [noun_candidates.OUTPUT_PATH],
        lambda: write_rows(noun_candidates.OUTPUT_PATH, NounCandidate, world.candidate_nouns),
    )
    _write(
        adjective_candidates.COMMAND,
        [adjective_candidates.OUTPUT_PATH],
        lambda: write_rows(
            adjective_candidates.OUTPUT_PATH, AdjectiveCandidate, world.candidate_adjectives
        ),
    )
    contexts = [NeutralContext(0, ""), NeutralContext(1, "Do you know about the")]
    contexts.append(NeutralContext(2, "I heard something about the"))
    _write(
        neutral_contexts.COMMAND,
        [neutral_contexts.CONTEXTS_PATH],
        lambda: write_rows(neutral_contexts.CONTEXTS_PATH, NeutralContext, contexts),
    )
    _write(
        passage_finalization.COMMAND,
        [passage_finalization.PASSAGES_PATH],
        lambda: _write_passages(settings, tokenizer, world),
    )
    paths = base_data.output_paths()
    _write(
        base_data.COMMAND,
        [paths["base"], paths["validation"]],
        lambda: _write_base_data(settings, tokenizer, paths),
    )


def _write(command: str, outputs: list[Path], compute: Callable[[], None]) -> None:
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    run_unless_unchanged(Step(command, inputs={}, outputs=outputs), compute)


def _write_passages(settings: Settings, tokenizer: WordTokenizer, world: World) -> None:
    """語ごとに、学習用と評価用に要る数の用例（文の途中に語を1回置いたもの）。"""
    length = settings.passages.tokens_per_passage
    exposures = max(settings.unknown_word_acquisition.exposure_levels)
    # 2つ目の語義の追加で、名詞を Y にする分（同じ名詞を何回か Y にしても足りるように）
    second = 3 * max(settings.second_sense_addition.second_exposure_levels)
    added = settings.sense_addition.passages_per_added_sense
    needed = {
        "noun": {
            "training": exposures + second,
            "evaluation": settings.passages.evaluation_passages_per_word,
        },
        "adjective": {
            "training": added,
            "evaluation": settings.passages.evaluation_passages_per_word,
        },
        "target": {"evaluation": settings.passages.evaluation_passages_per_word},
        "natural-noun": {"evaluation": settings.evaluation.passages_per_natural_word},
        "natural-adjective": {"evaluation": settings.evaluation.passages_per_natural_word},
    }
    rows = []
    for word in world.words:
        for split, count in needed.get(word.role, {}).items():
            for number in range(count):
                before = tokenizer.encode([f"Yes {number}. The"])
                ids = [int(token) for token in word.token_ids.split()]
                after = tokenizer.encode([" was right here today. It was fine."])
                token_ids = (before + ids + after * 4)[:length]
                rows.append(
                    {
                        "word": word.word,
                        "role": word.role,
                        "shard": 0,
                        "row": len(rows),
                        "start": 0,
                        "key": random.Random(f"{word.word}:{split}:{number}").random(),
                        "token_ids": token_ids,
                        "occurrence_positions": [len(before)],
                        "split": split,
                    }
                )
    schema = CANDIDATE_SCHEMA.append(arrow.field("split", arrow.string()))
    parquet.write_table(
        arrow.Table.from_pylist(rows, schema=schema), passage_finalization.PASSAGES_PATH
    )


def _write_base_data(settings: Settings, tokenizer: WordTokenizer, paths: dict[str, Path]) -> None:
    """文書の区切りを約20トークンごとに置いた、無作為なトークンの系列。"""
    length = settings.training.sequence_length
    random_numbers = np.random.default_rng(0)
    for name, count in (("base", 96), ("validation", 4)):
        tokens = random_numbers.integers(1000, 20000, (count, length)).astype(np.int32)
        tokens[:, ::20] = tokenizer.end_of_text_id
        np.save(paths[name], tokens)
