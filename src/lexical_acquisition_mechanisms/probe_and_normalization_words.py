"""実在語らしさのプローブの語と、未知語の指標を正規化するときの基準の語を作る
（evaluation-data build-probe-and-normalization-words）。

プローブの語（data/evaluation/probe_words.csv。トークン数 1・2・3 ごと）
- 正例: 実在語。名詞と形容詞の候補のうち、管理対象の語と既知の名詞（下の正規化の基準。プローブの
  学習に使った語を評価に使わないため）でなく、Pile で多く使われる語（頻度の高い順に
  evaluation.probe_words_per_token_count 語）。
- 負例: 2・3トークンは、実在語どうしで2つ目以降のトークンを入れ替えた非語（例: " gir|affe" と
  " las|agna" → " gir|agna" と " las|affe"）。実在語と同じ形（候補の一覧か WordNet にある語）や、
  実在語のトークン列になるもの、その文字列の標準のトークン化と違うトークン列になるもの（未知語は
  標準のトークン化なので）は除く。組を作り直して、実在語と同じ数まで作る。
  1トークンは、学習していない新しいトークン（未知語に使わなかった未使用のトークン番号。学習のときと
  同じ方法で初期化して使う）。
- 語の HELD_OUT_SHARE を、学習に使わない確認用（プローブが、学習に使わなかった語でも当てられるか）に
  する。
正規化の基準の語（data/evaluation/normalization_words.csv）
- 既知の名詞: 2・3トークンの名詞の候補のうち、管理対象の語でなく、頻度の高い
  evaluation.normalization_nouns_per_token_count 語（形への慣れと語としての統合の基準 m1）。
- 名詞72語を、人工的に2・3トークンに分けた形（例: " pas|ta"・" p|as|ta"。意味の基準 m1）。各部分が
  1トークンになる分け方のうち、部分の長さが最もそろうもの。そう分けられない名詞とトークン数の組は
  作らずに、数を表示する（その組の意味の正規化は、値なしになる）。
"""

import argparse
import itertools
import random
from dataclasses import dataclass
from pathlib import Path

from lexical_acquisition_mechanisms import (
    adjective_candidates,
    noun_candidates,
    word_finalization,
)
from lexical_acquisition_mechanisms.adjective_candidates import AdjectiveCandidate
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.noun_candidates import NounCandidate
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.settings import Settings
from lexical_acquisition_mechanisms.word_selection import SelectedWord
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer
from lexical_acquisition_mechanisms.wordnet_senses import load_wordnet

COMMAND = "evaluation-data build-probe-and-normalization-words"
OUTPUT_DIRECTORY = Path("data/evaluation")
PROBE_WORDS_PATH = OUTPUT_DIRECTORY / "probe_words.csv"
NORMALIZATION_WORDS_PATH = OUTPUT_DIRECTORY / "normalization_words.csv"
TOKEN_COUNTS = (1, 2, 3)
HELD_OUT_SHARE = 0.2
# 非語を作るときに、実在語の組を作り直す回数の上限
PAIRING_ROUNDS = 10
MANAGED_ROLES = ("noun", "target", "adjective", "unknown")


@dataclass
class ProbeWord:
    token_count: int
    # real-word: 実在語、nonword: 非語、untrained-token: 学習していない新しいトークン
    kind: str
    text: str
    token_ids: str
    # training: プローブの学習に使う、held-out: 確認用
    split: str


@dataclass
class NormalizationWord:
    # known-noun: 既知の名詞、split-noun: 名詞を人工的に分けた形
    kind: str
    token_count: int
    text: str
    token_ids: str
    # split-noun の元の名詞
    source_noun: str = ""


def build_probe_and_normalization_words(settings: Settings, arguments: argparse.Namespace) -> None:
    included = {"probe_words_per_token_count", "normalization_nouns_per_token_count"}
    step = Step(
        COMMAND,
        inputs={
            word_finalization.OUTPUT_PATH: word_finalization.COMMAND,
            noun_candidates.OUTPUT_PATH: noun_candidates.COMMAND,
            adjective_candidates.OUTPUT_PATH: adjective_candidates.COMMAND,
        },
        outputs=[PROBE_WORDS_PATH, NORMALIZATION_WORDS_PATH],
        settings={
            "evaluation": settings.evaluation.model_dump(include=included),
            "seed": settings.words.seed,
            "tokenizer": next(iter(settings.models.values())).model_dump(),
        },
    )
    run_unless_unchanged(
        step,
        lambda: write_probe_and_normalization_words(
            settings,
            (
                word_finalization.OUTPUT_PATH,
                noun_candidates.OUTPUT_PATH,
                adjective_candidates.OUTPUT_PATH,
            ),
            (PROBE_WORDS_PATH, NORMALIZATION_WORDS_PATH),
        ),
        arguments.recompute,
    )


def write_probe_and_normalization_words(
    settings: Settings, input_paths: tuple[Path, Path, Path], output_paths: tuple[Path, Path]
) -> None:
    words_path, nouns_path, adjectives_path = input_paths
    tokenizer = WordTokenizer.from_model(next(iter(settings.models.values())))
    words = read_rows(words_path, SelectedWord)
    managed = {word.word for word in words if word.role in MANAGED_ROLES}
    nouns = [row for row in read_rows(nouns_path, NounCandidate) if not row.rejected_reason]
    adjectives = [
        row for row in read_rows(adjectives_path, AdjectiveCandidate) if not row.rejected_reason
    ]
    real_words = {row.word for row in [*nouns, *adjectives]} | managed
    random_numbers = random.Random(settings.words.seed)
    evaluation = settings.evaluation
    known_nouns = {
        token_count: _frequent_words(nouns, managed, token_count, tokenizer)[
            : evaluation.normalization_nouns_per_token_count
        ]
        for token_count in (2, 3)
    }
    excluded = managed | {word for known in known_nouns.values() for word, _ in known}
    probe_words = []
    for token_count in TOKEN_COUNTS:
        chosen = _frequent_words([*nouns, *adjectives], excluded, token_count, tokenizer)
        chosen = chosen[: evaluation.probe_words_per_token_count]
        positives = [
            ProbeWord(token_count, "real-word", word, _ids(ids), "") for word, ids in chosen
        ]
        if token_count == 1:
            negatives = _untrained_tokens(words, tokenizer)
        else:
            negatives = _swapped_nonwords(chosen, real_words, tokenizer, random_numbers)
        probe_words += _split(positives, random_numbers) + _split(negatives, random_numbers)
    write_rows(output_paths[0], ProbeWord, probe_words)
    normalization = [
        NormalizationWord("known-noun", token_count, word, _ids(ids))
        for token_count, known in known_nouns.items()
        for word, ids in known
    ]
    unsplittable = []
    for noun in (word for word in words if word.role == "noun"):
        for token_count in (2, 3):
            try:
                pieces = split_into_tokens(noun.word, token_count, tokenizer)
            except ValueError:
                unsplittable.append(f"{noun.word}（{token_count}トークン）")
                continue
            token_ids = tokenizer.split_word_token_ids(pieces)
            normalization.append(
                NormalizationWord(
                    "split-noun", token_count, "|".join(pieces), _ids(token_ids), noun.word
                )
            )
    write_rows(output_paths[1], NormalizationWord, normalization)
    if unsplittable:
        print(
            f"各部分が1トークンになるように分けられない名詞が {len(unsplittable)} 組あります"
            f"（{', '.join(unsplittable)}）。その組の意味の正規化は、値なしになります。"
        )


def _frequent_words(
    candidates: list[NounCandidate | AdjectiveCandidate],
    excluded: set[str],
    token_count: int,
    tokenizer: WordTokenizer,
) -> list[tuple[str, list[int]]]:
    """候補のうち、除く語でなく、先頭スペース付きで token_count トークンの語（頻度の高い順）。"""
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate.word] = max(counts.get(candidate.word, 0), candidate.estimated_pile_count)
    chosen = []
    for word in sorted(counts, key=lambda word: (-counts[word], word)):
        token_ids = tokenizer.word_token_ids(word)
        if word not in excluded and len(token_ids) == token_count:
            chosen.append((word, token_ids))
    return chosen


def _untrained_tokens(words: list[SelectedWord], tokenizer: WordTokenizer) -> list[ProbeWord]:
    """未知語に使わなかった、使われていないトークン番号（学習していない新しいトークン）。"""
    unknown = {
        int(word.token_ids) for word in words if word.role == "unknown" and word.token_count == 1
    }
    first = tokenizer.unused_token_ids(1)[0]
    unused = [
        token_id
        for token_id in range(first, tokenizer.number_of_embedding_rows)
        if token_id not in unknown
    ]
    return [
        ProbeWord(1, "untrained-token", f"<untrained_{number:02d}>", str(token_id), "")
        for number, token_id in enumerate(unused, start=1)
    ]


def split_into_tokens(word: str, count: int, tokenizer: WordTokenizer) -> list[str]:
    """先頭スペース付きの語を、各部分が1トークンになる count 個の部分に分ける。

    分け方が複数あれば、部分の長さが最もそろうもの（同じなら、前で分けるもの）。
    """
    best = None
    for cuts in itertools.combinations(range(1, len(word)), count - 1):
        bounds = [0, *cuts, len(word)]
        pieces = [word[start:end] for start, end in itertools.pairwise(bounds)]
        pieces[0] = " " + pieces[0]
        if all(len(tokenizer.encode([piece])) == 1 for piece in pieces):
            spread = max(map(len, pieces)) - min(map(len, pieces))
            if best is None or spread < best[0]:
                best = (spread, pieces)
    if best is None:
        raise ValueError(f"{word!r} を、各部分が1トークンの {count} 個に分けられません。")
    return best[1]


def _swapped_nonwords(
    chosen: list[tuple[str, list[int]]],
    real_words: set[str],
    tokenizer: WordTokenizer,
    random_numbers: random.Random,
) -> list[ProbeWord]:
    """実在語を無作為に組にし、2つ目以降のトークンを入れ替えた非語（実在語と同じ数まで）。"""
    real_sequences = {tuple(ids) for _, ids in chosen}
    wordnet = load_wordnet()
    nonwords: dict[str, ProbeWord] = {}
    for _ in range(PAIRING_ROUNDS):
        if len(nonwords) >= len(chosen):
            break
        order = list(range(len(chosen)))
        random_numbers.shuffle(order)
        for first, second in zip(order[0::2], order[1::2], strict=False):
            pairs = ((chosen[first][1], chosen[second][1]), (chosen[second][1], chosen[first][1]))
            for head, tail in pairs:
                ids = [head[0], *tail[1:]]
                text = tokenizer.decode(ids).strip()
                usable = (
                    tuple(ids) not in real_sequences
                    and text.lower() not in real_words
                    and not wordnet.synsets(text.lower())
                    and tokenizer.word_token_ids(text) == ids
                )
                if usable and len(nonwords) < len(chosen):
                    nonwords.setdefault(text, ProbeWord(len(ids), "nonword", text, _ids(ids), ""))
    return list(nonwords.values())


def _split(rows: list[ProbeWord], random_numbers: random.Random) -> list[ProbeWord]:
    """HELD_OUT_SHARE を確認用に、残りを学習用にする。"""
    order = list(range(len(rows)))
    random_numbers.shuffle(order)
    held_out = set(order[: round(HELD_OUT_SHARE * len(rows))])
    for index, row in enumerate(rows):
        row.split = "held-out" if index in held_out else "training"
    return rows


def _ids(token_ids: list[int]) -> str:
    return " ".join(map(str, token_ids))
