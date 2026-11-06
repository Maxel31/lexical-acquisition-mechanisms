"""自然な語の用例から、課題の例とプローブの学習に使う文を取り出す
（evaluation-data extract-natural-word-sentences）。

自然な語ごとに、判定に合格した用例を鍵の順に見て、語の最初の出現を含む1文を取り出す。
- 文の区切りは、文の終わりの記号（. ! ?。後ろの引用符や括弧を含む）の後の空白と、改行。
- 用例の端で切れた文（前か後ろに区切りがない文。用例の末尾の記号は、元の文の途中（例: "3.5 kg" の
  "3."）でありうるので、文の終わりにしない）、大文字で始まらない文、文の終わりの記号で終わらない文、
  語の数が MINIMUM_WORDS〜MAXIMUM_WORDS でない文は使わない。
語ごとに evaluation.sentences_per_natural_word 文まで取り、
data/evaluation/natural_word_sentences.csv に書く。語ごとの文の数は natural_word_sentence_counts.csv
に書く。文が足りない語（passages finalize で外した、用例のない語を含む）は外して続ける。外すと、
自然な語の数が下限（word_selection.minimum_natural_nouns_per_category・minimum_natural_adjectives）を
下回るときは、書いた後に止まる（用例を増やすか、語を選び直す）。
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pyarrow.compute as compute
import pyarrow.parquet as parquet

from lexical_acquisition_mechanisms import (
    passage_finalization,
    passage_shortage,
    word_finalization,
    word_selection,
)
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.settings import Settings
from lexical_acquisition_mechanisms.word_selection import SelectedWord
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

COMMAND = "evaluation-data extract-natural-word-sentences"
OUTPUT_DIRECTORY = Path("data/evaluation")
SENTENCES_PATH = OUTPUT_DIRECTORY / "natural_word_sentences.csv"
COUNTS_PATH = OUTPUT_DIRECTORY / "natural_word_sentence_counts.csv"
MINIMUM_WORDS = 5
MAXIMUM_WORDS = 40
# 文の区切り（文の終わりの記号と、後ろの引用符・括弧の後の空白か、改行）
BOUNDARY = re.compile(r"[.!?][\"')\]]*\s+|\n+")
SENTENCE_END = re.compile(r"[.!?][\"')\]]*(?=\s)|\n")


@dataclass
class NaturalWordSentence:
    word: str
    role: str
    category: str
    sentence: str
    # 文の中の、語の最初の文字の位置（その前は空白）
    word_start: int


@dataclass
class NaturalWordSentenceCount:
    word: str
    role: str
    passages: int
    sentences: int


def extract_natural_word_sentences(settings: Settings, arguments: argparse.Namespace) -> None:
    step = Step(
        COMMAND,
        inputs={
            word_finalization.OUTPUT_PATH: word_finalization.COMMAND,
            passage_finalization.PASSAGES_PATH: passage_finalization.COMMAND,
        },
        outputs=[SENTENCES_PATH, COUNTS_PATH],
        settings={
            "sentences_per_natural_word": settings.evaluation.sentences_per_natural_word,
            "word_selection": settings.word_selection.model_dump(
                include={"minimum_natural_nouns_per_category", "minimum_natural_adjectives"}
            ),
            "tokenizer": next(iter(settings.models.values())).model_dump(),
        },
    )
    run_unless_unchanged(
        step,
        lambda: build_natural_word_sentences(
            settings,
            word_finalization.OUTPUT_PATH,
            passage_finalization.PASSAGES_PATH,
            (SENTENCES_PATH, COUNTS_PATH),
        ),
        arguments.recompute,
    )


def build_natural_word_sentences(
    settings: Settings, words_path: Path, passages_path: Path, output_paths: tuple[Path, Path]
) -> None:
    sentences_path, counts_path = output_paths
    tokenizer = WordTokenizer.from_model(next(iter(settings.models.values())))
    words = [
        word for word in read_rows(words_path, SelectedWord) if word.role.startswith("natural-")
    ]
    table = parquet.read_table(passages_path)
    wanted = settings.evaluation.sentences_per_natural_word
    sentences = []
    counts = []
    for word in words:
        passages = table.filter(compute.equal(table.column("word"), word.word)).sort_by("key")
        found = []
        for passage in passages.to_pylist():
            if len(found) == wanted:
                break
            extracted = _sentence(
                tokenizer, passage["token_ids"], passage["occurrence_positions"][0], word
            )
            if extracted is not None:
                found.append(NaturalWordSentence(word.word, word.role, word.category, *extracted))
        sentences += found
        counts.append(NaturalWordSentenceCount(word.word, word.role, passages.num_rows, len(found)))
    short = [count.word for count in counts if count.sentences < wanted]
    problems = passage_shortage.natural_word_problems(words, set(short), settings)
    if short and not problems:
        print(
            f"文が {wanted} 文に届かなかった自然な語を外して続けます（{len(short)} 語: {short}）。"
        )
        sentences = [sentence for sentence in sentences if sentence.word not in short]
    write_rows(sentences_path, NaturalWordSentence, sentences)
    write_rows(counts_path, NaturalWordSentenceCount, counts)
    if short and problems:
        raise CommandStopped(
            f"文が {wanted} 文に届かなかった自然な語があります（{len(short)} 語: {short[:10]}。"
            f"数は {counts_path}。外すと {', '.join(problems)} で、下限を下回ります）。"
            "その語を data/words/excluded_words.csv に足すか、"
            "settings.yaml の evaluation.passages_per_natural_word を増やしてください。どちらも、"
            "自然な語を選び直すことになるので、words select から実行し直してください。",
            rerun_from=word_selection.COMMAND,
        )


def _sentence(
    tokenizer: WordTokenizer, token_ids: list[int], position: int, word: SelectedWord
) -> tuple[str, int] | None:
    """用例から、位置 position の語を含む1文と、文の中の語の位置。使えなければ None。"""
    before = tokenizer.decode(token_ids[:position])
    text = before + " " + word.word + tokenizer.decode(token_ids[position + word.token_count :])
    offset = len(before) + 1
    starts = [match.end() for match in BOUNDARY.finditer(text, 0, offset)]
    end = SENTENCE_END.search(text, offset + len(word.word))
    if not starts or end is None or end.group().startswith("\n"):
        return None
    sentence = text[starts[-1] : end.end()]
    if (
        "\n" in sentence
        or not sentence[:1].isupper()
        or not MINIMUM_WORDS <= len(sentence.split()) <= MAXIMUM_WORDS
    ):
        return None
    return sentence, offset - starts[-1]
