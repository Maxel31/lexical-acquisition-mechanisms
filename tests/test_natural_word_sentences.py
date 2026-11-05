"""自然な語の文の取り出しのテスト。"""

from pathlib import Path

import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.natural_word_sentences import (
    NaturalWordSentence,
    NaturalWordSentenceCount,
    build_natural_word_sentences,
)
from lexical_acquisition_mechanisms.passage_collection import CANDIDATE_SCHEMA
from lexical_acquisition_mechanisms.settings import ModelSource, load_settings
from lexical_acquisition_mechanisms.word_selection import SelectedWord
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# 鍵の順。使えるのは1つ目だけ（2つ目は用例の端で切れた文、3つ目は短い文、4つ目は小文字で始まる文、
# 5つ目は用例の末尾で終わる文）
TEXTS = [
    "It was late. We ate hot soup for dinner tonight. Then we left.",
    "ate some soup for dinner. Then we slept.",
    "Short. A soup. End.",
    "It rained. we had soup at home today together. Done.",
    "Then we ate. We had cold soup with bread at noon.",
]


def test_one_whole_sentence_is_taken_from_each_passage(
    tmp_path: Path, tiny_model_source: ModelSource
) -> None:
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    evaluation = settings.evaluation.model_copy(update={"sentences_per_natural_word": 2})
    settings = settings.model_copy(
        update={"models": {"tiny": tiny_model_source}, "evaluation": evaluation}
    )
    tokenizer = WordTokenizer.from_model(tiny_model_source)
    soup = tokenizer.word_token_ids("soup")[0]
    write_rows(
        tmp_path / "words.csv",
        SelectedWord,
        [SelectedWord("soup", "natural-noun", category="food", token_count=1)],
    )
    rows = []
    for index, text in enumerate(TEXTS):
        token_ids = tokenizer.encode([text])
        rows.append(
            {
                "word": "soup",
                "role": "natural-noun",
                "shard": 0,
                "row": index,
                "start": 0,
                "key": index / 10,
                "token_ids": token_ids,
                "occurrence_positions": [token_ids.index(soup)],
                "split": "evaluation",
            }
        )
    schema = CANDIDATE_SCHEMA.append(arrow.field("split", arrow.string()))
    parquet.write_table(arrow.Table.from_pylist(rows, schema=schema), tmp_path / "passages.parquet")
    outputs = (tmp_path / "sentences.csv", tmp_path / "counts.csv")

    # 2文に届かないので止まる（文と数は書く）
    with pytest.raises(CommandStopped, match="2 文に届かなかった自然な語") as stopped:
        build_natural_word_sentences(
            settings, tmp_path / "words.csv", tmp_path / "passages.parquet", outputs
        )

    # 必要な用例の数は、自然な語を選ぶ条件でもあるので、words select から
    assert stopped.value.rerun_from == "words select"
    (sentence,) = read_rows(outputs[0], NaturalWordSentence)
    assert sentence.sentence == "We ate hot soup for dinner tonight."
    assert sentence.sentence[sentence.word_start :].startswith("soup")
    assert sentence.category == "food"
    (count,) = read_rows(outputs[1], NaturalWordSentenceCount)
    assert (count.passages, count.sentences) == (5, 1)
