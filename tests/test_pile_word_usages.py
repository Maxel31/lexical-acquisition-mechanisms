"""Pile での語の出現回数と見本のテスト。ダウンロードの代わりに、手元に作った parquet を使う。"""

from pathlib import Path

import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

from lexical_acquisition_mechanisms import pile
from lexical_acquisition_mechanisms.pile import PileDocument
from lexical_acquisition_mechanisms.pile_word_usages import sample_word_usages, scan_documents
from lexical_acquisition_mechanisms.settings import load_settings

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# 用例は 256 トークン、1つの文書から 3 件まで（リポジトリの設定と同じ）
PASSAGES = load_settings(REPOSITORY_ROOT / "settings.yaml").passages.model_copy(
    update={"tokens_per_passage": 256, "maximum_passages_per_document": 3}
)

TEXTS_BY_SHARD = [
    [
        "I ate pasta.\nThe pasta-like dough. Pasta is good.",
        "Two pastas and (pasta) in brackets; a car and a pasta dish.",
    ],
    ["We cooked pasta tonight.", "pasta at the start is not counted.", "More pasta, please."],
]
FILE_NAMES = ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"]


def write_local_shard(file_name: str, directory: Path) -> Path:
    shard = FILE_NAMES.index(file_name)
    directory.mkdir(parents=True)
    path = directory / "shard.parquet"
    parquet.write_table(arrow.table({"text": TEXTS_BY_SHARD[shard]}), path)
    return path


@pytest.fixture(autouse=True)
def local_shard_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pile, "shard_file_names", lambda: FILE_NAMES)


def sample(tmp_path: Path, words: list[str], examples: int, processes: int = 1):
    return sample_word_usages(
        words,
        2,
        examples,
        seed=0,
        passages=PASSAGES,
        number_of_processes=processes,
        download_directory=tmp_path,
        download=write_local_shard,
    )


def test_only_the_standard_form_is_counted(tmp_path: Path) -> None:
    usages = sample(tmp_path, ["pasta", "car", "lamp"], examples=10)

    # 数えるのは " pasta"（直後が英字でない）: 1つ目の断片に3回、2つ目に2回
    assert usages["pasta"].count == 5
    # 1つ目の文書の2回は近いので、1件の用例に数える
    assert usages["pasta"].passage_count == 4
    assert usages["car"].count == 1
    assert usages["lamp"].count == 0
    assert len(usages["pasta"].examples) == 5


def test_examples_do_not_depend_on_other_words_or_processes(tmp_path: Path) -> None:
    alone = sample(tmp_path, ["pasta"], examples=2)
    together = sample(tmp_path, ["pasta", "car"], examples=2, processes=2)

    assert alone["pasta"].example_texts == together["pasta"].example_texts
    assert len(alone["pasta"].examples) == 2
    keys = [key for key, _ in together["pasta"].examples]
    assert keys == sorted(keys)


def test_example_keeps_the_context_with_single_spaces(tmp_path: Path) -> None:
    usages = sample(tmp_path, ["pasta"], examples=10)

    # 見本ごとに、その出現だけを [[ ]] で囲む
    assert "I ate [[pasta]]. The pasta-like dough. Pasta is good." in usages["pasta"].example_texts
    assert "I ate pasta. The [[pasta]]-like dough. Pasta is good." in usages["pasta"].example_texts


def test_passages_are_counted_apart_and_up_to_the_limit_per_document() -> None:
    gap = " filler" * 20
    documents = [
        PileDocument(0, 0, f"a pasta and pasta{gap} pasta{gap} pasta{gap} pasta"),
        PileDocument(0, 1, "one pasta"),
    ]

    usages = scan_documents(
        iter(documents),
        frozenset({"pasta"}),
        examples_per_word=10,
        seed=0,
        characters_per_passage=100,
        maximum_passages_per_document=3,
    )

    assert usages["pasta"].count == 6
    # 1つ目の文書: 近い2回は1件にし、3件で打ち切る。2つ目の文書: 1件
    assert usages["pasta"].passage_count == 4
