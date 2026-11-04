"""用例の候補の切り出しのテスト。

ダウンロードの代わりに、手元に作った parquet を使う。
"""

from pathlib import Path

import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

from lexical_acquisition_mechanisms import pile
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.passage_collection import (
    CollectionSummary,
    build_passage_candidates,
)
from lexical_acquisition_mechanisms.passage_replacement import WINDOW_MARGIN
from lexical_acquisition_mechanisms.settings import ModelSource, Settings, load_settings
from lexical_acquisition_mechanisms.word_selection import SelectedWord
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FILLER = " filler" * 30
TEXTS_BY_SHARD = [
    [
        f"I ate pasta.{FILLER} More pasta.{FILLER} pasta again,{FILLER} and pasta!",
        f"A quiet pasta.{FILLER}",
    ],
    [f"We cooked Pasta and pasta tonight.{FILLER}", f"It was quiet and quiet.{FILLER}", "A pasta."],
]
FILE_NAMES = ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"]


class LocalShards:
    """断片をダウンロードする代わりに parquet を書く。

    読んだ断片を覚え、指定した断片では失敗する。
    """

    def __init__(self, failing_shard: int | None = None) -> None:
        self.failing_shard = failing_shard
        self.read_shards: list[int] = []

    def __call__(self, file_name: str, directory: Path) -> Path:
        shard = FILE_NAMES.index(file_name)
        if shard == self.failing_shard:
            raise OSError("ダウンロードに失敗した")
        self.read_shards.append(shard)
        directory.mkdir(parents=True)
        path = directory / "shard.parquet"
        parquet.write_table(arrow.table({"text": TEXTS_BY_SHARD[shard]}), path)
        return path


@pytest.fixture(autouse=True)
def two_local_shards(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pile, "shard_file_names", lambda: FILE_NAMES)
    monkeypatch.setattr(pile, "NUMBER_OF_SHARDS", 2)


@pytest.fixture
def settings(tiny_model_source: ModelSource) -> Settings:
    """名詞は 3 + 1 件、形容詞も同じ数、自然な語は 1 件を必要とし、その2倍を残す。

    用例1つ分は 25 トークン（100 文字）、1つの文書から取るのは 3 件まで。
    """
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    passages = settings.passages.model_copy(
        update={
            "tokens_per_passage": 25,
            "evaluation_passages_per_word": 1,
            "collection_factor": 2.0,
        }
    )
    exposure = settings.unknown_word_acquisition.model_copy(
        update={"exposure_levels": [0, 1, 2, 3]}
    )
    evaluation = settings.evaluation.model_copy(update={"passages_per_natural_word": 1})
    return settings.model_copy(
        update={
            "models": {"tiny": tiny_model_source},
            "passages": passages,
            "unknown_word_acquisition": exposure,
            "evaluation": evaluation,
        }
    )


def write_words(tmp_path: Path, pasta_passages: int = 8) -> Path:
    words = [
        SelectedWord("pasta", "noun", estimated_pile_passages=pasta_passages),
        SelectedWord("quiet", "natural-adjective", estimated_pile_passages=1),
        SelectedWord("<new_01>", "unknown"),
    ]
    path = tmp_path / "words.csv"
    write_rows(path, SelectedWord, words)
    return path


def collect(tmp_path: Path, settings: Settings, download: LocalShards, pasta_passages: int = 8):
    build_passage_candidates(
        settings,
        write_words(tmp_path, pasta_passages),
        tmp_path / "passages",
        number_of_processes=1,
        download_directory=tmp_path / "download",
        download=download,
    )
    return parquet.read_table(tmp_path / "passages" / "occurrences.parquet").to_pylist()


def test_occurrences_are_kept_with_their_contexts(tmp_path: Path, settings: Settings) -> None:
    rows = collect(tmp_path, settings, LocalShards())

    pasta = [row for row in rows if row["word"] == "pasta"]
    # 1つ目の文書の4回のうち3回（上限）と、残りの3つの文書の1回ずつ。鍵の小さい順に並ぶ
    assert len(pasta) == 6
    assert [(row["shard"], row["row"]) for row in pasta].count((0, 0)) == 3
    assert [row["key"] for row in pasta] == sorted(row["key"] for row in pasta)
    for row in pasta:
        assert row["role"] == "noun"
        assert row["context"][row["start"] - row["context_start"] :].startswith("pasta")
    # 自然な語は 1 × 2 件。同じ文書の近い2回は1件に数える
    assert len([row for row in rows if row["word"] == "quiet"]) == 2
    assert not (tmp_path / "passages" / "collection_shards").exists()


def test_windows_contain_only_the_standard_form(tmp_path: Path, settings: Settings) -> None:
    collect(tmp_path, settings, LocalShards())

    summary = {
        row.word: row
        for row in read_rows(tmp_path / "passages" / "collection_summary.csv", CollectionSummary)
    }
    # "Pasta" を含む窓と、短い文書（"A pasta."）は使わない
    assert (summary["pasta"].occurrences, summary["pasta"].candidates) == (6, 4)
    assert "語の標準の形でない形を含む: 1" in summary["pasta"].rejected
    assert "文書が短いか、語の区切りで切れない: 1" in summary["pasta"].rejected
    # quiet の窓のうち、管理対象の名詞 pasta を含むものは使わない
    assert (summary["quiet"].candidates, summary["quiet"].rejected) == (
        1,
        "ほかの管理対象の語を含む: 1",
    )
    tokenizer = WordTokenizer.from_model(settings.models["tiny"])
    pasta_id = tokenizer.word_token_ids("pasta")[0]
    candidates = parquet.read_table(tmp_path / "passages" / "candidates.parquet").to_pylist()
    for candidate in candidates:
        # 用例の長さ（25）より、置き換えで短くなる分だけ長く切り出す
        assert len(candidate["token_ids"]) == 25 + WINDOW_MARGIN
        if candidate["word"] == "pasta":
            assert candidate["occurrence_positions"]
            assert all(
                candidate["token_ids"][position] == pasta_id
                for position in candidate["occurrence_positions"]
            )
    quiet = next(candidate for candidate in candidates if candidate["word"] == "quiet")
    assert len(quiet["occurrence_positions"]) == 2


def test_stopped_collection_continues_from_the_saved_shards(
    tmp_path: Path, settings: Settings
) -> None:
    with pytest.raises(OSError, match="ダウンロードに失敗した"):
        collect(tmp_path, settings, LocalShards(failing_shard=1))

    download = LocalShards()
    rows = collect(tmp_path, settings, download)

    assert download.read_shards == [1]
    assert len([row for row in rows if row["word"] == "pasta"]) == 6


def test_word_rarer_than_estimated_stops(tmp_path: Path, settings: Settings) -> None:
    # 見積もりが実際より多いと、鍵の上限が小さくなり、残す数（8件）に届かない
    with pytest.raises(CommandStopped, match="残す数に届かなかった語があります（\\['pasta'\\]）"):
        collect(tmp_path, settings, LocalShards(), pasta_passages=10_000)
