"""判定の精度の確認と、用例の書き出しのテスト。"""

from pathlib import Path

import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.passage_collection import CANDIDATE_SCHEMA, CollectionSummary
from lexical_acquisition_mechanisms.passage_finalization import PassageSummary, build_passages
from lexical_acquisition_mechanisms.passage_review import PassageReviewRow
from lexical_acquisition_mechanisms.settings import Settings, load_settings
from lexical_acquisition_mechanisms.word_selection import ExcludedWord, SelectedWord

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
JUDGMENT_SCHEMA = (
    CANDIDATE_SCHEMA.append(arrow.field("split", arrow.string()))
    .append(arrow.field("passed", arrow.bool_()))
    .append(arrow.field("reasons", arrow.list_(arrow.string())))
)


@pytest.fixture
def settings() -> Settings:
    """名詞は学習用 3 件と評価用 1 件、自然な語は 1 件の合格が必要。"""
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    passages = settings.passages.model_copy(update={"evaluation_passages_per_word": 1})
    exposure = settings.unknown_word_acquisition.model_copy(update={"exposure_levels": [0, 3]})
    evaluation = settings.evaluation.model_copy(update={"passages_per_natural_word": 1})
    return settings.model_copy(
        update={
            "passages": passages,
            "unknown_word_acquisition": exposure,
            "evaluation": evaluation,
        }
    )


def judgment(word: str, split: str, row: int, passed: bool) -> dict:
    return {
        "word": word,
        "role": "",
        "shard": 0,
        "row": row,
        "start": 0,
        "key": row / 100,
        "token_ids": [1, 2, 3],
        "occurrence_positions": [1],
        "split": split,
        "passed": passed,
        "reasons": ["stub"],
    }


def finalize(
    tmp_path: Path,
    settings: Settings,
    approved: list[str],
    pasta_training: int = 3,
    pasta_evaluation: int = 1,
    pasta_reached_limit: bool = False,
    natural_adjectives: dict[str, int] | None = None,
):
    """pasta の合格は学習用 pasta_training 件・評価用 pasta_evaluation 件。

    自然な形容詞の合格は、natural_adjectives（語 → 評価用の件数。既定は quiet が 1 件）。
    """
    natural_adjectives = natural_adjectives or {"quiet": 1}
    words = [SelectedWord("pasta", "noun")]
    words += [SelectedWord(word, "natural-adjective") for word in natural_adjectives]
    write_rows(tmp_path / "words.csv", SelectedWord, words)
    collection = [CollectionSummary("pasta", "noun", 4, 10, 10, "", pasta_reached_limit)]
    collection += [
        CollectionSummary(word, "natural-adjective", 1, 2, 2, "", True)
        for word in natural_adjectives
    ]
    write_rows(tmp_path / "collection_summary.csv", CollectionSummary, collection)
    judgments = [judgment("pasta", "training", row, True) for row in range(pasta_training)]
    judgments += [
        judgment("pasta", "evaluation", 60 + row, True) for row in range(pasta_evaluation)
    ]
    judgments += [judgment("pasta", "training", 50, False)]
    for number, (word, passed) in enumerate(natural_adjectives.items()):
        judgments += [
            judgment(word, "evaluation", 70 + number * 10 + row, True) for row in range(passed)
        ]
    table = arrow.Table.from_pylist(judgments, schema=JUDGMENT_SCHEMA)
    parquet.write_table(table, tmp_path / "judgments.parquet")
    review = [
        PassageReviewRow("pasta", "noun", "a dish", 0, row, 0, "...", value)
        for row, value in enumerate(approved)
    ]
    write_rows(tmp_path / "review.csv", PassageReviewRow, review)
    build_passages(
        settings,
        tmp_path / "words.csv",
        tmp_path / "collection_summary.csv",
        tmp_path / "judgments.parquet",
        tmp_path / "review.csv",
        tmp_path / "excluded_words.csv",
        (tmp_path / "passages.parquet", tmp_path / "summary.csv"),
    )


def test_passed_passages_are_written(tmp_path: Path, settings: Settings) -> None:
    finalize(tmp_path, settings, ["1", "1"])

    passages = parquet.read_table(tmp_path / "passages.parquet")
    assert passages.num_rows == 5
    assert "passed" not in passages.column_names
    summary = {row.word: row for row in read_rows(tmp_path / "summary.csv", PassageSummary)}
    assert (summary["pasta"].precision, summary["pasta"].training) == ("1.00", 3)
    assert (summary["quiet"].precision, summary["quiet"].evaluation) == ("", 1)


def test_imprecise_judgments_stop(tmp_path: Path, settings: Settings) -> None:
    with pytest.raises(CommandStopped, match="pasta: 0.50") as stopped:
        finalize(tmp_path, settings, ["1", "0"])
    assert stopped.value.rerun_from == "passages judge-word-sense"


def test_word_without_enough_passages_is_excluded(tmp_path: Path, settings: Settings) -> None:
    # 外す語の精度（0.50）は確かめない
    with pytest.raises(CommandStopped, match="pasta（training 2 / 3 件）") as stopped:
        finalize(tmp_path, settings, ["1", "0"], pasta_training=2)

    assert stopped.value.rerun_from == "words select"
    excluded = read_rows(tmp_path / "excluded_words.csv", ExcludedWord)
    assert [(row.word, row.role) for row in excluded] == [("pasta", "noun")]


def test_word_without_passed_passages_is_excluded(tmp_path: Path, settings: Settings) -> None:
    # 合格が0件の語は、確認用の CSV に行がない
    with pytest.raises(
        CommandStopped, match="pasta（training 0 / 3 件）, pasta（evaluation 0 / 1 件）"
    ):
        finalize(tmp_path, settings, [], pasta_training=0, pasta_evaluation=0)

    excluded = read_rows(tmp_path / "excluded_words.csv", ExcludedWord)
    assert [(row.word, row.role) for row in excluded] == [("pasta", "noun")]


def test_word_with_more_occurrences_in_the_pile_is_kept(tmp_path: Path, settings: Settings) -> None:
    with pytest.raises(CommandStopped, match="passages.occurrence_margin") as stopped:
        finalize(tmp_path, settings, ["1", "1"], pasta_training=2, pasta_reached_limit=True)

    assert stopped.value.rerun_from == "passages collect"
    assert not (tmp_path / "excluded_words.csv").exists()


def test_natural_words_without_enough_passages_are_dropped(
    tmp_path: Path, settings: Settings
) -> None:
    selection = settings.word_selection.model_copy(update={"minimum_natural_adjectives": 1})
    settings = settings.model_copy(update={"word_selection": selection})

    finalize(tmp_path, settings, ["1", "1"], natural_adjectives={"quiet": 0, "calm": 1})

    # 自然な語が足りないだけなら止まらず、その語の用例を書かない
    passages = parquet.read_table(tmp_path / "passages.parquet")
    assert set(passages.column("word").to_pylist()) == {"pasta", "calm"}
    summary = {row.word: row for row in read_rows(tmp_path / "summary.csv", PassageSummary)}
    assert summary["quiet"].evaluation == 0
    assert not (tmp_path / "excluded_words.csv").exists()


def test_unfilled_rows_stop(tmp_path: Path, settings: Settings) -> None:
    with pytest.raises(CommandStopped, match="1・0 でない行が 1 行あります"):
        finalize(tmp_path, settings, ["1", ""])
