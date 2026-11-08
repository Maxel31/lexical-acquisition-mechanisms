"""2つ目の語義の追加の学習データ（training-data build --experiment second-sense-addition）。

割り当て表（assignments build --experiment second-sense-addition）のとおり、run ごとに、各未知語の
次の2種類の用例を、2つ目の露出回数ぶんずつ取り、名詞を未知語に置き換えて（passage_replacement）、
基データに差し込む（training_layout）。置き換えられなかった用例は飛ばし、次の用例を使う。
- 2つ目の語義（Y）の用例: Y の学習用の用例を、鍵の大きい順に取る（Y を1つ目の語義にした別の
  未知語が、未知語の獲得で鍵の小さい順に使った用例と重ならないように）
- 1つ目の語義（X）の用例: 未知語の獲得で使った X の用例を、鍵の小さい順にもう一度取る（1つ目の
  語義を学習データから消さず、多義として残るかを見るため）
基データは、未知語の獲得で使った系列（training.number_of_base_sequences_used 個。null なら全て）の
直後から、同じ数の系列を使う（同じ文章を2度学習させないため。足りなければ止まる）。
差し込む順番は、seed から決まる並べ替えにする。run ごとに data/training/second-sense-addition/<run>/
に、未知語の獲得と同じ3つのファイル（tokens.npy・passage_starts.npy・inserted.parquet。
inserted.parquet には、どちらの語義の用例かを sense に書く）を書く。確認の結果を checks.csv に、
無作為に選んだ用例を文字に戻したものを decoded_passages.txt に書き、元の名詞が残った用例があれば、
書いた後に止まる。
"""

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as arrow

from lexical_acquisition_mechanisms import (
    base_data,
    passage_finalization,
    second_sense_assignments,
    word_finalization,
)
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.passage_replacement import PassageReplacer
from lexical_acquisition_mechanisms.second_sense_assignments import SecondSenseAssignment
from lexical_acquisition_mechanisms.settings import Settings
from lexical_acquisition_mechanisms.training_layout import (
    insert_passages,
    load_base_tokens,
    plan_insertions,
)
from lexical_acquisition_mechanisms.unknown_word_training_data import (
    remaining_occurrences,
    replaced_passages,
    stop_if_words_remain,
    training_passages,
    write_run_files,
)
from lexical_acquisition_mechanisms.word_selection import SelectedWord
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

EXPERIMENT = second_sense_assignments.EXPERIMENT
COMMAND = f"training-data build --experiment {EXPERIMENT}"
OUTPUT_DIRECTORY = Path("data/training") / EXPERIMENT
DECODED_PASSAGES_PER_RUN = 5
RECORD_SCHEMA = arrow.schema(
    [
        ("unknown_word", arrow.string()),
        # first（1つ目の語義の X の用例）か second（2つ目の語義の Y の用例）か
        ("sense", arrow.string()),
        ("noun", arrow.string()),
        ("shard", arrow.int32()),
        ("row", arrow.int32()),
        ("start", arrow.int64()),
        ("positions", arrow.list_(arrow.int32())),
        ("records", arrow.list_(arrow.string())),
    ]
)


@dataclass
class SecondSenseTrainingDataCheck:
    run: str
    unknown_word: str
    sense: str
    noun: str
    # 差し込む件数（2つ目の露出回数）と、差し込んだ用例の数・置き換えた出現の数
    exposure: int
    inserted_passages: int
    replaced_occurrences: int
    # 置き換えられずに飛ばした用例の数と、置き換えた用例に残った元の名詞の出現の数
    skipped_passages: int
    remaining_occurrences: int
    total_tokens: int


def build_second_sense_training_data(settings: Settings, arguments: argparse.Namespace) -> None:
    base_path = base_data.output_paths()["base"]
    step = Step(
        COMMAND,
        inputs={
            word_finalization.OUTPUT_PATH: word_finalization.COMMAND,
            passage_finalization.PASSAGES_PATH: passage_finalization.COMMAND,
            second_sense_assignments.ASSIGNMENTS_PATH: second_sense_assignments.COMMAND,
            base_path: base_data.COMMAND,
        },
        outputs=[OUTPUT_DIRECTORY],
        settings={
            "tokens_per_passage": settings.passages.tokens_per_passage,
            "sequence_length": settings.training.sequence_length,
            "number_of_base_sequences_used": settings.training.number_of_base_sequences_used,
            "seed": settings.words.seed,
            "tokenizer": next(iter(settings.models.values())).model_dump(),
        },
    )
    run_unless_unchanged(
        step,
        lambda: write_second_sense_training_data(
            settings,
            word_finalization.OUTPUT_PATH,
            passage_finalization.PASSAGES_PATH,
            second_sense_assignments.ASSIGNMENTS_PATH,
            base_path,
            OUTPUT_DIRECTORY,
        ),
        arguments.recompute,
    )


def write_second_sense_training_data(
    settings: Settings,
    words_path: Path,
    passages_path: Path,
    assignments_path: Path,
    base_path: Path,
    output_directory: Path,
) -> None:
    tokenizer = WordTokenizer.from_model(next(iter(settings.models.values())))
    words = {word.word: word for word in read_rows(words_path, SelectedWord)}
    assignments = read_rows(assignments_path, SecondSenseAssignment)
    passages = training_passages(passages_path, "noun")
    base = second_sense_base_tokens(base_path, settings.training.number_of_base_sequences_used)
    replacer = PassageReplacer(tokenizer, settings.passages.tokens_per_passage)
    runs = second_sense_assignments.run_names()
    totals = {
        run: sum(2 * row.second_exposure for row in assignments if row.run == run) for run in runs
    }
    if len(set(totals.values())) != 1:
        raise RuntimeError(f"run ごとの差し込む用例の数がそろっていません: {totals}")
    layout = plan_insertions(
        base,
        tokenizer.end_of_text_id,
        totals[runs[0]],
        settings.passages.tokens_per_passage + 1,
        settings.training.sequence_length,
    )
    continues_word = tokenizer.word_boundaries().starts_with_letter
    length = settings.passages.tokens_per_passage
    checks = []
    decoded = []
    for run_index, run in enumerate(runs):
        rows = []
        run_checks = []
        for assignment in (row for row in assignments if row.run == run):
            for sense, noun, order in [
                ("second", assignment.second_noun, -1),
                ("first", assignment.first_noun, 1),
            ]:
                inserted, skipped = replaced_passages(
                    words[noun],
                    words[assignment.unknown_word],
                    _in_order(passages[noun], order),
                    replacer,
                    assignment.second_exposure,
                )
                rows += [{**row, "sense": sense} for row in inserted]
                run_checks.append(
                    SecondSenseTrainingDataCheck(
                        run,
                        assignment.unknown_word,
                        sense,
                        noun,
                        assignment.second_exposure,
                        len(inserted),
                        sum(len(row["records"]) for row in inserted),
                        skipped,
                        remaining_occurrences(inserted, words[noun], continues_word),
                        0,
                    )
                )
        random.Random(settings.words.seed + run_index).shuffle(rows)
        tokens = insert_passages(
            base, layout, [row.pop("token_ids") for row in rows], tokenizer.end_of_text_id
        )
        for check in run_checks:
            check.total_tokens = len(tokens)
        checks += run_checks
        records = arrow.Table.from_pylist(rows, schema=RECORD_SCHEMA)
        write_run_files(output_directory / run, tokens, layout.output_starts, records)
        chosen = random.Random(settings.words.seed).sample(
            range(len(rows)), min(DECODED_PASSAGES_PER_RUN, len(rows))
        )
        for index in chosen:
            start = layout.output_starts[index]
            text = tokenizer.decode(tokens[start : start + length].tolist())
            row = rows[index]
            decoded.append(
                f"## {run}: {row['noun']} → {row['unknown_word']}（{row['sense']}）\n{text}\n"
            )
    write_rows(output_directory / "checks.csv", SecondSenseTrainingDataCheck, checks)
    (output_directory / "decoded_passages.txt").write_text("\n".join(decoded), encoding="utf-8")
    stop_if_words_remain(
        [(check.run, check.noun) for check in checks if check.remaining_occurrences],
        output_directory / "checks.csv",
        COMMAND,
    )


def second_sense_base_tokens(path: Path, used: int | None) -> np.ndarray:
    """未知語の獲得で使った系列の直後から、同じ数の系列をつなげたトークン列。"""
    available = len(np.load(path, mmap_mode="r"))
    if used is None or 2 * used > available:
        raise CommandStopped(
            "2つ目の語義の追加に使う基データ（未知語の獲得で使った系列の直後の、同じ数の系列）が"
            f"足りません（基データの系列 {available} 個、未知語の獲得で使った系列 "
            f"{used or available} 個）。settings.yaml の training.number_of_base_sequences_used を"
            "減らすか、基データを増やしてください。",
            rerun_from=COMMAND,
        )
    return load_base_tokens(path, used, start=used)


def _in_order(noun_passages: dict[str, object], order: int) -> dict[str, object]:
    """名詞の用例を、鍵の小さい順（order = 1）か大きい順（order = -1）に並べたもの。"""
    return {key: value[::order] for key, value in noun_passages.items()}
