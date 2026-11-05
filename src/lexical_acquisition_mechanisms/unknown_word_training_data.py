"""未知語の獲得の学習データ（training-data build --experiment unknown-word-acquisition）。

割り当て表のとおり、run ごとに、各未知語に割り当てた名詞の学習用の用例を、鍵の小さい順に露出回数ぶん
取り、名詞を未知語に置き換えて（passage_replacement）、基データに差し込む（training_layout）。
置き換えられなかった用例は飛ばし、次の用例を使う。1トークンの未知語は、トークン番号で差し込む。
差し込む順番は、seed から決まる並べ替えにする（各未知語の用例が、学習の全体に散らばるように）。
対照の run（control）は、基データだけにする。
run ごとに data/training/unknown-word-acquisition/<run>/ に次を書く。
- tokens.npy: 学習データのトークン列（4,096トークンの系列を並べたもの）
- passage_starts.npy: 差し込んだ用例の始まりの位置（差し込んだ用例だけの損失を記録するのに使う）
- inserted.parquet: 差し込んだ用例ごとの、未知語・名詞・元の用例・置き換えの記録
全ての run の確認の結果（語ごとの差し込んだ件数・置き換えた出現の数・総トークン数）を
checks.csv に、無作為に選んだ用例を文字に戻したものを decoded_passages.txt に書く。
"""

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as arrow
import pyarrow.compute as compute
import pyarrow.parquet as parquet

from lexical_acquisition_mechanisms import (
    base_data,
    passage_finalization,
    unknown_word_assignments,
    word_finalization,
)
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.passage_replacement import (
    PassageReplacer,
    Replacement,
    ReplacementError,
)
from lexical_acquisition_mechanisms.settings import Settings
from lexical_acquisition_mechanisms.training_layout import insert_passages, plan_insertions
from lexical_acquisition_mechanisms.unknown_word_assignments import UnknownWordAssignment
from lexical_acquisition_mechanisms.word_selection import SelectedWord
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

EXPERIMENT = unknown_word_assignments.EXPERIMENT
COMMAND = f"training-data build --experiment {EXPERIMENT}"
OUTPUT_DIRECTORY = Path("data/training") / EXPERIMENT
CONTROL_RUN = "control"
DECODED_PASSAGES_PER_RUN = 5


@dataclass
class TrainingDataCheck:
    run: str
    unknown_word: str
    noun: str
    # 割り当てた露出回数と、差し込んだ用例の数・置き換えた出現の数
    exposure: int
    inserted_passages: int
    replaced_occurrences: int
    # 置き換えられずに飛ばした用例の数
    skipped_passages: int
    # この run の学習データの総トークン数
    total_tokens: int


def build_unknown_word_training_data(settings: Settings, arguments: argparse.Namespace) -> None:
    base_path = base_data.output_paths()["base"]
    step = Step(
        COMMAND,
        inputs={
            word_finalization.OUTPUT_PATH: word_finalization.COMMAND,
            passage_finalization.PASSAGES_PATH: passage_finalization.COMMAND,
            unknown_word_assignments.ASSIGNMENTS_PATH: unknown_word_assignments.COMMAND,
            base_path: base_data.COMMAND,
        },
        outputs=[OUTPUT_DIRECTORY],
        settings={
            "tokens_per_passage": settings.passages.tokens_per_passage,
            # 学習は、この長さの系列に分ける（用例が系列の区切りをまたがないように置く）
            "sequence_length": settings.training.sequence_length,
            "seed": settings.words.seed,
            "tokenizer": next(iter(settings.models.values())).model_dump(),
        },
    )
    run_unless_unchanged(
        step,
        lambda: write_unknown_word_training_data(
            settings,
            word_finalization.OUTPUT_PATH,
            passage_finalization.PASSAGES_PATH,
            unknown_word_assignments.ASSIGNMENTS_PATH,
            base_path,
            OUTPUT_DIRECTORY,
        ),
        arguments.recompute,
    )


def write_unknown_word_training_data(
    settings: Settings,
    words_path: Path,
    passages_path: Path,
    assignments_path: Path,
    base_path: Path,
    output_directory: Path,
) -> None:
    tokenizer = WordTokenizer.from_model(next(iter(settings.models.values())))
    words = {word.word: word for word in read_rows(words_path, SelectedWord)}
    assignments = read_rows(assignments_path, UnknownWordAssignment)
    passages = training_passages(passages_path, "noun")
    base = np.asarray(np.load(base_path, mmap_mode="r")).reshape(-1)
    replacer = PassageReplacer(tokenizer, settings.passages.tokens_per_passage)
    runs = unknown_word_assignments.run_names()
    number_of_passages = sum(row.exposure for row in assignments if row.run == runs[0])
    layout = plan_insertions(
        base,
        tokenizer.end_of_text_id,
        number_of_passages,
        settings.passages.tokens_per_passage + 1,
        settings.training.sequence_length,
    )
    checks = []
    decoded = []
    length = settings.passages.tokens_per_passage
    for run_index, run in enumerate(runs):
        rows = []
        run_checks = []
        for assignment in (row for row in assignments if row.run == run):
            inserted, skipped = _replaced_passages(
                assignment, words, passages, replacer, assignment.exposure
            )
            rows += inserted
            replaced = sum(len(row["records"]) for row in inserted)
            run_checks.append(
                TrainingDataCheck(
                    run,
                    assignment.unknown_word,
                    assignment.noun,
                    assignment.exposure,
                    len(inserted),
                    replaced,
                    skipped,
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
        write_run_files(output_directory / run, tokens, layout.output_starts, _record_table(rows))
        chosen = random.Random(settings.words.seed).sample(
            range(len(rows)), min(DECODED_PASSAGES_PER_RUN, len(rows))
        )
        for index in chosen:
            start = layout.output_starts[index]
            text = tokenizer.decode(tokens[start : start + length].tolist())
            row = rows[index]
            decoded.append(f"## {run}: {row['noun']} → {row['unknown_word']}\n{text}\n")
    sequence_length = settings.training.sequence_length
    control = base[: len(base) // sequence_length * sequence_length]
    write_run_files(
        output_directory / CONTROL_RUN, control, np.zeros(0, dtype=np.int64), _record_table([])
    )
    totals = {check.total_tokens for check in checks}
    if len(totals) != 1:
        raise RuntimeError(f"run の総トークン数がそろっていません: {sorted(totals)}")
    write_rows(output_directory / "checks.csv", TrainingDataCheck, checks)
    (output_directory / "decoded_passages.txt").write_text("\n".join(decoded), encoding="utf-8")


def training_passages(passages_path: Path, role: str) -> dict[str, dict[str, object]]:
    """役割（noun・adjective）の語ごとの、学習用の用例（鍵の小さい順のトークン列と出現の位置）。"""
    table = parquet.read_table(passages_path)
    table = table.filter(
        compute.and_(
            compute.equal(table.column("split"), "training"),
            compute.equal(table.column("role"), role),
        )
    )
    table = table.sort_by([("word", "ascending"), ("key", "ascending")])
    passages: dict[str, dict[str, object]] = {}
    for word in table.column("word").unique().to_pylist():
        rows = table.filter(compute.equal(table.column("word"), word))
        token_ids = rows.column("token_ids").combine_chunks().flatten().to_numpy()
        passages[word] = {
            "token_ids": token_ids.reshape(rows.num_rows, -1),
            "positions": rows.column("occurrence_positions").to_pylist(),
            "sources": rows.select(["shard", "row", "start"]).to_pylist(),
        }
    return passages


def _replaced_passages(
    assignment: UnknownWordAssignment,
    words: dict[str, SelectedWord],
    passages: dict[str, dict[str, object]],
    replacer: PassageReplacer,
    count: int,
) -> tuple[list[dict[str, object]], int]:
    """名詞の用例を鍵の小さい順に、未知語に置き換えて count 件取る（置き換えられないものは飛ばす）。

    飛ばした用例の数も返す。
    """
    if count == 0:
        return [], 0
    noun = words[assignment.noun]
    unknown_word = words[assignment.unknown_word]
    replacement = Replacement(
        tuple(int(token) for token in noun.token_ids.split()),
        tuple(int(token) for token in unknown_word.token_ids.split()),
        unknown_word.word,
        # 1トークンの未知語はトークン番号で差し込むので、冠詞は元のまま残す
        adjust_articles=unknown_word.token_count > 1,
    )
    noun_passages = passages[noun.word]
    inserted = []
    skipped = 0
    for index in range(len(noun_passages["positions"])):
        if len(inserted) == count:
            break
        try:
            replaced = replacer.replace(
                noun_passages["token_ids"][index].tolist(),
                noun_passages["positions"][index],
                replacement,
            )
        except ReplacementError:
            skipped += 1
            continue
        inserted.append(
            {
                "unknown_word": unknown_word.word,
                "noun": noun.word,
                **noun_passages["sources"][index],
                "token_ids": replaced.token_ids,
                "positions": replaced.positions,
                "records": [f"{original} → {new}" for original, new in replaced.records],
            }
        )
    if len(inserted) < count:
        raise RuntimeError(
            f"{noun.word} の学習用の用例が足りません（{len(inserted)} / {count} 件）。"
        )
    return inserted, skipped


def write_run_files(
    directory: Path, tokens: np.ndarray, starts: np.ndarray, records: arrow.Table
) -> None:
    """run の学習データ（トークン列・差し込んだ用例の始まり・差し込んだ用例の記録）を書く。"""
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "tokens.npy", tokens.astype(np.uint32))
    np.save(directory / "passage_starts.npy", np.asarray(starts, dtype=np.int64))
    parquet.write_table(records, directory / "inserted.parquet")


def _record_table(rows: list) -> arrow.Table:
    schema = arrow.schema(
        [
            ("unknown_word", arrow.string()),
            ("noun", arrow.string()),
            ("shard", arrow.int32()),
            ("row", arrow.int32()),
            ("start", arrow.int64()),
            ("positions", arrow.list_(arrow.int32())),
            ("records", arrow.list_(arrow.string())),
        ]
    )
    return arrow.Table.from_pylist(rows, schema=schema)
