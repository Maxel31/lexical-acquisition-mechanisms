"""語義の追加の学習データ（training-data build --experiment sense-addition）。

- 注入プールは、名詞用（名詞72語 × K 件）と形容詞用（形容詞72語 × K 件）に分ける（K は
  sense_addition.passages_per_added_sense）。同カテゴリ・別カテゴリの run と名詞の対照の run は
  名詞のプールを、形容詞の run と形容詞の対照の run は形容詞のプールを差し込む。
- 語ごとの K 件は、学習用の用例を鍵の小さい順に見て、その語を割り当てる全てのターゲット語に置き換え
  られるものから取る（同じ用例を、全ての run で使うため）。
- 各 run では、割り当ての表のとおり、割り当てのある donor の用例だけを、donor をターゲット語に
  置き換えて差し込み、割り当てのない donor の用例は元のまま差し込む。
- 差し込む位置と、プールの用例を差し込む順番は、11の run で共通にする（違うのは中身だけ）。
出力の形は、未知語の獲得の学習データと同じ（run ごとのトークン列・用例の始まり・記録、checks.csv、
decoded_passages.txt）。ターゲット語に置き換えた用例に donor が残っていれば、書いた後に止まる。
"""

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import pyarrow as arrow

from lexical_acquisition_mechanisms import (
    base_data,
    passage_finalization,
    sense_addition_assignments,
    word_finalization,
)
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.passage_replacement import (
    PassageReplacer,
    Replacement,
    ReplacementError,
)
from lexical_acquisition_mechanisms.sense_addition_assignments import SenseAdditionAssignment
from lexical_acquisition_mechanisms.settings import Settings
from lexical_acquisition_mechanisms.training_layout import (
    insert_passages,
    load_base_tokens,
    plan_insertions,
)
from lexical_acquisition_mechanisms.unknown_word_training_data import (
    remaining_occurrences,
    stop_if_words_remain,
    training_passages,
    write_run_files,
)
from lexical_acquisition_mechanisms.word_selection import SelectedWord
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

EXPERIMENT = sense_addition_assignments.EXPERIMENT
COMMAND = f"training-data build --experiment {EXPERIMENT}"
OUTPUT_DIRECTORY = Path("data/training") / EXPERIMENT
DECODED_PASSAGES_PER_RUN = 5
RECORD_SCHEMA = arrow.schema(
    [
        ("donor", arrow.string()),
        ("target", arrow.string()),
        ("shard", arrow.int32()),
        ("row", arrow.int32()),
        ("start", arrow.int64()),
        ("positions", arrow.list_(arrow.int32())),
        ("records", arrow.list_(arrow.string())),
    ]
)


@dataclass
class SenseAdditionDataCheck:
    run: str
    donor: str
    # 置き換え先のターゲット語（割り当てがなければ空）
    target: str
    inserted_passages: int
    replaced_occurrences: int
    # 差し込んだ用例に残った donor（標準の形）の出現の数（ターゲット語がなければ、元のままの出現）
    remaining_occurrences: int
    total_tokens: int


def pool_of(run: str) -> str:
    """run が差し込むプール（noun・adjective）。"""
    return "adjective" if run.startswith("adjective") else "noun"


def build_sense_addition_training_data(settings: Settings, arguments: argparse.Namespace) -> None:
    base_path = base_data.output_paths()["base"]
    step = Step(
        COMMAND,
        inputs={
            word_finalization.OUTPUT_PATH: word_finalization.COMMAND,
            passage_finalization.PASSAGES_PATH: passage_finalization.COMMAND,
            sense_addition_assignments.ASSIGNMENTS_PATH: sense_addition_assignments.COMMAND,
            base_path: base_data.COMMAND,
        },
        outputs=[OUTPUT_DIRECTORY],
        settings={
            "passages_per_added_sense": settings.sense_addition.passages_per_added_sense,
            "tokens_per_passage": settings.passages.tokens_per_passage,
            "sequence_length": settings.training.sequence_length,
            "number_of_base_sequences_used": settings.training.number_of_base_sequences_used,
            "seed": settings.words.seed,
            "tokenizer": next(iter(settings.models.values())).model_dump(),
        },
    )
    run_unless_unchanged(
        step,
        lambda: write_sense_addition_training_data(
            settings,
            (
                word_finalization.OUTPUT_PATH,
                passage_finalization.PASSAGES_PATH,
                sense_addition_assignments.ASSIGNMENTS_PATH,
                base_path,
            ),
            OUTPUT_DIRECTORY,
        ),
        arguments.recompute,
    )


def write_sense_addition_training_data(
    settings: Settings, input_paths: tuple[Path, Path, Path, Path], output_directory: Path
) -> None:
    words_path, passages_path, assignments_path, base_path = input_paths
    tokenizer = WordTokenizer.from_model(next(iter(settings.models.values())))
    words = {word.word: word for word in read_rows(words_path, SelectedWord)}
    assignments = read_rows(assignments_path, SenseAdditionAssignment)
    target_of = {(row.run, row.donor): row.target for row in assignments}
    replacer = PassageReplacer(tokenizer, settings.passages.tokens_per_passage)
    count = settings.sense_addition.passages_per_added_sense
    pools = {
        role: _pool(role, words, training_passages(passages_path, role), target_of, replacer, count)
        for role in ("noun", "adjective")
    }
    sizes = {role: len(pool) for role, pool in pools.items()}
    if len(set(sizes.values())) != 1:
        raise RuntimeError(f"名詞と形容詞のプールの大きさが違います: {sizes}")
    base = load_base_tokens(base_path, settings.training.number_of_base_sequences_used)
    layout = plan_insertions(
        base,
        tokenizer.end_of_text_id,
        sizes["noun"],
        settings.passages.tokens_per_passage + 1,
        settings.training.sequence_length,
    )
    order = list(range(sizes["noun"]))
    random.Random(settings.words.seed).shuffle(order)
    checks = []
    decoded = []
    continues_word = tokenizer.word_boundaries().starts_with_letter
    for run in sense_addition_assignments.run_names():
        rows = [
            _inserted(passage, target_of.get((run, passage["donor"]), ""), words, replacer)
            for passage in pools[pool_of(run)]
        ]
        rows = [rows[index] for index in order]
        remaining = {
            donor: remaining_occurrences(
                [row for row in rows if row["donor"] == donor], words[donor], continues_word
            )
            for donor in sorted({row["donor"] for row in rows})
        }
        tokens = insert_passages(
            base, layout, [row.pop("token_ids") for row in rows], tokenizer.end_of_text_id
        )
        write_run_files(
            output_directory / run,
            tokens,
            layout.output_starts,
            arrow.Table.from_pylist(rows, schema=RECORD_SCHEMA),
        )
        for donor in sorted({row["donor"] for row in rows}):
            of_donor = [row for row in rows if row["donor"] == donor]
            checks.append(
                SenseAdditionDataCheck(
                    run,
                    donor,
                    of_donor[0]["target"],
                    len(of_donor),
                    sum(len(row["records"]) for row in of_donor),
                    remaining[donor],
                    len(tokens),
                )
            )
        replaced = [index for index, row in enumerate(rows) if row["target"]]
        for index in random.Random(settings.words.seed).sample(
            replaced, min(DECODED_PASSAGES_PER_RUN, len(replaced))
        ):
            start = layout.output_starts[index]
            text = tokenizer.decode(
                tokens[start : start + settings.passages.tokens_per_passage].tolist()
            )
            decoded.append(f"## {run}: {rows[index]['donor']} → {rows[index]['target']}\n{text}\n")
    write_rows(output_directory / "checks.csv", SenseAdditionDataCheck, checks)
    (output_directory / "decoded_passages.txt").write_text("\n".join(decoded), encoding="utf-8")
    stop_if_words_remain(
        [
            (check.run, check.donor)
            for check in checks
            if check.target and check.remaining_occurrences
        ],
        output_directory / "checks.csv",
        COMMAND,
    )


def _pool(
    role: str,
    words: dict[str, SelectedWord],
    passages: dict[str, dict[str, object]],
    target_of: dict[tuple[str, str], str],
    replacer: PassageReplacer,
    count: int,
) -> list[dict[str, object]]:
    """役割の語ごとに、割り当てる全てのターゲット語に置き換えられる用例を、鍵の順に count 件。"""
    pool = []
    for donor in sorted(word.word for word in words.values() if word.role == role):
        targets = sorted(
            {target for (_, assigned), target in target_of.items() if assigned == donor}
        )
        replacements = [_replacement(donor, target, words) for target in targets]
        taken = 0
        for index in range(len(passages[donor]["positions"])):
            if taken == count:
                break
            token_ids = passages[donor]["token_ids"][index].tolist()
            positions = passages[donor]["positions"][index]
            try:
                for replacement in replacements:
                    replacer.replace(token_ids, positions, replacement)
            except ReplacementError:
                continue
            pool.append(
                {
                    "donor": donor,
                    "token_ids": token_ids,
                    "positions": positions,
                    **passages[donor]["sources"][index],
                }
            )
            taken += 1
        if taken < count:
            raise RuntimeError(f"{donor} の学習用の用例が足りません（{taken} / {count} 件）。")
    return pool


def _replacement(donor: str, target: str, words: dict[str, SelectedWord]) -> Replacement:
    return Replacement(
        tuple(int(token) for token in words[donor].token_ids.split()),
        tuple(int(token) for token in words[target].token_ids.split()),
        target,
        adjust_articles=True,
    )


def _inserted(
    passage: dict[str, object],
    target: str,
    words: dict[str, SelectedWord],
    replacer: PassageReplacer,
) -> dict[str, object]:
    """プールの用例を、ターゲット語があれば置き換えた、差し込む用例。"""
    row = {key: passage[key] for key in ("donor", "shard", "row", "start")}
    if not target:
        return {
            **row,
            "target": "",
            "token_ids": list(passage["token_ids"]),
            "positions": list(passage["positions"]),
            "records": [],
        }
    replaced = replacer.replace(
        passage["token_ids"], passage["positions"], _replacement(passage["donor"], target, words)
    )
    records = [f"{original} → {new}" for original, new in replaced.records]
    return {
        **row,
        "target": target,
        "token_ids": replaced.token_ids,
        "positions": replaced.positions,
        "records": records,
    }
