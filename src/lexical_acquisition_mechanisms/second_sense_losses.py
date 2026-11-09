"""2つ目の語義の追加の各 run で、1つ目と2つ目の語義の用例での未知語の損失を測る
（second-sense-addition measure-losses --model）。

名詞ごとに、学習に使っていない評価用の用例（鍵の小さい順）を PASSAGES_PER_NOUN 件取り、X の
用例の X と、Y の用例の Y を未知語に置き換えて読ませる。未知語の驚き度（語のトークンの −log p の
和。用例の中の出現で平均）を、measure-context-effects と同じ重みで測る。
- 1つ目の語義の用例で驚き度が上がれば、1つ目の語義を忘れかけている。
- 2つ目の語義の用例で下がれば、2つ目の語義が自然な文章でも入った。
結果は outputs/results/second-sense-addition/<モデル>/ に書く。
- losses.parquet: run・重み・未知語・語義（first・second）・用例ごとの驚き度
- losses.csv: 種類 × 2つ目の露出回数 × 語義 × 重みごとの、未知語ごとの平均の平均
"""

import argparse
import gc
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as arrow
import pyarrow.compute as compute
import pyarrow.parquet as parquet
import torch

from lexical_acquisition_mechanisms import (
    passage_finalization,
    second_sense_assignments,
    word_finalization,
)
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.passage_replacement import PassageReplacer
from lexical_acquisition_mechanisms.run_results import RunResults
from lexical_acquisition_mechanisms.second_sense_assignments import SecondSenseAssignment
from lexical_acquisition_mechanisms.second_sense_context_effects import (
    RESULTS_DIRECTORY,
    weight_directories,
    weight_order,
)
from lexical_acquisition_mechanisms.settings import ModelSource, Settings
from lexical_acquisition_mechanisms.unknown_word_indicators import checkpoint_states
from lexical_acquisition_mechanisms.unknown_word_training_data import replaced_passages
from lexical_acquisition_mechanisms.word_selection import SelectedWord
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

# 名詞ごとの評価用の用例の数（置き換えられない用例を飛ばすので、2 倍まで読む）
PASSAGES_PER_NOUN = 50
BATCH_SIZE = 16


@dataclass
class LossSummary:
    kind: str
    second_exposure: int
    sense: str
    weights: str
    surprisal: float
    words: int


def output_paths(model: str) -> tuple[Path, Path]:
    directory = RESULTS_DIRECTORY / model
    return directory / "losses.parquet", directory / "losses.csv"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="settings.yaml の models の名前")


def measure_losses(settings: Settings, arguments: argparse.Namespace) -> None:
    model = arguments.model
    command = f"second-sense-addition measure-losses --model {model}"
    directories = weight_directories(model, command)
    input_paths = (
        word_finalization.OUTPUT_PATH,
        second_sense_assignments.ASSIGNMENTS_PATH,
        passage_finalization.PASSAGES_PATH,
    )
    step = Step(
        command,
        inputs=dict(
            zip(
                input_paths,
                (
                    word_finalization.COMMAND,
                    second_sense_assignments.COMMAND,
                    passage_finalization.COMMAND,
                ),
                strict=True,
            )
        ),
        outputs=list(output_paths(model)),
        # 学習した重みには処理の記録がないので、ファイルの大きさと更新時刻を設定として記録する
        settings={
            "checkpoints": checkpoint_states(directories.values()),
            "passages_per_noun": PASSAGES_PER_NOUN,
            "tokens_per_passage": settings.passages.tokens_per_passage,
        },
    )
    replacer = PassageReplacer(
        WordTokenizer.from_model(settings.models[model]), settings.passages.tokens_per_passage
    )
    results = RunResults(step, arguments.recompute)
    run_unless_unchanged(
        step,
        lambda: write_losses(
            directories,
            lambda source: LanguageModel.load(source, batch_size=BATCH_SIZE),
            replacer,
            input_paths,
            output_paths(model),
            results,
        ),
        arguments.recompute,
    )
    results.clear()


def evaluation_passages(path: Path, nouns: set[str]) -> dict[str, dict[str, object]]:
    """名詞ごとの評価用の用例（鍵の小さい順の先頭 PASSAGES_PER_NOUN × 2 件）。"""
    table = parquet.read_table(
        path,
        columns=["word", "role", "split", "key", "token_ids", "occurrence_positions"],
        filters=[("split", "=", "evaluation"), ("role", "=", "noun")],
    )
    passages = {}
    for noun in sorted(nouns):
        rows = table.filter(compute.equal(table.column("word"), noun)).sort_by("key")
        rows = rows.slice(0, 2 * PASSAGES_PER_NOUN)
        passages[noun] = {
            "token_ids": [np.array(ids) for ids in rows.column("token_ids").to_pylist()],
            "positions": rows.column("occurrence_positions").to_pylist(),
            "sources": [{}] * rows.num_rows,
        }
    return passages


def write_losses(
    directories: dict[tuple[str, str], Path],
    load_model: Callable[[ModelSource], LanguageModel],
    replacer: PassageReplacer,
    input_paths: tuple[Path, Path, Path],
    output_paths: tuple[Path, Path],
    results: RunResults | None = None,
) -> None:
    words_path, assignments_path, passages_path = input_paths
    words = {word.word: word for word in read_rows(words_path, SelectedWord)}
    assignments = read_rows(assignments_path, SecondSenseAssignment)
    nouns = {row.first_noun for row in assignments} | {row.second_noun for row in assignments}
    passages = evaluation_passages(passages_path, nouns)
    rows = []
    for run in sorted({run for run, _ in directories}):
        done = results.get(run) if results is not None else None
        if done is None:
            items = _items(
                [row for row in assignments if row.run == run], words, passages, replacer
            )
            done = []
            for name in [name for key, name in directories if key == run]:
                model = load_model(ModelSource(repository=str(directories[(run, name)])))
                done += _measure(model, run, name, items)
                del model
                gc.collect()
                torch.cuda.empty_cache()
            if results is not None:
                results.put(run, done)
        rows += done
    losses_path, summary_path = output_paths
    losses_path.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(arrow.Table.from_pylist(rows), losses_path)
    write_rows(summary_path, LossSummary, summarize(rows))


def _items(
    assignments: Sequence[SecondSenseAssignment],
    words: dict[str, SelectedWord],
    passages: dict[str, dict[str, object]],
    replacer: PassageReplacer,
) -> list[tuple[SecondSenseAssignment, str, int, list[int], list[tuple[int, int]]]]:
    """未知語に置き換えた用例ごとの、(割り当て, 語義, 番号, トークン列, 未知語の位置の範囲)。"""
    items = []
    for row in assignments:
        unknown = words[row.unknown_word]
        for sense, noun in (("first", row.first_noun), ("second", row.second_noun)):
            replaced, _ = replaced_passages(
                words[noun], unknown, passages[noun], replacer, PASSAGES_PER_NOUN
            )
            for number, passage in enumerate(replaced):
                spans = [(start, start + unknown.token_count) for start in passage["positions"]]
                items.append((row, sense, number, list(passage["token_ids"]), spans))
    return items


def _measure(model: LanguageModel, run: str, weights: str, items: Sequence) -> list[dict]:
    log_probabilities = model.token_log_probabilities([item[3] for item in items])
    return [
        {
            "run": run,
            "weights": weights,
            "unknown_word": row.unknown_word,
            "kind": row.kind,
            "second_exposure": row.second_exposure,
            "sense": sense,
            "passage": number,
            "surprisal": float(np.mean([-values[start:stop].sum() for start, stop in spans])),
        }
        for (row, sense, number, _, spans), values in zip(items, log_probabilities, strict=True)
    ]


def summarize(rows: Sequence[dict[str, object]]) -> list[LossSummary]:
    """種類 × 2つ目の露出回数 × 語義 × 重みごとの、未知語ごとの平均の平均。"""
    per_word: dict[tuple, list[float]] = {}
    for row in rows:
        key = (row["kind"], row["second_exposure"], row["sense"], row["weights"])
        per_word.setdefault(key + (row["run"], row["unknown_word"]), []).append(row["surprisal"])
    groups: dict[tuple, list[float]] = {}
    for key, values in per_word.items():
        groups.setdefault(key[:4], []).append(float(np.mean(values)))
    return [
        LossSummary(kind, exposure, sense, weights, float(np.mean(values)), len(values))
        for (kind, exposure, sense, weights), values in sorted(
            groups.items(), key=lambda item: (*item[0][:3], weight_order(item[0][3]))
        )
    ]
