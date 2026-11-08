"""2つ目の語義の追加の試行で、1つ目と2つ目の語義の用例での未知語の損失を測る（試行の作業。25|）。

- measure <run>（GPU 1枚）: 学習に使っていない評価用の用例を、名詞ごとに PASSAGES_PER_NOUN 件
  取り、X の用例の X と、Y の用例の Y を未知語に置き換えて読ませる。未知語の驚き度（語のトークンの
  −log p の和。出現と用例で平均）を、2つ目の語義を足す前の重みと、足した後の学習の 25・50・75・
  100% の重みで測り、outputs/partial-results/second-sense-evaluation/losses/<run>.parquet に書く。
  1つ目の語義の用例で驚き度が上がれば、1つ目の語義を忘れかけている。2つ目の語義の用例で下がれば、
  2つ目の語義が自然な文章でも入った。
- summarize（CPU）: 重み × 2つ目の露出回数 × 種類 × 語義ごとに平均し、losses_summary.txt に書く。
"""

import gc
import sys
from pathlib import Path

import numpy as np
import pandas
import pyarrow as arrow
import pyarrow.compute as compute
import pyarrow.parquet as parquet
import torch

from lexical_acquisition_mechanisms import (
    passage_finalization,
    second_sense_assignments,
    word_finalization,
)
from lexical_acquisition_mechanisms.csv_tables import read_rows
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.passage_replacement import PassageReplacer
from lexical_acquisition_mechanisms.run_settings import first_phase_weights
from lexical_acquisition_mechanisms.second_sense_assignments import SecondSenseAssignment
from lexical_acquisition_mechanisms.settings import ModelSource, load_settings
from lexical_acquisition_mechanisms.unknown_word_training_data import replaced_passages
from lexical_acquisition_mechanisms.word_selection import SelectedWord
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

DIRECTORY = Path("outputs/partial-results/second-sense-evaluation")
MODEL = "OLMo-2-0425-1B"
CHECKPOINTS = ["before", "checkpoint-25", "checkpoint-50", "checkpoint-75", "final"]
PASSAGES_PER_NOUN = 50


def weights(run: str, checkpoint: str) -> Path:
    if checkpoint == "before":
        return first_phase_weights(MODEL, run)
    experiment = second_sense_assignments.EXPERIMENT
    return Path("outputs/checkpoints") / experiment / MODEL / run / checkpoint


def evaluation_passages(nouns: set[str]) -> dict[str, dict[str, object]]:
    """名詞ごとの、評価用の用例（鍵の小さい順の先頭 PASSAGES_PER_NOUN × 2 件）。"""
    table = parquet.read_table(
        passage_finalization.PASSAGES_PATH,
        columns=["word", "role", "split", "key", "token_ids", "occurrence_positions"],
        filters=[("split", "=", "evaluation"), ("role", "=", "noun")],
    )
    passages = {}
    for noun in nouns:
        rows = table.filter(compute.equal(table.column("word"), noun)).sort_by("key")
        rows = rows.slice(0, 2 * PASSAGES_PER_NOUN)
        passages[noun] = {
            "token_ids": np.array(rows.column("token_ids").to_pylist()),
            "positions": rows.column("occurrence_positions").to_pylist(),
            "sources": [{"shard": 0, "row": 0, "start": 0}] * rows.num_rows,
        }
    return passages


def measure(run: str) -> None:
    settings = load_settings(Path("settings.yaml"))
    words = {word.word: word for word in read_rows(word_finalization.OUTPUT_PATH, SelectedWord)}
    assignments = [
        row
        for row in read_rows(second_sense_assignments.ASSIGNMENTS_PATH, SecondSenseAssignment)
        if row.run == run
    ]
    passages = evaluation_passages(
        {row.first_noun for row in assignments} | {row.second_noun for row in assignments}
    )
    replacer = PassageReplacer(
        WordTokenizer.from_model(settings.models[MODEL]), settings.passages.tokens_per_passage
    )
    inputs, spans, keys = [], [], []
    for row in assignments:
        unknown = words[row.unknown_word]
        for sense, noun in (("first", row.first_noun), ("second", row.second_noun)):
            replaced, _ = replaced_passages(
                words[noun], unknown, passages[noun], replacer, PASSAGES_PER_NOUN
            )
            for number, passage in enumerate(replaced):
                inputs.append(passage["token_ids"])
                spans.append(
                    [(start, start + unknown.token_count) for start in passage["positions"]]
                )
                keys.append((row, sense, noun, number))
    rows = []
    for checkpoint in CHECKPOINTS:
        model = LanguageModel.load(ModelSource(repository=str(weights(run, checkpoint))))
        log_probabilities = model.token_log_probabilities(inputs)
        for (row, sense, noun, number), values, places in zip(
            keys, log_probabilities, spans, strict=True
        ):
            surprisal = np.mean([-values[start:stop].sum() for start, stop in places])
            rows.append(
                {
                    "run": run,
                    "checkpoint": checkpoint,
                    "unknown_word": row.unknown_word,
                    "token_count": row.token_count,
                    "second_exposure": row.second_exposure,
                    "kind": row.kind,
                    "sense": sense,
                    "noun": noun,
                    "passage": number,
                    "surprisal": float(surprisal),
                }
            )
        print(f"{run} {checkpoint}: {len(inputs)} 件", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    (DIRECTORY / "losses").mkdir(parents=True, exist_ok=True)
    parquet.write_table(arrow.Table.from_pylist(rows), DIRECTORY / "losses" / f"{run}.parquet")


def summarize() -> None:
    table = pandas.concat(
        parquet.read_table(path).to_pandas()
        for path in sorted((DIRECTORY / "losses").glob("*.parquet"))
    )
    table["checkpoint"] = pandas.Categorical(table["checkpoint"], CHECKPOINTS, ordered=True)
    per_word = table.groupby(
        ["run", "checkpoint", "unknown_word", "second_exposure", "kind", "sense"], observed=True
    )["surprisal"].mean()
    summary = (
        per_word.groupby(["kind", "second_exposure", "sense", "checkpoint"], observed=True)
        .mean()
        .unstack("checkpoint")
    )
    text = summary.round(3).to_string()
    (DIRECTORY / "losses_summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    if sys.argv[1] == "measure":
        measure(sys.argv[2])
    else:
        summarize()
