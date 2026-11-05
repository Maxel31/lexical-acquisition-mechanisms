"""未知語の3つの指標を測る処理のテスト。極小のモデルを CPU で使う。"""

from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet

from lexical_acquisition_mechanisms.csv_tables import write_rows
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.logistic_probes import train_logistic_probe
from lexical_acquisition_mechanisms.neutral_contexts import NeutralContext
from lexical_acquisition_mechanisms.output_records import Step
from lexical_acquisition_mechanisms.probe_and_normalization_words import NormalizationWord
from lexical_acquisition_mechanisms.run_results import RunResults
from lexical_acquisition_mechanisms.settings import ModelSource
from lexical_acquisition_mechanisms.unknown_word_assignments import UnknownWordAssignment
from lexical_acquisition_mechanisms.unknown_word_indicators import write_indicators
from lexical_acquisition_mechanisms.word_selection import SelectedWord
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def ids(tokenizer: WordTokenizer, word: str) -> str:
    return " ".join(map(str, tokenizer.word_token_ids(word)))


def test_indicators_are_measured_for_words_and_references(
    tmp_path: Path, tiny_model_source: ModelSource
) -> None:
    tokenizer = WordTokenizer.from_model(tiny_model_source)
    words = [
        SelectedWord(
            "zubbet",
            "unknown",
            token_count=2,
            token_ids=ids(tokenizer, "zubbet"),
            spelling_closeness="near",
        ),
        SelectedWord("<new_01>", "unknown", token_count=1, token_ids="100278"),
        SelectedWord("pasta", "noun", token_count=1, token_ids=ids(tokenizer, "pasta")),
        SelectedWord("bread", "noun", token_count=1, token_ids=ids(tokenizer, "bread")),
    ]
    write_rows(tmp_path / "words.csv", SelectedWord, words)
    assignments = [
        UnknownWordAssignment("run-1", "zubbet", 3, "near", 1, 0, "pasta", "food", 10),
        UnknownWordAssignment("run-1", "<new_01>", 1, "", 1, 1, "bread", "food", 0),
    ]
    write_rows(tmp_path / "assignments.csv", UnknownWordAssignment, assignments)
    write_rows(
        tmp_path / "contexts.csv",
        NeutralContext,
        [NeutralContext(0, ""), NeutralContext(1, "I saw the")],
    )
    split = tokenizer.split_word_token_ids([" pa", "sta"])
    references = [
        NormalizationWord("known-noun", 2, "giraffe", ids(tokenizer, "giraffe")),
        NormalizationWord("split-noun", 2, " pa|sta", " ".join(map(str, split)), "pasta"),
    ]
    write_rows(tmp_path / "references.csv", NormalizationWord, references)
    generator = np.random.default_rng(0)
    probes = {}
    for token_count in (1, 2, 3):
        for layer in range(3):
            probe = train_logistic_probe(generator.normal(size=(20, 32)), np.arange(20) % 2, 2)
            for name in ("weights", "biases", "mean", "scale"):
                probes[f"{token_count}-{layer}-{name}"] = getattr(probe, name)
    np.savez(tmp_path / "probes.npz", **probes)
    checkpoint = Path(tiny_model_source.repository)

    inputs = [
        tmp_path / name
        for name in ("words.csv", "assignments.csv", "contexts.csv", "references.csv")
    ]
    write_indicators(
        {"run-1": checkpoint, "control": checkpoint},
        lambda source: LanguageModel.load(source, device="cpu"),
        tmp_path / "probes.npz",
        tmp_path / "indicators.parquet",
        inputs,
    )

    rows = [
        row
        for row in parquet.read_table(tmp_path / "indicators.parquet").to_pylist()
        if row["run"] == "run-1"
    ]
    kinds = Counter((row["word"], row["indicator"]) for row in rows)
    # 形への慣れは多トークンの未知語と既知の名詞（3トークンは2つ目と3つ目も）、統合は分けた形の
    # ほか、意味は未知語と分けた形で測る
    assert kinds[("zubbet", "form")] == 1 and kinds[("giraffe", "form")] == 1
    assert kinds[("zubbet", "form-third")] == 1 and kinds[("giraffe", "form-third")] == 0
    assert kinds[("<new_01>", "form")] == 0 and kinds[(" pa|sta", "form")] == 0
    assert kinds[("zubbet", "integration")] == 3 and kinds[("pasta", "integration")] == 3
    assert kinds[(" pa|sta", "integration")] == 0
    assert kinds[("zubbet", "meaning")] == 3 and kinds[(" pa|sta", "meaning")] == 3
    zubbet = next(row for row in rows if row["word"] == "zubbet")
    assert (zubbet["noun"], zubbet["exposure"], zubbet["spelling_closeness"]) == (
        "pasta",
        10,
        "near",
    )
    assert all(np.isfinite(row["value"]) for row in rows)

    # 途中の結果があれば、測り終えた run は測り直さない（モデルを読まない）
    results = RunResults(Step("measure", inputs={}, outputs=[tmp_path / "again.parquet"]))
    results.put("run-1", rows)
    results.put("control", [])

    def load_model(source: ModelSource) -> LanguageModel:
        raise AssertionError("測り終えた run のモデルは読まない")

    write_indicators(
        {"run-1": checkpoint, "control": checkpoint},
        load_model,
        tmp_path / "probes.npz",
        tmp_path / "again.parquet",
        inputs,
        results=results,
    )
    assert parquet.read_table(tmp_path / "again.parquet").num_rows == len(rows)
