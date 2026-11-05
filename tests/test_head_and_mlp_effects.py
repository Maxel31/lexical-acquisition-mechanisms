"""各 head と MLP の入れ替えの効果を測る処理のテスト（極小の、無作為な重みのモデルで）。"""

from pathlib import Path

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

from lexical_acquisition_mechanisms import head_and_mlp_effects
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.head_and_mlp_effects import write_effects
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.output_records import Step
from lexical_acquisition_mechanisms.patching_checks import PatchingCheck, PatchingCheckSummary
from lexical_acquisition_mechanisms.run_results import RunResults
from lexical_acquisition_mechanisms.settings import ModelSource
from lexical_acquisition_mechanisms.task_prompts import (
    TaskItem,
    encode_prompt,
    encode_prompt_with_word_spans,
    prompt_template,
)
from lexical_acquisition_mechanisms.word_selection import SelectedWord
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

CHOICES = ("vehicle", "fruit")


def question(pair: str, cue: str, sentence: str, task: str = "sense") -> dict[str, object]:
    answer = "fruit" if cue == "donor" else "vehicle"
    example = (TaskItem(task, "We ate {word}.", ("food", "tool"), "food"), "soup")
    return {
        "kind": "other-category",
        "number_of_senses": 1,
        "target": "car",
        "donor": "fruit",
        "donor_sentence": pair,
        "cue": cue,
        "task": task,
        "prompt_template": prompt_template([example], TaskItem(task, sentence, CHOICES, answer)),
        "choices": list(CHOICES),
        "answer": answer,
    }


def test_effects_are_measured_for_both_directions(
    tmp_path: Path, tiny_model_source: ModelSource, capsys: pytest.CaptureFixture[str]
) -> None:
    tokenizer = WordTokenizer.from_model(tiny_model_source)
    car = " ".join(map(str, tokenizer.word_token_ids("car")))
    write_rows(tmp_path / "words.csv", SelectedWord, [SelectedWord("car", "target", token_ids=car)])
    questions = [
        question("ate", "donor", "She ate the {word}."),
        question("ate", "target", "She drove the {word}."),
        question("ate", "donor", "She ate the {word}.", task="category"),
        # 2文の長さが違う組は除く
        question("long", "donor", "She ate the {word}."),
        question("long", "target", "She drove the red {word}."),
    ]
    parquet.write_table(arrow.Table.from_pylist(questions), tmp_path / "questions.parquet")
    directory = Path(tiny_model_source.repository)
    checkpoints = {
        run: directory for run in ("other-category-1", "noun-control", "adjective-control")
    }

    write_effects(
        checkpoints,
        lambda source: LanguageModel.load(source, device="cpu"),
        (tmp_path / "words.csv", tmp_path / "questions.parquet"),
        (
            tmp_path / "pairs.parquet",
            tmp_path / "effects.npz",
            tmp_path / "checks.csv",
            tmp_path / "summary.csv",
        ),
        seed=0,
    )

    assert "1 組除きました" in capsys.readouterr().out
    rows = parquet.read_table(tmp_path / "pairs.parquet").to_pylist()
    # 別カテゴリの run と名詞の対照で、1組 × 2方向。形容詞の対照には問題がない
    assert [(row["run"], row["direction"]) for row in rows] == [
        ("other-category-1", "donor-to-target"),
        ("other-category-1", "target-to-donor"),
        ("noun-control", "donor-to-target"),
        ("noun-control", "target-to-donor"),
    ]
    assert (rows[0]["question"], rows[0]["partner"]) == (0, 1)
    assert rows[0]["total_effect"] == pytest.approx(rows[0]["metric"] + rows[1]["metric"])
    # 手順1で使える組かは、select-significant-heads-and-mlps で決める
    assert all("usable" not in row for row in rows)
    effects = np.load(tmp_path / "effects.npz")
    assert effects["target_position"].shape == (4, 2 * (4 + 1))
    assert list(effects["heads"][:5]) == [0, 1, 2, 3, -1]
    # 最後の層の出力は、最後の位置（"Answer:"）でだけ効く
    final_layer = effects["layers"] == 1
    assert np.all(effects["target_position"][:, final_layer] == 0)
    assert np.any(effects["after_target_positions"][:, final_layer] != 0)
    # 同じ重みなので、run によらない
    np.testing.assert_allclose(effects["target_position"][:2], effects["target_position"][2:])
    # 実際の入れ替えで確かめるのは対照でない run だけ。構成要素は10個なので、全てが上位に入る
    checks = read_rows(tmp_path / "checks.csv", PatchingCheck)
    assert {(check.run, check.chosen) for check in checks} == {("other-category-1", "largest")}
    assert len(checks) == 2 * 10
    final = [check for check in checks if check.layer == 1]
    assert all(check.estimated == 0 and abs(check.actual) < 1e-5 for check in final)
    summary = read_rows(tmp_path / "summary.csv", PatchingCheckSummary)
    assert [(row.components, row.pairs) for row in summary] == [(10, 2)]


def test_the_target_position_is_the_last_token_of_the_word(
    tmp_path: Path, tiny_model_source: ModelSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer = WordTokenizer.from_model(tiny_model_source)
    word = next(
        candidate
        for candidate in ("zorbaxel", "blicketory", "flumpering")
        if len(tokenizer.word_token_ids(candidate)) == 3
    )
    ids = " ".join(map(str, tokenizer.word_token_ids(word)))
    write_rows(tmp_path / "words.csv", SelectedWord, [SelectedWord(word, "target", token_ids=ids)])
    questions = [
        question("ate", "donor", "She ate the {word}.") | {"target": word},
        question("ate", "target", "She drove the {word}.") | {"target": word},
    ]
    parquet.write_table(arrow.Table.from_pylist(questions), tmp_path / "questions.parquet")
    captured = []
    original = head_and_mlp_effects.outputs_and_gradients

    def recording(model, inputs, positions, choices):
        captured.append(list(positions))
        return original(model, inputs, positions, choices)

    monkeypatch.setattr(head_and_mlp_effects, "outputs_and_gradients", recording)

    write_effects(
        {"other-category-1": Path(tiny_model_source.repository)},
        lambda source: LanguageModel.load(source, device="cpu"),
        (tmp_path / "words.csv", tmp_path / "questions.parquet"),
        tuple(tmp_path / name for name in ("p.parquet", "e.npz", "c.csv", "s.csv")),
        seed=0,
    )

    # 文の中の語の3つのトークンのうち、最後のトークンと、語より後の全ての位置
    template = questions[0]["prompt_template"]
    _, spans = encode_prompt_with_word_spans(template, tokenizer.word_token_ids(word), tokenizer)
    length = len(encode_prompt(template, tokenizer.word_token_ids(word), tokenizer))
    assert captured[0] == [spans[0][-1], *range(spans[0].stop, length)]


def test_resumed_measurement_gives_the_same_outputs(
    tmp_path: Path, tiny_model_source: ModelSource
) -> None:
    tokenizer = WordTokenizer.from_model(tiny_model_source)
    car = " ".join(map(str, tokenizer.word_token_ids("car")))
    write_rows(tmp_path / "words.csv", SelectedWord, [SelectedWord("car", "target", token_ids=car)])
    questions = [
        question("ate", "donor", "She ate the {word}."),
        question("ate", "target", "She drove the {word}."),
    ]
    parquet.write_table(arrow.Table.from_pylist(questions), tmp_path / "questions.parquet")
    directory = Path(tiny_model_source.repository)
    checkpoints = {run: directory for run in ("other-category-1", "noun-control")}
    inputs = (tmp_path / "words.csv", tmp_path / "questions.parquet")

    def measure(name: str, load_model, results: RunResults | None) -> tuple[Path, ...]:
        outputs = tuple(
            tmp_path / f"{name}-{part}"
            for part in ("pairs.parquet", "effects.npz", "checks.csv", "summary.csv")
        )
        write_effects(checkpoints, load_model, inputs, outputs, seed=0, results=results)
        return outputs

    def load(source: ModelSource) -> LanguageModel:
        return LanguageModel.load(source, device="cpu")

    # 止まらずに測った場合
    straight = measure("straight", load, None)
    # 1つ目の run を測り終えて止まり、続きから測った場合（測り終えた run のモデルは読まない）
    results = RunResults(Step("measure", inputs={}, outputs=[tmp_path / "resumed-pairs.parquet"]))
    loaded = []

    def stop_after_first(source: ModelSource) -> LanguageModel:
        if loaded:
            raise RuntimeError("止まった")
        loaded.append(source)
        return load(source)

    with pytest.raises(RuntimeError, match="止まった"):
        measure("resumed", stop_after_first, results)
    loaded.clear()

    def count_loads(source: ModelSource) -> LanguageModel:
        loaded.append(source)
        return load(source)

    resumed = measure("resumed", count_loads, results)

    assert len(loaded) == 1
    assert parquet.read_table(resumed[0]).equals(parquet.read_table(straight[0]))
    for name in ("target_position", "after_target_positions"):
        np.testing.assert_array_equal(np.load(resumed[1])[name], np.load(straight[1])[name])
    for path, other in zip(resumed[2:], straight[2:], strict=True):
        assert path.read_text() == other.read_text()
