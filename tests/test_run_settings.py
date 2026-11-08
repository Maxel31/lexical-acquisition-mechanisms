"""run ごとの学習の設定ファイルのテスト。"""

import argparse
from pathlib import Path

import pytest

from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.csv_tables import write_rows
from lexical_acquisition_mechanisms.run_settings import (
    build_run_settings,
    load_run_settings,
    write_run_settings,
)
from lexical_acquisition_mechanisms.settings import ModelSource, SettingsError, load_settings
from lexical_acquisition_mechanisms.word_selection import SelectedWord

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_settings_are_written_for_every_run(tmp_path: Path) -> None:
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    words = [
        SelectedWord("<new_01>", "unknown", token_count=1, token_ids="100278"),
        SelectedWord("<new_02>", "unknown", token_count=1, token_ids="100279"),
        SelectedWord("zubbet", "unknown", token_count=2, token_ids="10 11"),
    ]
    write_rows(tmp_path / "words.csv", SelectedWord, words)

    paths = build_run_settings(
        settings, "unknown-word-acquisition", "OLMo-2-1124-7B", tmp_path / "words.csv", tmp_path
    )

    assert [path.stem for path in paths] == [
        *(f"run-{number}" for number in range(1, 9)),
        "control",
    ]
    run = load_run_settings(paths[0])
    assert run.learning_rate == pytest.approx(6.1e-5)
    assert run.new_token_ids == [100278, 100279]
    assert run.training_tokens == Path("data/training/unknown-word-acquisition/run-1/tokens.npy")
    assert run.output_directory == Path(
        "outputs/checkpoints/unknown-word-acquisition/OLMo-2-1124-7B/run-1"
    )
    sense_paths = build_run_settings(
        settings, "sense-addition", "OLMo-2-0425-1B", tmp_path / "words.csv", tmp_path
    )
    assert len(sense_paths) == 11
    assert load_run_settings(sense_paths[0]).new_token_ids == []
    second_paths = build_run_settings(
        settings, "second-sense-addition", "OLMo-2-0425-1B", tmp_path / "words.csv", tmp_path
    )
    assert [path.stem for path in second_paths] == [f"run-{number}" for number in range(1, 9)]
    second = load_run_settings(second_paths[0])
    # 同じ名前の未知語の獲得の run の最後の重みから学習し、未知語の埋め込みは初期化し直さない
    assert second.model_source == ModelSource(
        repository="outputs/checkpoints/unknown-word-acquisition/OLMo-2-0425-1B/run-1/final"
    )
    assert second.new_token_ids == []
    assert second.training_tokens == Path("data/training/second-sense-addition/run-1/tokens.npy")


def test_second_sense_runs_stop_without_first_phase_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    arguments = argparse.Namespace(
        experiment="second-sense-addition", model="OLMo-2-0425-1B", recompute=False
    )

    with pytest.raises(CommandStopped, match="未知語の獲得の最後の重みがありません"):
        write_run_settings(load_settings(REPOSITORY_ROOT / "settings.yaml"), arguments)


def test_broken_run_settings_are_explained(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text("experiment: unknown-word-acquisition\n", encoding="utf-8")

    with pytest.raises(SettingsError, match="必要な項目がありません"):
        load_run_settings(path)
