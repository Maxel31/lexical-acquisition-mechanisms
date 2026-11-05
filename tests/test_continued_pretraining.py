"""継続事前学習のテスト。GPU は使わず、CPU の極小のモデルで学習する。"""

import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from lexical_acquisition_mechanisms.continued_pretraining import Process, SequenceBatches, train
from lexical_acquisition_mechanisms.run_settings import RunSettings, build_run_settings
from lexical_acquisition_mechanisms.settings import ModelSource, load_settings

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_LENGTH = 8


def free_port() -> int:
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        return connection.getsockname()[1]


def run_settings(tmp_path: Path, model: ModelSource, sequences: int = 5) -> RunSettings:
    """5つの系列（長さ 8）。1 step は2つの系列、用例は位置 3 と 20 から長さ 4。"""
    tokens = np.random.default_rng(0).integers(1, 1000, sequences * SEQUENCE_LENGTH)
    np.save(tmp_path / "tokens.npy", tokens.astype(np.uint32))
    np.save(tmp_path / "starts.npy", np.array([3, 20], dtype=np.int64))
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    training = settings.training.model_copy(
        update={
            "batch_size_in_tokens": 2 * SEQUENCE_LENGTH,
            "sequence_length": SEQUENCE_LENGTH,
            "learning_rate_by_model": {"tiny": 1e-3},
            "sequences_per_device_by_model": {"tiny": 1},
        }
    )
    settings = settings.model_copy(update={"models": {"tiny": model}, "training": training})
    (path,) = build_run_settings(
        settings, "sense-addition", "tiny", tmp_path / "words.csv", tmp_path / "runs"
    )[:1]
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    content.update(
        training_tokens=str(tmp_path / "tokens.npy"),
        passage_starts=str(tmp_path / "starts.npy"),
        passage_length=4,
        output_directory=str(tmp_path / "output"),
    )
    path.write_text(yaml.safe_dump(content), encoding="utf-8")
    return RunSettings.model_validate(content)


def test_last_step_is_filled_with_empty_sequences(tmp_path: Path, tiny_model_source) -> None:
    run = run_settings(tmp_path, tiny_model_source)

    batches = [SequenceBatches(run, Process(rank, 2, torch.device("cpu"))) for rank in range(2)]

    assert batches[0].number_of_steps == 3
    first, second = (list(batch.micro_batches(2)) for batch in batches)
    # 最後の step は系列4だけ。2つ目の GPU には、損失に数えない空の系列を渡す
    assert first[0].counted.sum() == SEQUENCE_LENGTH - 1
    assert not second[0].counted.any()
    # 系列2（位置16〜23）のうち、位置20〜23が差し込んだ用例
    (step_one,) = batches[0].micro_batches(1)
    assert step_one.inserted[0].tolist() == [False] * 4 + [True] * 4


def test_training_writes_metrics_for_every_step(
    tmp_path: Path, tiny_model_source, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("MASTER_PORT", str(free_port()))
    run = run_settings(tmp_path, tiny_model_source)

    train(run)

    lines = (tmp_path / "output" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    metrics = [json.loads(line) for line in lines]
    assert [record["step"] for record in metrics] == [1, 2, 3]
    assert all(np.isfinite(record["loss"]) and record["loss"] > 0 for record in metrics)
    assert metrics[1]["inserted_loss"] > 0


def test_two_processes_give_the_same_losses(
    tmp_path: Path, tiny_model_source, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = run_settings(tmp_path, tiny_model_source)
    shutil.copy(REPOSITORY_ROOT / "settings.yaml", tmp_path / "settings.yaml")
    path = tmp_path / "runs" / "sense-addition" / "tiny" / "same-category-1.yaml"
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc-per-node",
        "2",
        "--master-port",
        str(free_port()),
        "-m",
        "lexical_acquisition_mechanisms",
        "training",
        "run",
        "--run-settings",
        str(path),
    ]

    subprocess.run(command, cwd=tmp_path, env=environment, check=True, capture_output=True)
    together = [json.loads(line) for line in (run.output_directory / "metrics.jsonl").open()]
    shutil.rmtree(run.output_directory)
    monkeypatch.setenv("MASTER_PORT", str(free_port()))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    train(run)
    alone = [json.loads(line) for line in (run.output_directory / "metrics.jsonl").open()]

    # 計算は bf16 なので、少しずれてよい
    for one, two in zip(alone, together, strict=True):
        assert one["loss"] == pytest.approx(two["loss"], rel=2e-2)
