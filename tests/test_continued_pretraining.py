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
from transformers import AutoModelForCausalLM

from lexical_acquisition_mechanisms import continued_pretraining, training_checkpoints
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.continued_pretraining import (
    Process,
    SequenceBatches,
    initialize_new_tokens,
    train,
)
from lexical_acquisition_mechanisms.run_settings import RunSettings, build_run_settings
from lexical_acquisition_mechanisms.settings import ModelSource, load_settings
from lexical_acquisition_mechanisms.training_checkpoints import final_directories

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_LENGTH = 8


def free_port() -> int:
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        return connection.getsockname()[1]


def run_settings(tmp_path: Path, model: ModelSource, sequences: int = 5) -> RunSettings:
    """5つの系列（長さ 8）。1 step は2つの系列、用例は位置 3 と 20 から長さ 4。"""
    tmp_path.mkdir(parents=True, exist_ok=True)
    tokens = np.random.default_rng(0).integers(1, 1000, sequences * SEQUENCE_LENGTH)
    np.save(tmp_path / "tokens.npy", tokens.astype(np.uint32))
    np.save(tmp_path / "starts.npy", np.array([3, 20], dtype=np.int64))
    validation = np.random.default_rng(1).integers(1, 1000, (3, SEQUENCE_LENGTH))
    np.save(tmp_path / "validation.npy", validation.astype(np.uint32))
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    training = settings.training.model_copy(
        update={
            "batch_size_in_tokens": 2 * SEQUENCE_LENGTH,
            "sequence_length": SEQUENCE_LENGTH,
            "learning_rate_by_model": {"tiny": 1e-3},
            "sequences_per_device_by_model": {"tiny": 1},
            "gradient_checkpointing_by_model": {"tiny": True},
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
        validation_sequences=str(tmp_path / "validation.npy"),
        output_directory=str(tmp_path / "output"),
        checkpoint_fractions=[0.5],
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
    records = [json.loads(line) for line in lines]
    metrics = [record for record in records if "loss" in record]
    assert [record["step"] for record in metrics] == [1, 2, 3]
    assert all(np.isfinite(record["loss"]) and record["loss"] > 0 for record in metrics)
    assert metrics[1]["inserted_loss"] > 0
    # 最後の step は1系列だけ（1 step の半分）なので、学習率を半分にする
    assert [record["learning_rate"] for record in metrics] == pytest.approx([1e-3, 1e-3, 5e-4])
    # 検証データでのパープレキシティは、学習前（step 0）と、重みを保存した step 2・3 に記録する
    validation = [record for record in records if "validation_perplexity" in record]
    assert [record["step"] for record in validation] == [0, 2, 3]
    assert validation[0]["validation_change"] == 0
    assert all(record["validation_perplexity"] > 1 for record in validation)


def test_gradient_checkpointing_follows_the_setting_and_keeps_the_losses(
    tmp_path: Path, tiny_model_source, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    load_model = continued_pretraining._load_model
    used = []

    def recording_load_model(run: RunSettings, process):
        model = load_model(run, process)
        used.append(model.is_gradient_checkpointing)
        return model

    monkeypatch.setattr(continued_pretraining, "_load_model", recording_load_model)
    losses = []
    for checkpointing in (True, False):
        monkeypatch.setenv("MASTER_PORT", str(free_port()))
        run = run_settings(tmp_path / str(checkpointing), tiny_model_source)
        train(run.model_copy(update={"gradient_checkpointing": checkpointing}))
        losses.append([record["loss"] for record in metrics_of(run) if "loss" in record])

    assert used == [True, False]
    # 順伝播を計算し直しても、しなくても、学習は同じ
    assert losses[0] == pytest.approx(losses[1], rel=1e-5)


def test_runs_without_inserted_passages_are_trained(
    tmp_path: Path, tiny_model_source, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("MASTER_PORT", str(free_port()))
    run = run_settings(tmp_path, tiny_model_source)
    # 未知語の獲得の対照の run は、用例を差し込まない（用例の始まりが空）
    np.save(run.passage_starts, np.zeros(0, dtype=np.int64))

    train(run)

    metrics = [record for record in metrics_of(run) if "loss" in record]
    assert [record["step"] for record in metrics] == [1, 2, 3]
    assert all(record["inserted_loss"] is None for record in metrics)


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
    together = [
        record
        for record in map(json.loads, (run.output_directory / "metrics.jsonl").open())
        if "loss" in record
    ]
    shutil.rmtree(run.output_directory)
    monkeypatch.setenv("MASTER_PORT", str(free_port()))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    train(run)
    alone = [
        record
        for record in map(json.loads, (run.output_directory / "metrics.jsonl").open())
        if "loss" in record
    ]

    # 計算は bf16 なので、少しずれてよい（勾配のノルムが合うので、GPU の数で掛ける係数も正しい）
    for one, two in zip(alone, together, strict=True):
        assert one["loss"] == pytest.approx(two["loss"], rel=2e-2)
        assert one["gradient_norm"] == pytest.approx(two["gradient_norm"], rel=2e-2)


def test_training_stops_when_the_loss_is_not_finite(
    tmp_path: Path, tiny_model_source, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("MASTER_PORT", str(free_port()))
    run = run_settings(tmp_path, tiny_model_source)

    def not_finite(*arguments) -> dict[str, float]:
        return {"loss": float("nan"), "tokens": 1.0, "inserted_loss": 0.0, "inserted_tokens": 0.0}

    monkeypatch.setattr(continued_pretraining, "_train_step", not_finite)

    with pytest.raises(CommandStopped, match="1 step 目で、損失か勾配のノルムが有限でなく"):
        train(run)
    assert not (run.output_directory / "completed.json").exists()
    assert not [record for record in metrics_of(run) if "loss" in record]


def metrics_of(run: RunSettings) -> list[dict]:
    with (run.output_directory / "metrics.jsonl").open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def test_weights_are_saved_and_finished_runs_are_skipped(
    tmp_path: Path, tiny_model_source, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MASTER_PORT", str(free_port()))
    run = run_settings(tmp_path, tiny_model_source)

    train(run)
    train(run)

    # 3 step のうち、半分（2 step）と最後に保存する。終わった run は学習し直さない
    for name in ("checkpoint-50", "final"):
        assert (run.output_directory / name / "config.json").exists()
        assert (run.output_directory / name / "tokenizer.json").exists()
    assert (run.output_directory / "completed.json").exists()
    assert not (run.output_directory / "state").exists()
    assert [record["step"] for record in metrics_of(run) if "loss" in record] == [1, 2, 3]


def test_stopped_training_continues_from_the_saved_state(
    tmp_path: Path, tiny_model_source, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(continued_pretraining, "STATE_INTERVAL_STEPS", 1)
    monkeypatch.setenv("MASTER_PORT", str(free_port()))
    whole = run_settings(tmp_path / "whole", tiny_model_source)
    train(whole)
    stopped = run_settings(tmp_path / "stopped", tiny_model_source)
    original_step = continued_pretraining._train_step

    def stop_at_the_third_step(model, batches, step, run, process):
        if step == 2:
            raise RuntimeError("学習を止めた")
        return original_step(model, batches, step, run, process)

    monkeypatch.setattr(continued_pretraining, "_train_step", stop_at_the_third_step)
    monkeypatch.setenv("MASTER_PORT", str(free_port()))
    with pytest.raises(RuntimeError, match="学習を止めた"):
        train(stopped)
    monkeypatch.setattr(continued_pretraining, "_train_step", original_step)
    monkeypatch.setenv("MASTER_PORT", str(free_port()))
    train(stopped)

    # 2 step 目の後の状態から、3 step 目を学習し直す（止めずに学習したときと同じ損失）
    def losses(run: RunSettings) -> list[dict]:
        return [record for record in metrics_of(run) if "loss" in record]

    assert [record["step"] for record in losses(stopped)] == [1, 2, 3]
    assert losses(stopped)[2]["loss"] == pytest.approx(losses(whole)[2]["loss"], rel=1e-3)


def test_new_tokens_start_from_the_mean_of_existing_embeddings(
    tmp_path: Path, tiny_model_source
) -> None:
    run = run_settings(tmp_path, tiny_model_source)
    model = AutoModelForCausalLM.from_pretrained(tiny_model_source.repository)
    vocabulary_size = 100278
    exact = run.model_copy(update={"new_token_ids": [100278, 100279], "new_token_noise_scale": 0.0})
    noisy = run.model_copy(update={"new_token_ids": [100278, 100279], "new_token_noise_scale": 0.1})

    initialize_new_tokens(model, exact)

    for matrix in (model.get_input_embeddings().weight, model.get_output_embeddings().weight):
        mean = matrix[:vocabulary_size].mean(dim=0)
        assert torch.allclose(matrix[100278], mean) and torch.allclose(matrix[100279], mean)
    initialize_new_tokens(model, noisy)
    embedding = model.get_input_embeddings().weight
    difference = (embedding[100278] - embedding[:vocabulary_size].mean(dim=0)).detach()
    assert 0 < float(difference.abs().max()) < 1
    assert not torch.allclose(embedding[100278], embedding[100279])


def test_changed_runs_stop_until_they_are_recomputed(
    tmp_path: Path, tiny_model_source, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MASTER_PORT", str(free_port()))
    run = run_settings(tmp_path, tiny_model_source)
    train(run)
    changed = run.model_copy(update={"learning_rate": run.learning_rate * 2})

    # 学習を終えた run でも、設定が違えば古い重みを使わずに止まる
    with pytest.raises(CommandStopped, match="--recompute"):
        train(changed)
    monkeypatch.setenv("MASTER_PORT", str(free_port()))
    train(changed, recompute=True)

    completed = json.loads((run.output_directory / "completed.json").read_text(encoding="utf-8"))
    assert completed["fingerprint"]["settings"]["learning_rate"] == changed.learning_rate
    assert [record["step"] for record in metrics_of(run) if "loss" in record] == [1, 2, 3]


def test_saved_states_of_other_training_data_are_not_continued(
    tmp_path: Path, tiny_model_source, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(continued_pretraining, "STATE_INTERVAL_STEPS", 1)
    monkeypatch.setenv("MASTER_PORT", str(free_port()))
    run = run_settings(tmp_path, tiny_model_source)
    original_step = continued_pretraining._train_step

    def stop_at_the_third_step(model, batches, step, run, process):
        if step == 2:
            raise RuntimeError("学習を止めた")
        return original_step(model, batches, step, run, process)

    monkeypatch.setattr(continued_pretraining, "_train_step", stop_at_the_third_step)
    with pytest.raises(RuntimeError, match="学習を止めた"):
        train(run)
    monkeypatch.setattr(continued_pretraining, "_train_step", original_step)
    # 学習データを作り直すと、途中までの状態は使わない
    tokens = np.load(run.training_tokens)
    np.save(run.training_tokens, tokens[::-1].copy())
    monkeypatch.setenv("MASTER_PORT", str(free_port()))

    with pytest.raises(CommandStopped, match="途中までの学習"):
        train(run)


def test_final_directories_need_completed_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(training_checkpoints, "CHECKPOINTS_DIRECTORY", tmp_path)
    for run in ("run-1", "run-2"):
        # 最後の重みの config.json だけでは、学習を終えたとみなさない
        (tmp_path / "experiment" / "tiny" / run / "final").mkdir(parents=True)
        (tmp_path / "experiment" / "tiny" / run / "final" / "config.json").write_text("{}")
    (tmp_path / "experiment" / "tiny" / "run-1" / "completed.json").write_text("{}")

    with pytest.raises(CommandStopped, match="run-2"):
        final_directories("experiment", "tiny", ["run-1", "run-2"], "command")
    finished = final_directories("experiment", "tiny", ["run-1"], "command")
    assert finished == {"run-1": tmp_path / "experiment" / "tiny" / "run-1" / "final"}
