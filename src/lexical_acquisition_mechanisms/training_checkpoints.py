"""継続事前学習の、重みと途中の状態の保存。

- 重み: 学習の進み具合が checkpoint_fractions に届いた step と最後の step に、Hugging Face の
  形式（トークナイザを含む。数値の型は checkpoint_dtype）で、出力のディレクトリの
  checkpoint-<百分率>/ と final/ に保存する。
- 続きから学習するための状態（重みと最適化器の状態。GPU ごとに分けたまま）: STATE_INTERVAL_STEPS
  step ごとと、重みを保存した step に、state/ に保存する（新しい状態を書き終えてから、前の状態と
  入れ替える）。run を実行し直すと、state/ があれば、その step の次から学習する（記録は、その step
  までを残す）。
- 学習を終えたら completed.json を書く。あれば、実行し直しても学習しない。
"""

import json
import math
import shutil
from collections.abc import Sequence

import torch
import torch.distributed as distributed
import torch.distributed.checkpoint as checkpoint
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_state_dict,
)
from transformers import AutoTokenizer, PreTrainedModel

from lexical_acquisition_mechanisms.run_settings import RunSettings

METRICS_FILE_NAME = "metrics.jsonl"
STATE_DIRECTORY = "state"
STEP_FILE_NAME = "step.json"
COMPLETED_FILE_NAME = "completed.json"
FINAL_DIRECTORY = "final"
STATE_INTERVAL_STEPS = 20


def checkpoint_names(number_of_steps: int, fractions: Sequence[float]) -> dict[int, str]:
    """重みを保存する step（1 から数える）と、保存するディレクトリの名前。"""
    names = {
        max(1, math.ceil(fraction * number_of_steps)): f"checkpoint-{round(fraction * 100)}"
        for fraction in fractions
    }
    names[number_of_steps] = FINAL_DIRECTORY
    return names


def is_completed(run: RunSettings) -> bool:
    return (run.output_directory / COMPLETED_FILE_NAME).exists()


def save_weights(model: PreTrainedModel, run: RunSettings, rank: int, name: str) -> None:
    """重みの全体を1つ目のプロセスに集め、Hugging Face の形式で保存する。"""
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    full_state = get_model_state_dict(model, options=options)
    if rank == 0:
        directory = run.output_directory / name
        dtype = getattr(torch, run.checkpoint_dtype)
        state = {key: value.to(dtype) for key, value in full_state.items()}
        model.save_pretrained(directory, state_dict=state)
        source = run.model_source
        tokenizer = AutoTokenizer.from_pretrained(source.repository, revision=source.revision)
        tokenizer.save_pretrained(directory)
    distributed.barrier()


def save_state(
    model: PreTrainedModel, optimizer: torch.optim.Optimizer, run: RunSettings, rank: int, step: int
) -> None:
    """続きから学習するための状態を保存する（step は、学習を終えた step の数）。"""
    writing = run.output_directory / f"{STATE_DIRECTORY}.writing"
    if rank == 0:
        shutil.rmtree(writing, ignore_errors=True)
    distributed.barrier()
    model_state, optimizer_state = get_state_dict(model, optimizer)
    checkpoint.save({"model": model_state, "optimizer": optimizer_state}, checkpoint_id=writing)
    distributed.barrier()
    if rank == 0:
        (writing / STEP_FILE_NAME).write_text(json.dumps({"step": step}), encoding="utf-8")
        state = run.output_directory / STATE_DIRECTORY
        shutil.rmtree(state, ignore_errors=True)
        writing.rename(state)
    distributed.barrier()


def resume(
    model: PreTrainedModel, optimizer: torch.optim.Optimizer, run: RunSettings, rank: int
) -> int:
    """保存した状態があれば読み込み、学習を終えた step の数を返す（なければ 0）。"""
    state_directory = run.output_directory / STATE_DIRECTORY
    step_path = state_directory / STEP_FILE_NAME
    if not step_path.exists():
        return 0
    model_state, optimizer_state = get_state_dict(model, optimizer)
    state = {"model": model_state, "optimizer": optimizer_state}
    checkpoint.load(state, checkpoint_id=state_directory)
    set_state_dict(
        model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optimizer"]
    )
    step = json.loads(step_path.read_text(encoding="utf-8"))["step"]
    if rank == 0:
        _keep_metrics_until(run, step)
        print(f"{state_directory} から、{step} step の次から学習します。")
    distributed.barrier()
    return step


def mark_completed(run: RunSettings, rank: int, number_of_steps: int) -> None:
    """学習を終えたことを書き、続きから学習するための状態を消す（重みの数倍の大きさがあるため）。"""
    if rank == 0:
        record = {"steps": number_of_steps}
        (run.output_directory / COMPLETED_FILE_NAME).write_text(
            json.dumps(record), encoding="utf-8"
        )
        shutil.rmtree(run.output_directory / STATE_DIRECTORY, ignore_errors=True)
    distributed.barrier()


def _keep_metrics_until(run: RunSettings, step: int) -> None:
    """記録のうち、step までの行だけを残す（その後の step は、学習し直すので書き直す）。"""
    path = run.output_directory / METRICS_FILE_NAME
    if not path.exists():
        return
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("step", 0) <= step
    ]
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
