"""継続事前学習（training run --run-settings <run>.yaml）。

torchrun から、GPU ごとに1つのプロセスで動かす（GPU がなければ CPU で動く。テスト用）。
- 新しいトークン（1トークンの未知語）の埋め込みは、入力側・出力側とも、既存のトークンの埋め込みの
  平均に、正規分布の乱数（標準偏差は、既存の埋め込みの次元ごとの標準偏差 × new_token_noise_scale）を
  加えたものにする。乱数は seed から決まるので、未知語の獲得の全ての run で同じ初期値になる。
- モデル: run の設定の版の重みを float32 で読み、FSDP2（fully_shard。重みと最適化器の状態を GPU
  ごとに分けて持つ）で分ける。計算は bf16 にし、勾配は float32 で足し合わせる。勾配チェックポイント
  （層の途中の値を残さず、逆伝播のときに計算し直す）でメモリを減らす。
- データ: 学習データのトークン列を系列の長さずつに分け、先頭から順に読む（並べ替えない。基データは
  事前学習の順のまま）。1 step は batch_size_in_tokens ÷ 系列の長さ 個の系列で、最後の step は残りの
  系列だけ。step の系列を GPU に順に配り、全ての GPU で計算の回数がそろうように、足りない分は
  損失に数えない空の系列で埋める。
- 損失: 次のトークンの交差エントロピー + z-loss（出力の対数の分配関数の2乗 × 重み）。step の全ての
  トークンで平均する。
- 最適化器: AdamW（埋め込みには重みの減衰を掛けない）。学習率は、最初の warmup_fraction の step で
  0 から上げ、あとは一定（事前学習の最後の値のまま）。勾配のノルムを gradient_clipping に抑える。
- 記録: step ごとに、損失（全体と、差し込んだ用例のトークンだけ）・学習率・勾配のノルムを、
  出力のディレクトリの metrics.jsonl に書き、W&B にも送る（環境変数 WANDB_MODE=disabled なら
  送らない。実行し直したときは、同じ W&B の run に続けて記録する）。
- 検証データでのパープレキシティを、学習前と、重みを保存するたびに記録する
  （validation_perplexity）。
- 重みと、続きから学習するための状態の保存は training_checkpoints。
"""

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as distributed
import torch.nn.functional as functional
import wandb
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from lexical_acquisition_mechanisms.run_settings import RunSettings, load_run_settings
from lexical_acquisition_mechanisms.settings import Settings
from lexical_acquisition_mechanisms.training_checkpoints import (
    COMPLETED_FILE_NAME,
    METRICS_FILE_NAME,
    STATE_INTERVAL_STEPS,
    checkpoint_names,
    is_completed,
    mark_completed,
    resume,
    save_state,
    save_weights,
)
from lexical_acquisition_mechanisms.validation_perplexity import validation_perplexity

# 検証データでのパープレキシティの、学習前からの変化の目安（設計書: 1%以内）
PERPLEXITY_CHANGE_LIMIT = 0.01


@dataclass(frozen=True)
class Process:
    rank: int
    world_size: int
    device: torch.device


@dataclass
class MicroBatch:
    input_ids: torch.Tensor
    # 損失に数えるか（空の系列と、各系列の最初のトークンは数えない）と、差し込んだ用例のトークンか
    counted: torch.Tensor
    inserted: torch.Tensor


class SequenceBatches:
    """学習データを、step ごと・GPU ごとの系列のまとまりに分ける。"""

    def __init__(self, run: RunSettings, process: Process) -> None:
        self._tokens = np.load(run.training_tokens, mmap_mode="r")
        self._starts = np.load(run.passage_starts)
        self._run = run
        self._process = process
        self.number_of_sequences = len(self._tokens) // run.sequence_length
        self.sequences_per_step = run.batch_size_in_tokens // run.sequence_length
        self.number_of_steps = math.ceil(self.number_of_sequences / self.sequences_per_step)

    def micro_batches(self, step: int) -> Iterator[MicroBatch]:
        """step の系列のうち、この GPU が計算するものを、sequences_per_device 個ずつ返す。"""
        first = step * self.sequences_per_step
        sequences = list(
            range(first, min(first + self.sequences_per_step, self.number_of_sequences))
        )
        per_round = self._process.world_size * self._run.sequences_per_device
        # 全ての GPU で計算の回数をそろえるため、空の系列（-1）で埋める
        sequences += [-1] * (-len(sequences) % per_round)
        mine = sequences[self._process.rank :: self._process.world_size]
        for start in range(0, len(mine), self._run.sequences_per_device):
            yield self._micro_batch(mine[start : start + self._run.sequences_per_device])

    def _micro_batch(self, sequences: list[int]) -> MicroBatch:
        length = self._run.sequence_length
        input_ids = np.zeros((len(sequences), length), dtype=np.int64)
        counted = np.zeros((len(sequences), length), dtype=bool)
        inserted = np.zeros((len(sequences), length), dtype=bool)
        for row, sequence in enumerate(sequences):
            if sequence < 0:
                continue
            offset = sequence * length
            input_ids[row] = self._tokens[offset : offset + length]
            counted[row, 1:] = True
            positions = np.arange(offset, offset + length)
            latest = np.searchsorted(self._starts, positions, side="right") - 1
            within = latest >= 0
            inserted[row] = within & (
                positions < self._starts[latest.clip(min=0)] + self._run.passage_length
            )
        device = self._process.device
        return MicroBatch(
            torch.from_numpy(input_ids).to(device),
            torch.from_numpy(counted).to(device),
            torch.from_numpy(inserted).to(device),
        )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-settings",
        type=Path,
        required=True,
        help="run の設定ファイル（training write-run-settings が書いたもの）",
    )


def run_training(settings: Settings, arguments: argparse.Namespace) -> None:
    train(load_run_settings(arguments.run_settings))


def train(run: RunSettings) -> None:
    if is_completed(run):
        print(
            f"{run.output_directory} の学習は終わっています（{COMPLETED_FILE_NAME} があります）。"
        )
        return
    process = _start_process()
    try:
        run.output_directory.mkdir(parents=True, exist_ok=True)
        _start_weights_and_biases(run, process.rank)
        model = _load_model(run, process)
        optimizer = _optimizer(model, run)
        batches = SequenceBatches(run, process)
        number_of_steps = batches.number_of_steps
        warmup_steps = max(1, math.ceil(run.warmup_fraction * number_of_steps))
        checkpoints = checkpoint_names(number_of_steps, run.checkpoint_fractions)
        first_step = resume(model, optimizer, run, process.rank)
        if first_step == 0:
            _record_validation(model, run, process, 0)
        for step in range(first_step, number_of_steps):
            started = time.time()
            learning_rate = run.learning_rate * min(1.0, (step + 1) / warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            totals = _train_step(model, batches, step, run, process)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), run.gradient_clipping)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if process.rank == 0:
                _write_metrics(
                    run,
                    {
                        "step": step + 1,
                        "steps": batches.number_of_steps,
                        "loss": totals["loss"] / totals["tokens"],
                        "inserted_loss": totals["inserted_loss"]
                        / max(totals["inserted_tokens"], 1),
                        "learning_rate": learning_rate,
                        "gradient_norm": float(_full(norm)),
                        "seconds": time.time() - started,
                    },
                )
            finished = step + 1
            if finished in checkpoints:
                _record_validation(model, run, process, finished)
                save_weights(model, run, process.rank, checkpoints[finished])
            if finished < number_of_steps and (
                finished % STATE_INTERVAL_STEPS == 0 or finished in checkpoints
            ):
                save_state(model, optimizer, run, process.rank, finished)
        mark_completed(run, process.rank, number_of_steps)
    finally:
        if wandb.run is not None:
            wandb.finish()
        distributed.destroy_process_group()


def _start_weights_and_biases(run: RunSettings, rank: int) -> None:
    """1つ目のプロセスで、W&B の run を始める（実行し直したときは、同じ run に続ける）。"""
    if rank != 0:
        return
    name = f"{run.experiment}/{run.model}/{run.run}"
    wandb.init(
        project=run.weights_and_biases_project,
        name=name,
        id=hashlib.sha256(name.encode("utf-8")).hexdigest()[:16],
        resume="allow",
        config=run.model_dump(mode="json"),
        dir=run.output_directory,
    )


def _record_validation(
    model: PreTrainedModel, run: RunSettings, process: Process, step: int
) -> None:
    """検証データでのパープレキシティと、学習前（step 0 の記録）からの変化を記録する。"""
    perplexity = validation_perplexity(
        model,
        run.validation_sequences,
        run.sequences_per_device,
        process.rank,
        process.world_size,
        process.device,
    )
    if process.rank != 0:
        return
    before = perplexity
    path = run.output_directory / METRICS_FILE_NAME
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("step") == 0:
                before = record["validation_perplexity"]
    change = perplexity / before - 1
    _write_metrics(
        run, {"step": step, "validation_perplexity": perplexity, "validation_change": change}
    )
    if abs(change) > PERPLEXITY_CHANGE_LIMIT:
        print(
            f"検証データでのパープレキシティが、学習前から {change:+.2%} 変わりました"
            f"（{step} step）。原因を調べて報告してください。"
        )


def _train_step(
    model: PreTrainedModel, batches: SequenceBatches, step: int, run: RunSettings, process: Process
) -> dict[str, float]:
    """1 step の勾配を足し合わせ、全ての GPU での損失の和とトークン数を返す。"""
    micro_batches = list(batches.micro_batches(step))
    local_tokens = sum(int(batch.counted.sum()) for batch in micro_batches)
    step_tokens = torch.tensor(float(local_tokens), device=process.device)
    distributed.all_reduce(step_tokens)
    sums = torch.zeros(4, device=process.device)
    for batch in micro_batches:
        logits = model(input_ids=batch.input_ids, use_cache=False).logits.float()
        predicted = logits[:, :-1].reshape(-1, logits.shape[-1])
        targets = batch.input_ids[:, 1:].reshape(-1)
        counted = batch.counted[:, 1:].reshape(-1)
        inserted = batch.inserted[:, 1:].reshape(-1) & counted
        cross_entropy = functional.cross_entropy(predicted, targets, reduction="none")
        z_loss = run.z_loss_weight * torch.logsumexp(predicted, dim=-1) ** 2
        loss = ((cross_entropy + z_loss) * counted).sum()
        # FSDP2 は勾配を GPU の数で平均するので、その分を掛けて、step の全てのトークンでの平均にする
        (loss * process.world_size / step_tokens).backward()
        with torch.no_grad():
            sums += torch.stack(
                [
                    (cross_entropy * counted).sum(),
                    counted.sum(),
                    (cross_entropy * inserted).sum(),
                    inserted.sum(),
                ]
            )
    distributed.all_reduce(sums)
    loss_sum, tokens, inserted_loss, inserted_tokens = sums.tolist()
    return {
        "loss": loss_sum,
        "tokens": tokens,
        "inserted_loss": inserted_loss,
        "inserted_tokens": inserted_tokens,
    }


def _start_process() -> Process:
    """torchrun が渡す環境変数から、このプロセスの番号と GPU を決める（なければ1つのプロセス）。"""
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if torch.cuda.is_available():
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
        distributed.init_process_group("nccl", device_id=device)
    else:
        device = torch.device("cpu")
        distributed.init_process_group("gloo")
    return Process(distributed.get_rank(), distributed.get_world_size(), device)


def _load_model(run: RunSettings, process: Process) -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(
        run.model_source.repository, revision=run.model_source.revision, dtype=torch.float32
    )
    initialize_new_tokens(model, run)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(process.device)
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    # メッシュ（プロセスの並び）を、このプロセスの機器で作って渡す（渡さないと、GPU が見えるときは
    # CPU で学習していても GPU を使おうとする）
    mesh = init_device_mesh(process.device.type, (process.world_size,))
    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=policy)
    fully_shard(model, mesh=mesh, mp_policy=policy)
    model.train()
    return model


def initialize_new_tokens(model: PreTrainedModel, run: RunSettings) -> None:
    """新しいトークンの埋め込み（入力側・出力側）を、既存のトークンの平均 + 乱数にする。"""
    if not run.new_token_ids:
        return
    source = run.model_source
    vocabulary_size = len(
        AutoTokenizer.from_pretrained(source.repository, revision=source.revision)
    )
    initialize_token_embeddings(
        model, run.new_token_ids, vocabulary_size, run.new_token_noise_scale, run.seed
    )


def initialize_token_embeddings(
    model: PreTrainedModel,
    token_ids: Sequence[int],
    vocabulary_size: int,
    noise_scale: float,
    seed: int,
) -> None:
    """トークン番号の行（入力側・出力側）を、既存のトークン（語彙の行）の平均 + 乱数にする。

    乱数の標準偏差は、既存の埋め込みの次元ごとの標準偏差 × noise_scale。乱数は seed から決まる。
    """
    if min(token_ids) < vocabulary_size:
        raise ValueError(
            f"新しいトークン番号 {min(token_ids)} は、トークナイザが使っている番号です。"
        )
    generator = torch.Generator().manual_seed(seed)
    rows = list(token_ids)
    with torch.no_grad():
        for matrix in (model.get_input_embeddings().weight, model.get_output_embeddings().weight):
            existing = matrix[:vocabulary_size]
            noise = torch.randn(
                (len(rows), matrix.shape[1]), generator=generator, dtype=matrix.dtype
            )
            scale = existing.std(dim=0) * noise_scale
            matrix[rows] = existing.mean(dim=0) + noise.to(matrix.device) * scale
    print(f"新しいトークン {len(rows)} 個の埋め込みを初期化しました。")


def _optimizer(model: PreTrainedModel, run: RunSettings) -> torch.optim.Optimizer:
    """AdamW。埋め込み（入力側）には重みの減衰を掛けない（事前学習と同じ）。"""
    embedding = model.get_input_embeddings().weight
    decayed = [parameter for parameter in model.parameters() if parameter is not embedding]
    groups = [
        {"params": decayed, "weight_decay": run.weight_decay},
        {"params": [embedding], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups, lr=run.learning_rate, betas=run.optimizer_betas, eps=run.optimizer_epsilon
    )


def _full(value: torch.Tensor) -> torch.Tensor:
    """FSDP2 の分けたテンソル（DTensor）なら、全体の値にする。"""
    return value.full_tensor() if hasattr(value, "full_tensor") else value


def _write_metrics(run: RunSettings, record: dict[str, float | int]) -> None:
    with (run.output_directory / METRICS_FILE_NAME).open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")
    if wandb.run is not None:
        wandb.log(record, step=int(record["step"]))
