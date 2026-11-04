"""検証データでのパープレキシティ（継続事前学習の、学習前と、重みを保存するたび）。

検証データ（基データの直後の系列）の系列を GPU に順に配り、次のトークンの交差エントロピーの平均から
パープレキシティ（exp(平均)）を出す。学習前の値に対する変化（割合）も記録する（設計書: 学習前から
1%以内であることを確かめる）。
"""

import math

import numpy as np
import torch
import torch.distributed as distributed
from transformers import PreTrainedModel

from lexical_acquisition_mechanisms.next_token_losses import next_token_losses


def validation_perplexity(
    model: PreTrainedModel,
    validation_path,
    sequences_per_device: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> float:
    """検証データ全体でのパープレキシティ（全ての GPU で同じ値を返す）。"""
    sequences = np.load(validation_path, mmap_mode="r")
    # 全ての GPU で計算の回数をそろえるため、空の系列（-1）で埋める
    indices = list(range(len(sequences)))
    indices += [-1] * (-len(indices) % (world_size * sequences_per_device))
    mine = indices[rank::world_size]
    sums = torch.zeros(2, device=device)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(mine), sequences_per_device):
            chosen = mine[start : start + sequences_per_device]
            input_ids = torch.from_numpy(
                np.stack([sequences[max(index, 0)] for index in chosen]).astype(np.int64)
            ).to(device)
            counted = torch.tensor([index >= 0 for index in chosen], device=device)
            logits = model(input_ids=input_ids, use_cache=False).logits
            # 系列の最後の位置には、予測するトークンがない
            losses = next_token_losses(logits, input_ids)[0][:, :-1]
            sums += torch.stack(
                [(losses * counted[:, None]).sum(), counted.sum() * losses.shape[1]]
            )
    model.train()
    distributed.all_reduce(sums)
    loss, tokens = sums.tolist()
    return math.exp(loss / tokens)
