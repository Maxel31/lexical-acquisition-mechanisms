"""次のトークンの交差エントロピーと、出力の対数の分配関数（logsumexp。z-loss に使う）。

継続事前学習の損失と、検証データでのパープレキシティが使う。語彙が大きい（約10万）ので、全ての位置の
logits を float32 にした写しや、log_softmax の全体をメモリに持たないように、位置のまとまり
（CHUNK_POSITIONS 個）ごとに float32 にして計算する。
- 交差エントロピー = logsumexp − 正解のトークンの logit（logsumexp は1回だけ計算する）
- 逆伝播も、まとまりごとに softmax を計算し直して、logits と同じ型の勾配に書き込む
  （d logsumexp = softmax、d 交差エントロピー = softmax − 正解の one-hot）
"""

import torch

CHUNK_POSITIONS = 4096


def next_token_losses(
    logits: torch.Tensor, input_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """形が（系列, 位置）の、次のトークンの交差エントロピーと logsumexp。

    位置 t の値は、トークン t + 1 の予測。系列の最後の位置には予測するトークンがないので、値を
    使わない（先頭のトークンを正解として計算した値が入る）。
    """
    batch_size, length, vocabulary = logits.shape
    targets = torch.roll(input_ids, -1, dims=1).reshape(-1)
    cross_entropy, log_partition = _NextTokenLosses.apply(
        logits.reshape(batch_size * length, vocabulary), targets
    )
    return cross_entropy.view(batch_size, length), log_partition.view(batch_size, length)


class _NextTokenLosses(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, targets: torch.Tensor):
        log_partition = torch.empty(len(logits), dtype=torch.float32, device=logits.device)
        target_logits = torch.empty_like(log_partition)
        for start in range(0, len(logits), CHUNK_POSITIONS):
            chunk = slice(start, start + CHUNK_POSITIONS)
            values = logits[chunk].float()
            log_partition[chunk] = torch.logsumexp(values, dim=-1)
            target_logits[chunk] = values.gather(-1, targets[chunk, None]).squeeze(-1)
        ctx.save_for_backward(logits, targets, log_partition)
        return log_partition - target_logits, log_partition

    @staticmethod
    def backward(ctx, cross_entropy_gradient, log_partition_gradient):
        logits, targets, log_partition = ctx.saved_tensors
        zeros = torch.zeros_like(log_partition)
        cross_entropy_gradient = (
            zeros if cross_entropy_gradient is None else cross_entropy_gradient.float()
        )
        log_partition_gradient = (
            zeros if log_partition_gradient is None else log_partition_gradient.float()
        )
        softmax_scale = cross_entropy_gradient + log_partition_gradient
        gradient = torch.empty_like(logits)
        for start in range(0, len(logits), CHUNK_POSITIONS):
            chunk = slice(start, start + CHUNK_POSITIONS)
            values = logits[chunk].float()
            values.sub_(log_partition[chunk, None]).exp_().mul_(softmax_scale[chunk, None])
            values.scatter_add_(-1, targets[chunk, None], -cross_entropy_gradient[chunk, None])
            gradient[chunk] = values.to(gradient.dtype)
        return gradient, None
