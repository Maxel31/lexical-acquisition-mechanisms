"""次のトークンの交差エントロピーと logsumexp を、まとまりごとに計算する処理のテスト。"""

import pytest
import torch
import torch.nn.functional as functional

from lexical_acquisition_mechanisms import next_token_losses as losses_module
from lexical_acquisition_mechanisms.next_token_losses import next_token_losses


def reference(logits: torch.Tensor, input_ids: torch.Tensor, weight: float) -> torch.Tensor:
    """ずらした logits で計算した、交差エントロピー + weight × logsumexp² の和。"""
    predicted = logits[:, :-1].reshape(-1, logits.shape[-1])
    cross_entropy = functional.cross_entropy(predicted, input_ids[:, 1:].reshape(-1))
    log_partition = torch.logsumexp(predicted, dim=-1)
    return cross_entropy * len(predicted) + weight * (log_partition**2).sum()


@pytest.mark.parametrize("chunk", [3, 4096])
def test_values_and_gradients_match_the_direct_computation(
    monkeypatch: pytest.MonkeyPatch, chunk: int
) -> None:
    # まとまり（3つの位置）が系列の区切りと合わない場合も、まとめて計算した場合と同じ
    monkeypatch.setattr(losses_module, "CHUNK_POSITIONS", chunk)
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 5, 11, generator=generator, dtype=torch.float64) * 3
    input_ids = torch.randint(0, 11, (2, 5), generator=generator)
    expected_logits = logits.clone().requires_grad_()
    logits.requires_grad_()

    cross_entropy, log_partition = next_token_losses(logits, input_ids)
    total = (cross_entropy[:, :-1] + 1e-4 * log_partition[:, :-1] ** 2).sum()
    total.backward()
    expected = reference(expected_logits, input_ids, 1e-4)
    expected.backward()

    assert total.item() == pytest.approx(expected.item(), rel=1e-6)
    assert torch.allclose(logits.grad, expected_logits.grad, atol=1e-6)


def test_gradients_have_the_type_of_the_logits() -> None:
    logits = torch.randn(1, 4, 7, dtype=torch.bfloat16, requires_grad=True)
    cross_entropy, _ = next_token_losses(logits, torch.tensor([[1, 2, 3, 4]]))

    cross_entropy.sum().backward()

    assert cross_entropy.dtype == torch.float32
    assert logits.grad.dtype == torch.bfloat16
