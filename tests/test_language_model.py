"""モデルの読み込みと、内部表現・確率の取り出しのテスト（極小の OLMo 2 を CPU で使う）。"""

import numpy as np
import pytest
import torch

from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.settings import ModelSource


@pytest.fixture(scope="module")
def language_model(tiny_model_source: ModelSource) -> LanguageModel:
    return LanguageModel.load(tiny_model_source, device="cpu", dtype=torch.float32, batch_size=2)


@pytest.fixture(scope="module")
def inputs(language_model: LanguageModel) -> list[list[int]]:
    """長さの違う3つの入力。2つ目は、1トークンの未知語（使われていない番号）を含む。"""
    tokenizer = language_model.tokenizer
    new_token = tokenizer.unused_token_ids(1)
    return [
        tokenizer.encode(["Do you know about the", " pasta"]),
        tokenizer.encode(["I heard something about the", new_token]),
        tokenizer.encode([" pasta"]),
    ]


def forward_one(language_model: LanguageModel, token_ids: list[int]):
    """区切りを先頭に足した1つの入力を、詰め物なしでモデルに通す。"""
    sequence = torch.tensor([[language_model.tokenizer.end_of_text_id, *token_ids]])
    with torch.inference_mode():
        return language_model.model(input_ids=sequence, output_hidden_states=True)


def test_residual_streams_match_unbatched_forward(
    language_model: LanguageModel, inputs: list[list[int]]
) -> None:
    positions = [len(token_ids) - 1 for token_ids in inputs]

    streams = language_model.residual_streams(inputs, positions)

    assert streams.shape == (3, language_model.number_of_layers + 1, 32)
    for index, token_ids in enumerate(inputs):
        outputs = forward_one(language_model, token_ids)
        expected = torch.stack(outputs.hidden_states)[:, 0, positions[index] + 1]
        np.testing.assert_allclose(streams[index], expected.numpy(), rtol=1e-4, atol=1e-5)


def test_next_token_log_probabilities_match_unbatched_forward(
    language_model: LanguageModel, inputs: list[list[int]]
) -> None:
    candidates = language_model.tokenizer.encode([" food", " animal"])
    positions = [0, 0, 0]

    log_probabilities = language_model.next_token_log_probabilities(inputs, positions, candidates)

    assert log_probabilities.shape == (3, 2)
    for index, token_ids in enumerate(inputs):
        logits = forward_one(language_model, token_ids).logits[0, 1]
        expected = torch.log_softmax(logits, dim=-1)[candidates]
        np.testing.assert_allclose(log_probabilities[index], expected.numpy(), rtol=1e-4)


def test_token_log_probabilities_predict_each_token_from_the_previous_ones(
    language_model: LanguageModel, inputs: list[list[int]]
) -> None:
    log_probabilities = language_model.token_log_probabilities(inputs)

    for index, token_ids in enumerate(inputs):
        assert log_probabilities[index].shape == (len(token_ids),)
        logits = forward_one(language_model, token_ids).logits[0]
        last = torch.log_softmax(logits[len(token_ids) - 1], dim=-1)[token_ids[-1]]
        assert log_probabilities[index][-1] == pytest.approx(last.item(), rel=1e-4)


def test_positions_outside_the_input_are_rejected(
    language_model: LanguageModel, inputs: list[list[int]]
) -> None:
    with pytest.raises(ValueError, match="入力の外"):
        language_model.residual_streams(inputs[:1], [len(inputs[0])])
    with pytest.raises(ValueError, match="位置の数"):
        language_model.residual_streams(inputs, [0])


def test_token_ids_outside_the_embedding_are_rejected(language_model: LanguageModel) -> None:
    with pytest.raises(ValueError, match="埋め込みの行"):
        language_model.token_log_probabilities([[100352]])
