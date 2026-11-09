"""各 head と MLP の出力の取り出しと入れ替えのテスト（極小の、無作為な重みのモデルで）。"""

import numpy as np
import pytest
import torch

from lexical_acquisition_mechanisms.component_patching import (
    MLP,
    Component,
    ComponentOutputs,
    components,
    estimated_effects,
    outputs_and_gradients,
    patched_metrics,
)
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.settings import ModelSource


@pytest.fixture(scope="module")
def model(tiny_model_source: ModelSource) -> LanguageModel:
    return LanguageModel.load(tiny_model_source, device="cpu", batch_size=4)


def pair(model: LanguageModel) -> tuple[list[int], list[int], tuple[int, int]]:
    """手がかりの1語だけが違う、長さが同じ2つの入力と、選択肢。"""
    first = model.tokenizer.encode(["She cooked the pasta. Answer:"])
    second = model.tokenizer.encode(["She parked the pasta. Answer:"])
    assert len(first) == len(second)
    choice = tuple(model.tokenizer.word_token_ids(word)[0] for word in ("food", "car"))
    return first, second, choice


def test_components_are_the_heads_and_then_the_mlp_of_each_layer(model: LanguageModel) -> None:
    order = components(model)

    assert len(order) == 2 * (4 + 1)
    assert order[:6] == [
        *(Component(0, head) for head in range(4)),
        Component(0, MLP),
        Component(1, 0),
    ]


def test_replacing_with_the_original_outputs_keeps_the_metric(model: LanguageModel) -> None:
    first, _, choice = pair(model)
    positions = list(range(2, len(first)))
    metrics, outputs, _ = outputs_and_gradients(model, [first], positions, [choice])

    patches = [(component, outputs[0].of(component)) for component in components(model)]
    patched = patched_metrics(model, first, positions, choice, patches)

    np.testing.assert_allclose(patched, metrics[0], atol=1e-5)


def test_estimated_effects_match_small_replacements(model: LanguageModel) -> None:
    first, second, choice = pair(model)
    positions = [2, 3]
    metrics, outputs, gradients = outputs_and_gradients(model, [first], positions, [choice])
    generator = torch.Generator().manual_seed(0)

    def nudged(values: torch.Tensor) -> torch.Tensor:
        # 出力の大きさの 1% の乱数を足す（近似は、入れ替える値が近いほど正確）
        noise = torch.randn(values.shape, generator=generator)
        return values + 1e-2 * values.std() * noise

    replacements = ComponentOutputs(nudged(outputs[0].heads), nudged(outputs[0].mlps))

    estimated = estimated_effects(outputs[0], gradients[0], replacements)
    patches = [(component, replacements.of(component)) for component in components(model)]
    actual = metrics[0] - patched_metrics(model, first, positions, choice, patches)

    np.testing.assert_allclose(estimated, actual, rtol=0.05, atol=1e-6)
    # 2つの入力をまとめても、入力ごとの勾配は変わらない
    _, _, together = outputs_and_gradients(
        model, [first, second], positions, [choice, choice[::-1]]
    )
    torch.testing.assert_close(together[0].heads, gradients[0].heads)


def test_final_layer_outputs_affect_the_metric_only_at_the_last_position(
    model: LanguageModel,
) -> None:
    first, second, choice = pair(model)
    final_layer = [
        index for index, component in enumerate(components(model)) if component.layer == 1
    ]
    effects = {}
    for position in (len(first) - 2, len(first) - 1):
        metrics, outputs, gradients = outputs_and_gradients(
            model, [first, second], [position], [choice, choice]
        )
        estimated = estimated_effects(outputs[0], gradients[0], outputs[1])
        patches = [
            (components(model)[index], outputs[1].of(components(model)[index]))
            for index in final_layer
        ]
        actual = metrics[0] - patched_metrics(model, first, [position], choice, patches)
        effects[position] = (estimated[final_layer], actual)

    before_last, last = effects[len(first) - 2], effects[len(first) - 1]
    assert np.all(before_last[0] == 0) and np.allclose(before_last[1], 0, atol=1e-6)
    assert np.all(np.abs(last[1]) > 0)
    # 最後の層の後には非線形な部分が最後の正規化しかないので、近似はほぼ実際の値になる
    np.testing.assert_allclose(last[0], last[1], rtol=0.2, atol=1e-5)


def test_inputs_of_different_lengths_are_rejected(model: LanguageModel) -> None:
    first, _, choice = pair(model)

    with pytest.raises(ValueError, match="長さがそろっていません"):
        outputs_and_gradients(model, [first, first[:-1]], [0], [choice, choice])
