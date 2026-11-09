"""変わった重みを部品ごとに移したモデルを作る処理のテスト。"""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from lexical_acquisition_mechanisms.settings import ModelSource
from lexical_acquisition_mechanisms.weight_grafting import (
    component,
    component_sets,
    grafted_weights,
)


def test_parameters_belong_to_the_attention_or_mlp_of_their_layer() -> None:
    assert component("model.embed_tokens.weight") == "embeddings"
    assert component("model.layers.3.self_attn.q_proj.weight") == "attention-3"
    assert component("model.layers.3.post_attention_layernorm.weight") == "attention-3"
    assert component("model.layers.3.mlp.down_proj.weight") == "mlp-3"
    assert component("model.layers.3.post_feedforward_layernorm.weight") == "mlp-3"
    assert component("model.norm.weight") == "final-norm"
    assert component("lm_head.weight") == "unembedding"


def test_component_sets_include_single_parts_and_groups_of_layers() -> None:
    sets = component_sets(6)

    assert sets["mlp-5"] == {"mlp-5"}
    assert sets["all-attention"] == {f"attention-{layer}" for layer in range(6)}
    assert sets["layers-4-5"] == {"attention-4", "mlp-4", "attention-5", "mlp-5"}
    # 単独の部品 3 + 2 × 6、全ての attention・MLP、4 層ずつの組 2 つ
    assert len(sets) == 3 + 12 + 2 + 2


def test_grafted_weights_move_only_the_changes_of_the_chosen_parts(
    tmp_path: Path, tiny_model_source: ModelSource
) -> None:
    before = AutoModelForCausalLM.from_pretrained(tiny_model_source.repository)
    after = AutoModelForCausalLM.from_pretrained(tiny_model_source.repository)
    with torch.no_grad():
        for parameter in after.parameters():
            parameter.add_(1.0)
    after.save_pretrained(tmp_path / "after")
    first = {name: value.detach().clone() for name, value in before.named_parameters()}

    seen = {}
    for name, mode in grafted_weights(before, tmp_path / "after" / "model.safetensors"):
        # 部品ごとに、θ2 になった（1 足された）パラメータの名前
        seen[(name, mode)] = {
            parameter_name
            for parameter_name, value in before.named_parameters()
            if torch.allclose(value, first[parameter_name] + 1.0)
        }

    names = set(first)
    attention = {name for name in names if component(name).startswith("attention-")}
    assert list(seen)[:2] == [("before", "baseline"), ("after", "baseline")]
    assert seen[("before", "baseline")] == set()
    assert seen[("after", "baseline")] == names
    assert seen[("mlp-0", "sufficiency")] == {name for name in names if component(name) == "mlp-0"}
    assert seen[("all-attention", "necessity")] == names - attention
    assert len(seen) == 2 + 2 * len(component_sets(2))
