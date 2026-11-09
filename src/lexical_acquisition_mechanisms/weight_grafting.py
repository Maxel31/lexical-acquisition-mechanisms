"""学習の前と後の重みから、変わった重みを部品ごとに移したモデルを作る。

学習の前の重みを θ1、後の重みを θ2、差を Δ = θ2 − θ1 とする。部品 S ごとに、次の2つを作る。
- sufficiency: θ1 + Δ_S（S の変化だけを足す。値が θ2 の値まで戻れば、S の変化で足りる）
- necessity: θ2 − Δ_S（S の変化だけを戻す。値が θ1 の値まで消えれば、S の変化が要る）
部品は、埋め込み・各層の attention と MLP（それぞれの正規化を含む）・最後の正規化・出力の行列と、
それらをまとめたもの（全ての attention・全ての MLP・LAYER_GROUP 層ずつの層）。
"""

import re
from collections.abc import Iterator
from pathlib import Path

import torch
from safetensors.torch import load_file

LAYER_GROUP = 4


def component(name: str) -> str:
    """パラメータの名前から、部品の名前（embeddings・attention-<層>・mlp-<層> など）。"""
    if name == "model.embed_tokens.weight":
        return "embeddings"
    if name == "model.norm.weight":
        return "final-norm"
    if name == "lm_head.weight":
        return "unembedding"
    layer, rest = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", name).groups()
    if rest.startswith(("self_attn.", "post_attention_layernorm")):
        return f"attention-{layer}"
    return f"mlp-{layer}"


def component_sets(number_of_layers: int) -> dict[str, set[str]]:
    """移す部品の組（名前 → 部品の集合）。"""
    single = ["embeddings", "final-norm", "unembedding"]
    single += [
        f"{kind}-{layer}" for layer in range(number_of_layers) for kind in ("attention", "mlp")
    ]
    sets = {name: {name} for name in single}
    sets["all-attention"] = {f"attention-{layer}" for layer in range(number_of_layers)}
    sets["all-mlp"] = {f"mlp-{layer}" for layer in range(number_of_layers)}
    for start in range(0, number_of_layers, LAYER_GROUP):
        layers = range(start, min(start + LAYER_GROUP, number_of_layers))
        sets[f"layers-{start}-{layers[-1]}"] = {
            f"{kind}-{layer}" for layer in layers for kind in ("attention", "mlp")
        }
    return sets


def grafted_weights(network: torch.nn.Module, after: Path) -> Iterator[tuple[str, str]]:
    """network（θ1 を読み込んだもの）の重みを順に置き換え、そのたびに (部品の組, 移し方) を返す。

    最初の2つは、θ1 そのもの（before, baseline）と θ2 そのもの（after, baseline）。続けて、部品の組
    ごとに sufficiency と necessity。after は θ2 の safetensors のファイル。
    """
    parameters = dict(network.named_parameters())
    first = {name: value.detach().clone() for name, value in parameters.items()}
    second = load_file(str(after), device=str(next(network.parameters()).device))
    delta = {name: second[name].to(first[name].dtype) - first[name] for name in first}
    del second
    owner = {name: component(name) for name in first}
    configurations: list[tuple[str, str, set[str] | None]] = [
        ("before", "baseline", set()),
        ("after", "baseline", None),
    ]
    for name, members in component_sets(network.config.num_hidden_layers).items():
        configurations += [(name, "sufficiency", members), (name, "necessity", members)]
    for name, mode, members in configurations:
        with torch.no_grad():
            for parameter_name, parameter in parameters.items():
                moved = members is None or owner[parameter_name] in members
                if mode == "necessity":
                    moved = not moved
                parameter.copy_(first[parameter_name] + (delta[parameter_name] if moved else 0))
        yield name, mode
