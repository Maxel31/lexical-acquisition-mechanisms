"""各 head と MLP（各層の全結合のブロック）の出力の、取り出しと入れ替え。

語義の追加の手順2（head の地図）で使う。構成要素は、各層の各 head と MLP。
- head の出力: 注意の重みで値を混ぜた、その head の部分（出力の射影 o_proj に入る前の値）。
  OLMo 2 は、注意の出力を射影し正規化してから残差ストリームに足すので、head ごとに分かれている
  射影の前の値を使う。
- MLP の出力: MLP の出力を正規化（post_feedforward_layernorm）した値（残差ストリームに足す値）。
指標 m は、入力の最後のトークン（"Answer:"）の次に来る2つの選択肢の語の、ロジットの差
（対数確率の差と同じ）。もう一方の選択肢を None にすると、1つの選択肢の対数確率になる。
入れ替えの効果は e = m − m_c（m_c は、構成要素 c の出力を入れ替えたときの m）。
- 勾配による近似（attribution patching）: e ≈ −g·(入れ替える値 − 元の値)。g は、元の出力に対する
  m の勾配。和は、入れ替える位置と次元でとる。1回の順伝播と逆伝播で、全ての構成要素の分を出せる。
- 実際の入れ替え: 指定した構成要素の、指定した位置の出力を置き換えて、m を測る。
- 部分空間の入れ替え: head の出力のうち、指定した部分空間（例: 属性の方向）の成分だけを入れ替えて、
  最後の位置の次に来る候補の対数確率を測る。
入力は、長さがそろったもの（最小対の2文の問題や、同じ問題の繰り返し）だけを扱う（詰め物が要らない）。
位置は、LanguageModel と同じく、先頭に置く区切りを数えずに、渡した列の中で数える。
"""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch

from lexical_acquisition_mechanisms.language_model import LanguageModel

# Component.head の、MLP を表す値
MLP = -1


@dataclass(frozen=True)
class Component:
    layer: int
    # head の番号（MLP なら MLP）
    head: int


@dataclass(frozen=True)
class SubspacePatch:
    """1つの head の出力のうち、部分空間の成分だけを入れ替える。"""

    layer: int
    head: int
    # 部分空間の正規直交基底（形は（head の次元, 部分空間の次元））
    basis: np.ndarray
    # 入れ替えのもとの、その head の出力（形は（位置, head の次元））
    replacement: np.ndarray


@dataclass
class ComponentOutputs:
    """1つの入力の、全ての構成要素の、指定した位置での出力（か、出力に対する勾配）。"""

    # 形は（層, head, 位置, head の次元）
    heads: torch.Tensor
    # 形は（層, 位置, 隠れ次元）
    mlps: torch.Tensor

    def at(self, indices: Sequence[int]) -> "ComponentOutputs":
        """指定した位置（取り出した位置の中での番号）だけにしたもの。"""
        return ComponentOutputs(self.heads[:, :, list(indices)], self.mlps[:, list(indices)])

    def of(self, component: Component) -> torch.Tensor:
        """1つの構成要素の値（形は（位置, 次元））。"""
        if component.head == MLP:
            return self.mlps[component.layer]
        return self.heads[component.layer, component.head]


def components(model: LanguageModel) -> list[Component]:
    """構成要素の並び。層の順に、その層の head を番号の順に並べ、最後に MLP を置く。"""
    config = model.model.config
    return [
        Component(layer, head)
        for layer in range(config.num_hidden_layers)
        for head in [*range(config.num_attention_heads), MLP]
    ]


def outputs_and_gradients(
    model: LanguageModel,
    inputs: Sequence[Sequence[int]],
    positions: Sequence[int],
    choices: Sequence[tuple[int, int | None]],
) -> tuple[np.ndarray, list[ComponentOutputs], list[ComponentOutputs]]:
    """各入力の指標 m と、positions での構成要素の出力と、その出力に対する m の勾配。

    choices は、入力ごとの（m で好む選択肢, もう一方の選択肢）のトークン番号。
    m = logit(好む選択肢) − logit(もう一方)。もう一方が全て None なら、m = log p(好む選択肢)。
    """
    token_ids = _token_tensor(model, inputs)
    columns = _columns(model, inputs, positions)
    # 重みの勾配は要らないので、重みの勾配を止め、埋め込みの出力から計算のグラフを作る
    model.model.requires_grad_(False)
    captured: dict[tuple[str, int], torch.Tensor] = {}

    def capture_head(layer: int, module, arguments) -> None:
        captured["head", layer] = arguments[0]

    def capture_mlp(layer: int, module, arguments, output: torch.Tensor) -> None:
        captured["mlp", layer] = output

    with torch.enable_grad(), _hooks(model, capture_head, capture_mlp, start_graph=True):
        metrics = _metrics(model, token_ids, choices)
        layers = range(model.number_of_layers)
        tensors = [captured["head", layer] for layer in layers]
        tensors += [captured["mlp", layer] for layer in layers]
        gradients = torch.autograd.grad(metrics.sum(), tensors, allow_unused=True)
    gradients = [
        torch.zeros_like(tensor) if gradient is None else gradient
        for tensor, gradient in zip(tensors, gradients, strict=True)
    ]
    return (
        metrics.detach().cpu().numpy(),
        _per_input(model, [tensor.detach() for tensor in tensors], columns),
        _per_input(model, gradients, columns),
    )


def estimated_effects(
    outputs: ComponentOutputs, gradients: ComponentOutputs, replacements: ComponentOutputs
) -> np.ndarray:
    """構成要素ごとに、出力を replacements に入れ替えたときの効果 e の、勾配による近似。

    並びは components と同じ。e ≈ −g·(入れ替える値 − 元の値)（位置と次元で和）。
    """
    heads = -(gradients.heads * (replacements.heads - outputs.heads)).sum(dim=(2, 3))
    mlps = -(gradients.mlps * (replacements.mlps - outputs.mlps)).sum(dim=(1, 2))
    return torch.cat([heads, mlps[:, None]], dim=1).reshape(-1).cpu().numpy()


def patched_metrics(
    model: LanguageModel,
    token_ids: Sequence[int],
    positions: Sequence[int],
    choice: tuple[int, int],
    patches: Sequence[tuple[Component, torch.Tensor]],
) -> np.ndarray:
    """1つの入力で、置き換えごとに（構成要素, positions での値）を入れ替えたときの指標 m。

    置き換えの値の形は（位置, 次元）。1回の順伝播で、batch_size 個の置き換えをまとめて測る
    （行ごとに、別の構成要素を入れ替える）。
    """
    columns = _columns(model, [token_ids], positions)
    head_dimension = _head_dimension(model)
    results = np.zeros(len(patches), dtype=np.float32)
    for start in range(0, len(patches), model.batch_size):
        chosen = patches[start : start + model.batch_size]

        def replace_head(layer: int, module, arguments, chosen=chosen) -> None:
            for row, (component, value) in enumerate(chosen):
                if component.layer == layer and component.head != MLP:
                    span = slice(
                        component.head * head_dimension, (component.head + 1) * head_dimension
                    )
                    arguments[0][row, columns, span] = value

        def replace_mlp(layer: int, module, arguments, output, chosen=chosen) -> None:
            for row, (component, value) in enumerate(chosen):
                if component.layer == layer and component.head == MLP:
                    output[row, columns] = value

        batch = _token_tensor(model, [token_ids] * len(chosen))
        with torch.inference_mode(), _hooks(model, replace_head, replace_mlp):
            metrics = _metrics(model, batch, [choice] * len(chosen))
        results[start : start + len(chosen)] = metrics.cpu().numpy()
    return results


def subspace_patched_log_probabilities(
    model: LanguageModel,
    token_ids: Sequence[int],
    positions: Sequence[int],
    rows: Sequence[Sequence[SubspacePatch]],
    candidate_ids: Sequence[int],
) -> np.ndarray:
    """1つの入力で、行ごとに head の出力の部分空間の成分を入れ替えたときの、最後の位置の次に来る
    候補の対数確率（形は（行, 候補））。

    行ごとに、入れ替え（SubspacePatch）の並びを渡す（空なら入れ替えない。複数なら全てを入れ替える）。
    出力 x を x − P x + P r にする（P は部分空間への射影、r は入れ替えのもとの値）。1回の順伝播で、
    batch_size 行をまとめて測る。
    """
    columns = _columns(model, [token_ids], positions)
    size = _head_dimension(model)
    device = model.model.device
    candidates = torch.tensor(list(candidate_ids), device=device)
    results = np.zeros((len(rows), len(candidate_ids)), dtype=np.float32)
    for start in range(0, len(rows), model.batch_size):
        chosen = [
            [
                (
                    patch,
                    torch.tensor(patch.basis @ patch.basis.T, dtype=torch.float32, device=device),
                    torch.tensor(patch.replacement, dtype=torch.float32, device=device),
                )
                for patch in row
            ]
            for row in rows[start : start + model.batch_size]
        ]

        def replace(layer: int, module, arguments, chosen=chosen) -> None:
            for row, entries in enumerate(chosen):
                for patch, projection, replacement in entries:
                    if patch.layer != layer:
                        continue
                    span = slice(patch.head * size, (patch.head + 1) * size)
                    current = arguments[0][row, columns, span]
                    changed = current - current @ projection + replacement @ projection
                    arguments[0][row, columns, span] = changed.to(current.dtype)

        batch = _token_tensor(model, [token_ids] * len(chosen))
        with torch.inference_mode(), _hooks(model, replace, lambda *arguments: None):
            hidden = model.model.model(input_ids=batch, use_cache=False).last_hidden_state
            logits = model.model.lm_head(hidden[:, -1]).float()
            values = torch.log_softmax(logits, dim=-1)[:, candidates]
        results[start : start + len(chosen)] = values.cpu().numpy()
    return results


def _head_dimension(model: LanguageModel) -> int:
    """head の出力の次元（OLMo 2 の注意と同じ求め方）。"""
    config = model.model.config
    return getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)


@contextmanager
def _hooks(
    model: LanguageModel, head_hook: Callable, mlp_hook: Callable, start_graph: bool = False
) -> Iterator[None]:
    """各層の、出力の射影の入力（head の出力）と、MLP の正規化の出力に、関数をかける。"""
    base = model.model.model
    handles = []
    if start_graph:
        handles.append(
            base.embed_tokens.register_forward_hook(
                lambda module, arguments, output: output.detach().requires_grad_(True)
            )
        )
    for index, layer in enumerate(base.layers):
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(partial(head_hook, index)))
        handles.append(
            layer.post_feedforward_layernorm.register_forward_hook(partial(mlp_hook, index))
        )
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _metrics(
    model: LanguageModel, token_ids: torch.Tensor, choices: Sequence[tuple[int, int | None]]
) -> torch.Tensor:
    """各行の、最後の位置の次に来る2つの選択肢のロジットの差（もう一方が None なら対数確率）。"""
    hidden = model.model.model(input_ids=token_ids, use_cache=False).last_hidden_state
    logits = model.model.lm_head(hidden[:, -1]).float()
    rows = torch.arange(len(choices), device=logits.device)
    preferred = torch.tensor([choice[0] for choice in choices], device=logits.device)
    if all(choice[1] is None for choice in choices):
        return torch.log_softmax(logits, dim=-1)[rows, preferred]
    other = torch.tensor([choice[1] for choice in choices], device=logits.device)
    return logits[rows, preferred] - logits[rows, other]


def _token_tensor(model: LanguageModel, inputs: Sequence[Sequence[int]]) -> torch.Tensor:
    """長さがそろった入力の先頭に区切りを置いた、トークン番号のテンソル。"""
    lengths = {len(token_ids) for token_ids in inputs}
    if len(lengths) != 1:
        raise ValueError(f"入力の長さがそろっていません（{sorted(lengths)}）。")
    length = lengths.pop()
    if not 1 <= length <= model.maximum_input_length:
        raise ValueError(f"入力の長さ {length} が、1〜{model.maximum_input_length} にありません。")
    for token_ids in inputs:
        model.tokenizer.check_token_ids(token_ids)
    end_of_text = model.tokenizer.end_of_text_id
    rows = [[end_of_text, *token_ids] for token_ids in inputs]
    return torch.tensor(rows, dtype=torch.long, device=model.model.device)


def _columns(
    model: LanguageModel, inputs: Sequence[Sequence[int]], positions: Sequence[int]
) -> torch.Tensor:
    """取り出す位置（先頭の区切りの分、1つずらす）。"""
    length = len(inputs[0])
    if not positions or not all(0 <= position < length for position in positions):
        raise ValueError(f"位置 {list(positions)} が、長さ {length} の入力の外にあります。")
    return torch.tensor([position + 1 for position in positions], device=model.model.device)


def _per_input(
    model: LanguageModel, tensors: Sequence[torch.Tensor], columns: torch.Tensor
) -> list[ComponentOutputs]:
    """層ごとの（入力, 位置, 次元）の値を、入力ごとの ComponentOutputs にする。"""
    layers = model.number_of_layers
    heads = torch.stack([tensor[:, columns] for tensor in tensors[:layers]], dim=1)
    count, _, length, _ = heads.shape
    heads = heads.view(count, layers, length, model.model.config.num_attention_heads, -1)
    heads = heads.permute(0, 1, 3, 2, 4)
    mlps = torch.stack([tensor[:, columns] for tensor in tensors[layers:]], dim=1)
    return [ComponentOutputs(heads[index], mlps[index]) for index in range(count)]
