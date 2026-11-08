"""2つ目の語義の追加で変わった重みを部品ごとに移し、使い分けを担う部品を探す（試行の作業。40|）。

2つ目の語義を足す前の重みを θ1（未知語の獲得の最後の重み）、足した後の重みを θ2、差を Δ = θ2 − θ1
とする。部品 S ごとに、次の2つのモデルで課題1を解かせ、選択肢ごとの対数確率を書く。
- sufficiency: θ1 + Δ_S（S の変化だけを足す。文脈の効果が戻れば、S の変化で足りる）
- necessity: θ2 − Δ_S（S の変化だけを戻す。文脈の効果が消えれば、S の変化が要る）
部品は、埋め込み、各層の attention（正規化を含む）、各層の MLP（正規化を含む）、最後の正規化、
出力の行列と、それらをまとめたもの（全ての attention・全ての MLP・4層ずつの層）。
解かせるのは、2つ目の露出回数が 0 回（対照）と 1000 回の未知語の問題。
同じことを、未知語が既知語になるフェーズ（39|）にも行える（phase に first を渡す）。そのときは、
θ1 を学習前の重み、θ2 を未知語の獲得の最後の重みとし、run の全ての未知語で、X の手がかりの文で
X を選ぶ割合（1つ目の語義の獲得）を測る。
- measure <phase> <run>（GPU 1枚。phase は first か second）:
  outputs/partial-results/second-sense-evaluation/grafting-<phase>/<run>.parquet
- summarize <phase>（CPU）: 部品ごとに、種類 × 2つ目の露出回数の値（second は文脈の効果、first は
  X の手がかりの文で X を選ぶ割合）の平均と、θ1 から θ2 までの変化のうち、そのモデルが戻した割合
  （recovered）を grafting-<phase>_summary.txt に書く。
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas
import pyarrow as arrow
import pyarrow.parquet as parquet
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from save_sense_choice_probabilities import choice_log_probabilities  # noqa: E402

from lexical_acquisition_mechanisms import second_sense_assignments, word_finalization  # noqa: E402
from lexical_acquisition_mechanisms.csv_tables import read_rows  # noqa: E402
from lexical_acquisition_mechanisms.language_model import LanguageModel  # noqa: E402
from lexical_acquisition_mechanisms.run_settings import first_phase_weights  # noqa: E402
from lexical_acquisition_mechanisms.second_sense_assignments import (  # noqa: E402
    SecondSenseAssignment,
)
from lexical_acquisition_mechanisms.settings import ModelSource, load_settings  # noqa: E402
from lexical_acquisition_mechanisms.word_selection import SelectedWord  # noqa: E402
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer  # noqa: E402

DIRECTORY = Path("outputs/partial-results/second-sense-evaluation")
MODEL = "OLMo-2-0425-1B"
LEVELS = (0, 1000)
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


def measure(phase: str, run: str) -> None:
    words = {word.word: word for word in read_rows(word_finalization.OUTPUT_PATH, SelectedWord)}
    questions = parquet.read_table(DIRECTORY / "questions.parquet").to_pylist()
    by_pair: dict[tuple[str, str], list[dict]] = {}
    for question in questions:
        by_pair.setdefault((question["first_noun"], question["second_noun"]), []).append(question)
    chosen = [
        (row, question)
        for row in read_rows(second_sense_assignments.ASSIGNMENTS_PATH, SecondSenseAssignment)
        if row.run == run and (phase == "first" or row.second_exposure in LEVELS)
        for question in by_pair.get((row.first_noun, row.second_noun), [])
    ]
    ids = [[int(token) for token in words[row.unknown_word].token_ids.split()] for row, _ in chosen]
    if phase == "first":
        before = load_settings(Path("settings.yaml")).models[MODEL]
        after = first_phase_weights(MODEL, run)
    else:
        before = ModelSource(repository=str(first_phase_weights(MODEL, run)))
        after = (
            Path("outputs/checkpoints")
            / second_sense_assignments.EXPERIMENT
            / MODEL
            / run
            / "final"
        )
    network = AutoModelForCausalLM.from_pretrained(
        before.repository, revision=before.revision, dtype=torch.float32
    ).to("cuda")
    model = LanguageModel(network, WordTokenizer.from_model(before))
    parameters = dict(network.named_parameters())
    first = {name: value.detach().clone() for name, value in parameters.items()}
    second = load_file(str(after / "model.safetensors"), device="cuda")
    delta = {name: second[name].float() - first[name] for name in first}
    del second
    owner = {name: component(name) for name in first}
    configurations = [("before", "baseline", set()), ("after", "baseline", None)]
    for name, members in component_sets(network.config.num_hidden_layers).items():
        configurations += [(name, "sufficiency", members), (name, "necessity", members)]
    rows = []
    for name, mode, members in configurations:
        with torch.no_grad():
            for parameter_name, parameter in parameters.items():
                moved = members is None or owner[parameter_name] in members
                if mode == "necessity":
                    moved = not moved
                parameter.copy_(first[parameter_name] + (delta[parameter_name] if moved else 0))
        values = choice_log_probabilities(model, [question for _, question in chosen], ids)
        rows += [
            {
                "run": run,
                "configuration": name,
                "mode": mode,
                "unknown_word": row.unknown_word,
                "second_exposure": row.second_exposure,
                "kind": row.kind,
                "first_noun": row.first_noun,
                "second_noun": row.second_noun,
                "pair": question["pair"],
                "cue": question["cue"],
                "choices": question["choices"],
                "log_probabilities": value,
            }
            for (row, question), value in zip(chosen, values, strict=True)
        ]
        print(f"{run} {mode} {name}", flush=True)
    (DIRECTORY / f"grafting-{phase}").mkdir(parents=True, exist_ok=True)
    parquet.write_table(
        arrow.Table.from_pylist(rows), DIRECTORY / f"grafting-{phase}" / f"{run}.parquet"
    )


def summarize(phase: str) -> None:
    table = pandas.concat(
        parquet.read_table(path).to_pandas()
        for path in sorted((DIRECTORY / f"grafting-{phase}").glob("*.parquet"))
    )
    probabilities = [np.exp(np.array(value)) for value in table["log_probabilities"]]
    table["share"] = [
        row[list(choices).index(second)] / row.sum()
        for row, choices, second in zip(
            probabilities, table["choices"], table["second_noun"], strict=True
        )
    ]
    keys = ["configuration", "mode", "run", "unknown_word", "second_exposure", "kind", "pair"]
    wide = table.pivot_table(index=keys, columns="cue", values="share")
    # second: 文脈の効果。first: X の手がかりの文で X を選ぶ割合
    wide["effect"] = wide["second"] - wide["first"] if phase == "second" else 1 - wide["first"]
    per_word = wide.groupby(level=keys[:6])["effect"].mean()
    effect = per_word.groupby(level=["configuration", "mode", "kind", "second_exposure"]).mean()
    effect = effect.unstack(["kind", "second_exposure"])
    start, end = effect.loc[("before", "baseline")], effect.loc[("after", "baseline")]
    recovered = (effect - start) / (end - start)
    text = (
        f"## {'文脈の効果' if phase == 'second' else 'X の手がかりの文で X を選ぶ割合'}"
        "（未知語ごとの平均の平均）\n"
        + effect.round(3).to_string()
        + "\n\n## θ1 から θ2 までの変化のうち、戻した割合（recovered）\n"
        + recovered.round(2).to_string()
        + "\n"
    )
    (DIRECTORY / f"grafting-{phase}_summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    if sys.argv[1] == "measure":
        measure(sys.argv[2], sys.argv[3])
    else:
        summarize(sys.argv[2])
