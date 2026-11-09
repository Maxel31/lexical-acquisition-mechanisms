"""2つ目の語義の追加で、手がかりで語義を選べるかを、次のトークンの分布全体で測る（試行の作業。55|）。

課題1の文脈の効果（measure_second_sense_context_effects.py）は、"Answer:" の次の、2つの選択肢の語の
確率だけを比べる。ここでは、問題の形式に頼らず、分布全体で同じ向きの値を測り、課題1を確かめる。
最小対の文（checked_minimal_pairs.csv）の空欄の前まで（手がかりを含む）に続けて、未知語・X（1つ目の
名詞）・Y（2つ目の名詞）をそれぞれ置き、語の最後のトークンの次に来るトークンの分布を出す。
- 距離: Jensen–Shannon 距離 d(p, q) = √JS(p, q)（自然対数。距離の公理を満たす）
- Y への近さ: r = (d(未知語, X) − d(未知語, Y)) ÷ d(X, Y)。三角不等式から −1〜1 に入り、
  −1 なら X と、1 なら Y と同じ分布。
- 文脈の効果: Y の手がかりの文での r − X の手がかりの文での r（課題1の文脈の効果と同じ向き）
使い方（リポジトリの最上位のディレクトリで）:
- measure <run>（GPU 1枚）: 2つ目の語義を足す前と、足した後の学習の 25・50・75・100% の重みで、
  run の全ての未知語の JS を、distribution/<run>.parquet に書く。
- summarize（CPU）: 重み × 種類 × 2つ目の露出回数ごとの文脈の効果の平均、足す前からの増え方、
  2つ目の露出回数が 0 回の未知語（対照）との差と、課題1の文脈の効果との順位相関を
  distribution_summary.txt に書く。
"""

import gc
import math
import sys
from pathlib import Path

import numpy as np
import pandas
import pyarrow as arrow
import pyarrow.parquet as parquet
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from second_sense_experiment import DIRECTORY, RESULTS, weights  # noqa: E402

from lexical_acquisition_mechanisms import second_sense_assignments, word_finalization  # noqa: E402
from lexical_acquisition_mechanisms.csv_tables import read_rows  # noqa: E402
from lexical_acquisition_mechanisms.language_model import LanguageModel  # noqa: E402
from lexical_acquisition_mechanisms.minimal_pairs import BLANK, MinimalPairCandidate  # noqa: E402
from lexical_acquisition_mechanisms.second_sense_assignments import (  # noqa: E402
    SecondSenseAssignment,
)
from lexical_acquisition_mechanisms.settings import ModelSource  # noqa: E402
from lexical_acquisition_mechanisms.word_selection import SelectedWord  # noqa: E402

CHECKPOINTS = ["before", "checkpoint-25", "checkpoint-50", "checkpoint-75", "final"]
ROLES = ("unknown", "first", "second")
# 一度に分布を出す最小対の文の数（入力は、その 3 倍）
CHUNK = 64
BATCH_SIZE = 96
DRAWS = 10000
SEED = 0


def jensen_shannon(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """対数確率の行どうしの Jensen–Shannon ダイバージェンス（自然対数。0〜log 2）。"""
    middle = torch.logaddexp(first, second) - math.log(2)
    value = 0.5 * (first.exp() * (first - middle)).sum(-1)
    value += 0.5 * (second.exp() * (second - middle)).sum(-1)
    return value.clamp(min=0)


def next_token_log_probabilities(model: LanguageModel, inputs: list[list[int]]) -> torch.Tensor:
    """各入力の最後のトークンの次に来るトークンの、全ての語彙の対数確率（float64、GPU 上）。"""
    values = torch.empty(
        (len(inputs), model.model.lm_head.out_features),
        dtype=torch.float64,
        device=model.model.device,
    )
    for indices, token_ids, attention_mask in model._batches(inputs):
        rows, columns = model._rows_and_columns(indices, [len(tokens) - 1 for tokens in inputs])
        with torch.inference_mode():
            hidden = model._last_hidden_state(token_ids, attention_mask)
            logits = model.model.lm_head(hidden[rows, columns]).float()
            values[indices] = torch.log_softmax(logits, dim=-1).double()
    return values


def items(run: str) -> tuple[dict[str, list[int]], list[tuple]]:
    """run の未知語ごとの、(割り当て, 最小対の番号, 手がかり, 空欄の前までの文) の組。"""
    words = {
        word.word: [int(token) for token in word.token_ids.split()]
        for word in read_rows(word_finalization.OUTPUT_PATH, SelectedWord)
    }
    pairs: dict[tuple[str, str], list[MinimalPairCandidate]] = {}
    for pair in read_rows(DIRECTORY / "checked_minimal_pairs.csv", MinimalPairCandidate):
        pairs.setdefault((pair.target, pair.donor), []).append(pair)
    chosen = []
    for row in read_rows(second_sense_assignments.ASSIGNMENTS_PATH, SecondSenseAssignment):
        if row.run != run:
            continue
        for number, pair in enumerate(pairs.get((row.first_noun, row.second_noun), [])):
            for cue, sentence in (("first", pair.target_sentence), ("second", pair.donor_sentence)):
                chosen.append((row, number, cue, sentence.split(BLANK)[0].removesuffix(" ")))
    return words, chosen


def measure(run: str) -> None:
    words, chosen = items(run)
    rows = []
    for checkpoint in CHECKPOINTS:
        model = LanguageModel.load(
            ModelSource(repository=str(weights(run, checkpoint))), batch_size=BATCH_SIZE
        )
        for start in range(0, len(chosen), CHUNK):
            part = chosen[start : start + CHUNK]
            inputs = [
                model.tokenizer.encode_with_spans([prefix, words[word]])[0]
                for row, _, _, prefix in part
                for word in (row.unknown_word, row.first_noun, row.second_noun)
            ]
            values = next_token_log_probabilities(model, inputs)
            unknown, first, second = (values[index :: len(ROLES)] for index in range(len(ROLES)))
            divergences = {
                "js_unknown_first": jensen_shannon(unknown, first).tolist(),
                "js_unknown_second": jensen_shannon(unknown, second).tolist(),
                "js_first_second": jensen_shannon(first, second).tolist(),
            }
            rows += [
                {
                    "run": run,
                    "checkpoint": checkpoint,
                    "unknown_word": row.unknown_word,
                    "token_count": row.token_count,
                    "second_exposure": row.second_exposure,
                    "kind": row.kind,
                    "first_noun": row.first_noun,
                    "second_noun": row.second_noun,
                    "pair": number,
                    "cue": cue,
                    **{name: value[index] for name, value in divergences.items()},
                }
                for index, (row, number, cue, _) in enumerate(part)
            ]
        print(f"{run} {checkpoint}: {len(chosen)} 文", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    (RESULTS / "distribution").mkdir(parents=True, exist_ok=True)
    parquet.write_table(arrow.Table.from_pylist(rows), RESULTS / "distribution" / f"{run}.parquet")


def per_word_effects(table: pandas.DataFrame, value: str) -> pandas.Series:
    """未知語ごとの文脈の効果（最小対ごとの、Y の手がかりの文 − X の手がかりの文の、平均）。"""
    keys = ["run", "checkpoint", "unknown_word", "second_exposure", "kind"]
    wide = table.pivot_table(index=[*keys, "pair"], columns="cue", values=value)
    return (wide["second"] - wide["first"]).groupby(level=keys).mean()


def choice_effects() -> pandas.Series | None:
    """課題1の、未知語ごとの文脈の効果（Y の割合 = P(Y) ÷ (P(X) + P(Y)) で測ったもの）。"""
    paths = sorted((RESULTS / "context").glob("*.parquet"))
    if not paths:
        return None
    table = pandas.concat(parquet.read_table(path).to_pandas() for path in paths)
    table["second_share"] = [
        np.exp(values[list(choices).index(second)]) / np.exp(np.array(values)).sum()
        for values, choices, second in zip(
            table["log_probabilities"], table["choices"], table["second_noun"], strict=True
        )
    ]
    return per_word_effects(table, "second_share")


def interval(values: np.ndarray, generator: np.random.Generator) -> str:
    """平均と、未知語を単位にした 95% の bootstrap 区間。"""
    means = generator.choice(values, (DRAWS, len(values))).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return f"{values.mean():+.3f} [{low:+.3f}, {high:+.3f}]"


def summarize() -> None:
    generator = np.random.default_rng(SEED)
    table = pandas.concat(
        parquet.read_table(path).to_pandas()
        for path in sorted((RESULTS / "distribution").glob("*.parquet"))
    )
    distance = {
        name: np.sqrt(table[f"js_{name}"])
        for name in ("unknown_first", "unknown_second", "first_second")
    }
    table["closeness"] = (distance["unknown_first"] - distance["unknown_second"]) / distance[
        "first_second"
    ]
    table["divergence_difference"] = table["js_unknown_first"] - table["js_unknown_second"]
    effects = pandas.DataFrame(
        {
            "context_effect": per_word_effects(table, "closeness"),
            "divergence_effect": per_word_effects(table, "divergence_difference"),
        }
    )
    groups = ["kind", "second_exposure", "checkpoint"]
    flat = effects.reset_index()
    flat["checkpoint"] = pandas.Categorical(flat["checkpoint"], CHECKPOINTS, ordered=True)
    grouped = flat.groupby(groups, observed=True)
    summary = grouped[["context_effect", "divergence_effect"]].mean()
    summary["words"] = grouped.size()
    lines = [
        "## 分布での文脈の効果（未知語ごとの平均の平均）",
        "context_effect: Y への近さ r（JS 距離で、X と Y の間のどこにあるか。−1〜1）の、"
        "Y の手がかりの文 − X の手がかりの文",
        "divergence_effect: JS(未知語, X) − JS(未知語, Y) の、同じ差",
        summary.round(4).to_string(),
        "",
        "## 足す前（before）から最後（final）までの、r での文脈の効果の増え方"
        "（未知語ごとの差の平均と 95% 区間）と、2つ目の露出回数 0 回（対照）との差",
    ]
    wide = effects["context_effect"].unstack("checkpoint")
    increase = (wide["final"] - wide["before"]).reset_index()
    for kind, by_kind in increase.groupby("kind"):
        control = by_kind.loc[by_kind["second_exposure"] == 0, 0].to_numpy()
        for exposure, group in by_kind.groupby("second_exposure"):
            values = group[0].to_numpy()
            draws = generator.choice(values, (DRAWS, len(values))).mean(axis=1)
            draws -= generator.choice(control, (DRAWS, len(control))).mean(axis=1)
            low, high = np.quantile(draws, [0.025, 0.975])
            lines.append(
                f"{kind} {exposure} 回: 増え方 {interval(values, generator)}、"
                f"対照との差 {values.mean() - control.mean():+.3f} [{low:+.3f}, {high:+.3f}]"
            )
    choice = choice_effects()
    if choice is not None:
        lines += ["", "## 課題1の文脈の効果との、未知語ごとの順位相関（Spearman）"]
        both = pandas.concat([effects["context_effect"], choice.rename("choice")], axis=1).dropna()
        both = both.reset_index()
        for checkpoint in ("before", "final"):
            for kind, group in both[both["checkpoint"] == checkpoint].groupby("kind"):
                correlation = group["context_effect"].rank().corr(group["choice"].rank())
                lines.append(f"{checkpoint} {kind}: {correlation:.3f}（{len(group)} 語）")
        rise = both.pivot_table(
            index=["run", "unknown_word", "kind"],
            columns="checkpoint",
            values=["context_effect", "choice"],
        )
        rise = pandas.DataFrame(
            {
                name: rise[(name, "final")] - rise[(name, "before")]
                for name in ("context_effect", "choice")
            }
        ).reset_index()
        for kind, group in rise.groupby("kind"):
            correlation = group["context_effect"].rank().corr(group["choice"].rank())
            lines.append(f"増え方 {kind}: {correlation:.3f}（{len(group)} 語）")
    text = "\n".join(lines) + "\n"
    (RESULTS / "distribution_summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    if sys.argv[1] == "measure":
        measure(sys.argv[2])
    else:
        summarize()
