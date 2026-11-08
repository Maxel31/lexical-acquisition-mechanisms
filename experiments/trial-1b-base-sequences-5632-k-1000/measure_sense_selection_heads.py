"""手がかりで語義を選ぶ head を、自然な多義語と、2つ目の語義を足した未知語で探し、同じかを
確かめる（試行の作業。41|）。

head を止める: その head の出力（出力の射影 o_proj に入る前の部分）を、語の位置でだけ、データ全体の
語の位置での平均に置き換える（mean ablation）。手がかりを語の位置に運ぶ head を止めると、語の表現や
答えが、手がかりで動かなくなるはず。

- natural（GPU 1枚）: 学習前の 1B（未知語の獲得を始めた重み）で、RAW-C（Trott & Bergen, 2021。
  多義語・同音異義語ごとに、語義 2 つ × 文脈 2 つの 4 文）の、語の最後のトークンの層ごとの表現から、
  語義の判別 = 同じ語義の文の組の cos の平均 − 違う語義の文の組の cos の平均 を出す。止めない
  ときと、head を 1 つずつ止めたとき。続けて、半分の語で選んだ k 個の head をまとめて止め、
  残り半分の語で判別が下がるかを、同じ層から無作為に選んだ k 個と比べる（語を分けた確認）。
  → heads/natural.parquet、heads/natural_sets.parquet
- scan <run>（GPU 1枚）: 2つ目の語義を足した後の重みで、2つ目の露出回数が 0 回と 1000 回の未知語の
  課題1の答え（Y の割合）と、表現での文脈の効果（38| と同じ。層 12・16）を、止めないときと head を
  1 つずつ止めたときで測る（止める位置は、課題1 の文の中の未知語と、最小対の文の中の未知語）。
  → heads/scan/<run>.parquet
- sets <run>（GPU 1枚。natural と全ての run の scan の後）: 次の head の組をまとめて止め、2つ目の
  語義を足す前と後の重みで、同じ値を測る。→ heads/sets/<run>.parquet
  - natural-<k>: natural で、止めたときに語義の判別が最も下がった k 個
  - addition-<k>: ほかの 7 run の scan で、止めたときに文脈の効果（別のカテゴリ・1000 回）が最も
    下がった k 個（その run の未知語は、選ぶのに使わない）
  - random-<組>-<i>: <組> の head と同じ層から、無作為に選んだ k 個（RANDOM_DRAWS 通り）
- summarize（CPU）: heads_summary.txt に書く。
"""

import re
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas
import pyarrow.parquet as parquet
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from second_sense_experiment import RESULTS, weights  # noqa: E402

from lexical_acquisition_mechanisms import second_sense_assignments, word_finalization  # noqa: E402
from lexical_acquisition_mechanisms.csv_tables import read_rows
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.minimal_pairs import BLANK, MinimalPairCandidate
from lexical_acquisition_mechanisms.second_sense_assignments import SecondSenseAssignment
from lexical_acquisition_mechanisms.settings import ModelSource, load_settings
from lexical_acquisition_mechanisms.task_prompts import (
    choice_token_ids,
    encode_prompt_with_word_spans,
)
from lexical_acquisition_mechanisms.word_selection import SelectedWord

DIRECTORY = Path("outputs/partial-results/second-sense-evaluation")
# 自然な多義語の結果は学習前のモデルのものなので、どの学習の重みの解析でも同じものを使う
NATURAL = DIRECTORY / "heads"
OUTPUT = RESULTS / "heads"
RAW_C_URL = "https://raw.githubusercontent.com/seantrott/raw-c/main/data/processed/raw-c.csv"
RAW_C_PATH = Path("outputs/partial-results/natural-polysemy/raw-c.csv")
MODEL = "OLMo-2-0425-1B"
LEVELS = (0, 1000)
REPRESENTATION_LAYERS = (12, 16)
SET_SIZES = (4, 8, 16)
RANDOM_DRAWS = 10
BATCH_SIZE = 32
SEED = 0

Ablation = dict[int, list[int]]  # 層 → 止める head


class HeadAblator:
    """語の位置で head の出力を平均に置き換えながら、モデルを動かす。"""

    def __init__(self, model: LanguageModel):
        self.model = model
        config = model.model.config
        self.heads = config.num_attention_heads
        self.size = getattr(config, "head_dim", config.hidden_size // self.heads)
        self.layers = config.num_hidden_layers
        self.ablation: Ablation = {}
        self.mask: torch.Tensor | None = None
        self.means: dict[int, torch.Tensor] = {}
        self.sums: dict[int, torch.Tensor] | None = None
        self.count = 0
        attention = [block.self_attn.o_proj for block in model.model.model.layers]
        for layer, module in enumerate(attention):
            module.register_forward_pre_hook(self._hook(layer))

    def _hook(self, layer: int):
        def replace(module, arguments):
            values = arguments[0]
            batch, length, _ = values.shape
            if self.sums is not None:
                selected = values[self.mask].float().view(-1, self.heads, self.size)
                self.sums[layer] += selected.sum(dim=0)
                if layer == 0:
                    self.count += selected.shape[0]
            heads = self.ablation.get(layer)
            if not heads:
                return None
            values = values.view(batch, length, self.heads, self.size).clone()
            rows, columns = self.mask.nonzero(as_tuple=True)
            index = torch.tensor(heads, device=values.device)
            mean = self.means[layer][index].to(values.dtype)
            values[rows[:, None], columns[:, None], index[None, :]] = mean[None, :, :]
            return (values.view(batch, length, self.heads * self.size),)

        return replace

    def run(
        self,
        inputs: Sequence[Sequence[int]],
        spans: Sequence[Sequence[int]],
        reads: Sequence[int],
        ablations: dict[str, Ablation],
        layers: Sequence[int] = (),
        candidates: Sequence[int] = (),
    ) -> dict[str, dict[str, np.ndarray]]:
        """止め方ごとに、reads の位置の層ごとの表現（"layer-<l>"）と、次のトークンの候補の対数確率
        （"choices"）を返す。spans は、入力ごとの止める位置（区切りを数えない位置）。"""
        results: dict[str, dict[str, np.ndarray]] = {name: {} for name in ablations}
        blocks = self.model.model.model.layers
        captured: dict[int, torch.Tensor] = {}
        place: dict[str, torch.Tensor] = {}

        def capture(layer: int):
            def keep(module, arguments, output):
                captured[layer] = output[place["rows"], place["columns"]].float().cpu()

            return keep

        handles = [blocks[layer - 1].register_forward_hook(capture(layer)) for layer in layers]
        try:
            for indices, token_ids, attention_mask in self.model._batches(inputs):
                device = token_ids.device
                rows = torch.arange(len(indices), device=device)
                columns = torch.tensor([reads[index] + 1 for index in indices], device=device)
                place.update(rows=rows, columns=columns)
                mask = torch.zeros_like(attention_mask, dtype=torch.bool)
                for row, index in enumerate(indices):
                    mask[row, [position + 1 for position in spans[index]]] = True
                self.mask = mask
                for name, ablation in ablations.items():
                    self.ablation = ablation
                    with torch.inference_mode():
                        hidden = self.model.model.model(
                            input_ids=token_ids, attention_mask=attention_mask, use_cache=False
                        ).last_hidden_state
                        if candidates:
                            logits = self.model.model.lm_head(hidden[rows, columns]).float()
                            values = torch.log_softmax(logits, dim=-1)[:, list(candidates)]
                            _store(results[name], "choices", indices, values.cpu(), len(inputs))
                    for layer in layers:
                        _store(
                            results[name], f"layer-{layer}", indices, captured[layer], len(inputs)
                        )
        finally:
            for handle in handles:
                handle.remove()
            self.ablation = {}
        return results

    def collect_means(
        self, inputs: Sequence[Sequence[int]], spans: Sequence[Sequence[int]]
    ) -> None:
        """止めないで動かし、head ごとに、spans の位置での出力の平均を、止めるときの値にする。"""
        device = self.model.model.device
        self.sums = {
            layer: torch.zeros(self.heads, self.size, device=device) for layer in range(self.layers)
        }
        self.count = 0
        try:
            self.run(inputs, spans, [0] * len(inputs), {"none": {}})
        finally:
            sums, self.sums = self.sums, None
        self.means = {layer: total / self.count for layer, total in sums.items()}


def _store(result: dict, key: str, indices: list[int], values: torch.Tensor, total: int) -> None:
    if key not in result:
        result[key] = np.zeros((total, *values.shape[1:]), dtype=np.float32)
    result[key][indices] = values.numpy()


def cosine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return (first * second).sum(axis=-1) / (
        np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1) + 1e-12
    )


def single_head_ablations(layers: int, heads: int) -> dict[str, Ablation]:
    ablations: dict[str, Ablation] = {"none": {}}
    for layer in range(layers):
        for head in range(heads):
            ablations[f"{layer}-{head}"] = {layer: [head]}
    return ablations


def as_ablation(heads: Sequence[tuple[int, int]]) -> Ablation:
    ablation: Ablation = {}
    for layer, head in heads:
        ablation.setdefault(layer, []).append(head)
    return ablation


def random_like(
    heads: Sequence[tuple[int, int]], number_of_heads: int, generator: np.random.Generator
) -> list[tuple[int, int]]:
    """heads と同じ層から、heads に含まれない head を同じ数だけ無作為に選ぶ。"""
    chosen = []
    for layer in sorted({layer for layer, _ in heads}):
        taken = {head for other, head in heads if other == layer}
        pool = [head for head in range(number_of_heads) if head not in taken]
        count = sum(1 for other, _ in heads if other == layer)
        chosen += [(layer, int(head)) for head in generator.choice(pool, count, replace=False)]
    return chosen


def head_of(name: str) -> tuple[int, int]:
    layer, head = name.split("-")
    return int(layer), int(head)


# 自然な多義語（RAW-C）


def raw_c_sentences() -> pandas.DataFrame:
    """語・語義・文・手がかりの語（RAW-C の組の表から）。手がかりが語より前にある語だけを残す。"""
    if not RAW_C_PATH.exists():
        RAW_C_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(RAW_C_URL, str(RAW_C_PATH))
    table = pandas.read_csv(RAW_C_PATH)
    columns = ["string", "ambiguity_type", "Class"]
    sentences = pandas.concat(
        table[[*columns, f"sentence{side}", f"v{side}", f"disambiguating_word{side}"]].set_axis(
            ["word", "ambiguity", "word_class", "sentence", "version", "cue"], axis=1
        )
        for side in (1, 2)
    ).drop_duplicates(["word", "version"])
    sentences["sense"] = sentences["version"].str.split("_").str[0]
    sentences["cue_first"] = [
        _cue_before_word(sentence, cue, word)
        for sentence, cue, word in zip(
            sentences["sentence"], sentences["cue"], sentences["word"], strict=True
        )
    ]
    keep = sentences.groupby("word")["cue_first"].transform("all")
    return sentences[keep].sort_values(["word", "version"]).reset_index(drop=True)


def _plain_words(sentence: str) -> list[str]:
    return [token.strip('.,!?;:"').lower() for token in sentence.split()]


def _cue_before_word(sentence: str, cue: str, word: str) -> bool:
    tokens = _plain_words(sentence)
    if cue.lower() not in tokens or word.lower() not in tokens:
        return False
    return tokens.index(cue.lower()) < len(tokens) - 1 - tokens[::-1].index(word.lower())


def raw_c_inputs(model: LanguageModel, sentences: pandas.DataFrame):
    inputs, spans = [], []
    for word, sentence in zip(sentences["word"], sentences["sentence"], strict=True):
        start = [match.start() for match in re.finditer(rf" {re.escape(word)}\b", sentence)][-1]
        parts = [
            sentence[:start],
            model.tokenizer.word_token_ids(word),
            sentence[start + 1 + len(word) :],
        ]
        token_ids, part_spans = model.tokenizer.encode_with_spans([part for part in parts if part])
        inputs.append(token_ids)
        spans.append(list(part_spans[1]))
    return inputs, spans


def distinctions(sentences: pandas.DataFrame, states: dict[str, np.ndarray], layers: int):
    """語・層ごとの語義の判別（同じ語義の組の cos の平均 − 違う語義の組の cos の平均）。"""
    rows = []
    for word, group in sentences.groupby("word"):
        indices = list(group.index)
        senses = list(group["sense"])
        for layer in range(1, layers + 1):
            vectors = states[f"layer-{layer}"][indices]
            same, different = [], []
            for first in range(len(indices)):
                for second in range(first + 1, len(indices)):
                    value = float(cosine(vectors[first], vectors[second]))
                    (same if senses[first] == senses[second] else different).append(value)
            rows.append((word, layer, np.mean(same) - np.mean(different)))
    return rows


def natural() -> None:
    settings = load_settings(Path("settings.yaml"))
    model = LanguageModel.load(settings.models[MODEL], batch_size=BATCH_SIZE)
    ablator = HeadAblator(model)
    sentences = raw_c_sentences()
    inputs, spans = raw_c_inputs(model, sentences)
    reads = [span[-1] for span in spans]
    layers = list(range(1, ablator.layers + 1))
    ablator.collect_means(inputs, spans)
    baseline = ablator.run(inputs, spans, reads, {"none": {}}, layers)
    ablations = single_head_ablations(ablator.layers, ablator.heads)
    del ablations["none"]
    results = baseline | ablator.run(inputs, spans, reads, ablations, layers)
    rows = []
    for name, states in results.items():
        rows += [
            {"configuration": name, "word": word, "layer": layer, "distinction": value}
            for word, layer, value in distinctions(sentences, states, ablator.layers)
        ]
        print(f"natural {name}", flush=True)
    table = pandas.DataFrame(rows)
    NATURAL.mkdir(parents=True, exist_ok=True)
    table.to_parquet(NATURAL / "natural.parquet")
    # 語を分けた確認: 半分の語で選んだ head を、残り半分の語で止める
    words = sorted(sentences["word"].unique())
    generator = np.random.default_rng(SEED)
    halves = {"even": words[0::2], "odd": words[1::2]}
    readout = natural_readout_layer(table)
    set_rows = []
    for held_out, chosen_words in (("odd", halves["even"]), ("even", halves["odd"])):
        scores = natural_scores(table[table["word"].isin(chosen_words)], readout)
        evaluated = sentences[sentences["word"].isin(halves[held_out])].reset_index(drop=True)
        sub_inputs, sub_spans = raw_c_inputs(model, evaluated)
        sets: dict[str, Ablation] = {"none": {}}
        for size in SET_SIZES:
            top = [head_of(name) for name in scores.index[:size]]
            sets[f"natural-{size}"] = as_ablation(top)
            for draw in range(RANDOM_DRAWS):
                sets[f"random-natural-{size}-{draw}"] = as_ablation(
                    random_like(top, ablator.heads, generator)
                )
        outputs = ablator.run(sub_inputs, sub_spans, [s[-1] for s in sub_spans], sets, layers)
        for name, states in outputs.items():
            set_rows += [
                {
                    "held_out": held_out,
                    "configuration": name,
                    "word": word,
                    "layer": layer,
                    "distinction": value,
                }
                for word, layer, value in distinctions(evaluated, states, ablator.layers)
            ]
    pandas.DataFrame(set_rows).to_parquet(NATURAL / "natural_sets.parquet")


def natural_readout_layer(table: pandas.DataFrame) -> int:
    """止めないときに、語義の判別が最も大きい層。"""
    baseline = table[table["configuration"] == "none"].groupby("layer")["distinction"].mean()
    return int(baseline.idxmax())


def natural_scores(table: pandas.DataFrame, readout: int) -> pandas.Series:
    """head ごとの、止めたときの語義の判別の下がり方（語で平均。大きい順）。"""
    table = table[table["layer"] == readout]
    wide = table.pivot_table(index="word", columns="configuration", values="distinction")
    drops = wide.drop(columns="none").rsub(wide["none"], axis=0).mean()
    return drops.sort_values(ascending=False)


# 2つ目の語義を足した未知語


def addition_items(run: str):
    words = {word.word: word for word in read_rows(word_finalization.OUTPUT_PATH, SelectedWord)}
    assignments = [
        row
        for row in read_rows(second_sense_assignments.ASSIGNMENTS_PATH, SecondSenseAssignment)
        if row.run == run and row.second_exposure in LEVELS
    ]
    questions: dict[tuple[str, str], list[dict]] = {}
    for question in parquet.read_table(DIRECTORY / "questions.parquet").to_pylist():
        questions.setdefault((question["first_noun"], question["second_noun"]), []).append(question)
    pairs: dict[tuple[str, str], list[MinimalPairCandidate]] = {}
    for pair in read_rows(DIRECTORY / "checked_minimal_pairs.csv", MinimalPairCandidate):
        pairs.setdefault((pair.target, pair.donor), []).append(pair)
    return words, assignments, questions, pairs


def measure_addition(
    model: LanguageModel, ablator: HeadAblator, run: str, ablations: dict[str, Ablation]
) -> list[dict]:
    """止め方ごとに、未知語の課題1の Y の割合と、表現での文脈の効果を、最小対の文ごとに返す。"""
    words, assignments, questions, pairs = addition_items(run)
    ids = {name: [int(token) for token in word.token_ids.split()] for name, word in words.items()}
    # 課題1
    chosen = [
        (row, question)
        for row in assignments
        for question in questions.get((row.first_noun, row.second_noun), [])
    ]
    encoded = [
        encode_prompt_with_word_spans(
            question["prompt_template"], ids[row.unknown_word], model.tokenizer
        )
        for row, question in chosen
    ]
    inputs = [token_ids for token_ids, _ in encoded]
    spans = [list(word_spans[0]) for _, word_spans in encoded]
    candidates = sorted({choice for _, question in chosen for choice in question["choices"]})
    column = {choice: index for index, choice in enumerate(candidates)}
    candidate_ids = choice_token_ids(candidates, model.tokenizer)
    reads = [len(token_ids) - 1 for token_ids in inputs]
    ablator.collect_means(inputs, spans)
    choice_results = ablator.run(inputs, spans, reads, ablations, candidates=candidate_ids)
    rows = []
    for name, result in choice_results.items():
        for (row, question), values in zip(chosen, result["choices"], strict=True):
            first = np.exp(values[column[row.first_noun]])
            second = np.exp(values[column[row.second_noun]])
            rows.append(
                {
                    "configuration": name,
                    "measure": "second_share",
                    "unknown_word": row.unknown_word,
                    "kind": row.kind,
                    "second_exposure": row.second_exposure,
                    "token_count": row.token_count,
                    "pair": question["pair"],
                    "cue": question["cue"],
                    "value": float(second / (first + second)),
                }
            )
    # 表現での文脈の効果（未知語の位置だけで止め、X と Y の表現は止めないときのもの）
    prefixes, placed, keys = [], [], []
    for row in assignments:
        for number, pair in enumerate(pairs.get((row.first_noun, row.second_noun), [])):
            for cue, sentence in (("first", pair.target_sentence), ("second", pair.donor_sentence)):
                prefix = sentence.split(BLANK)[0].removesuffix(" ")
                for role in ("unknown", "first", "second"):
                    word = {
                        "unknown": row.unknown_word,
                        "first": row.first_noun,
                        "second": row.second_noun,
                    }[role]
                    token_ids, part_spans = model.tokenizer.encode_with_spans([prefix, ids[word]])
                    prefixes.append(token_ids)
                    placed.append(list(part_spans[1]) if role == "unknown" else [])
                    keys.append((row.unknown_word, number, cue, role))
    reads = [len(token_ids) - 1 for token_ids in prefixes]
    unknown = [index for index, key in enumerate(keys) if key[3] == "unknown"]
    ablator.collect_means([prefixes[i] for i in unknown], [placed[i] for i in unknown])
    references = ablator.run(prefixes, placed, reads, {"none": {}}, REPRESENTATION_LAYERS)["none"]
    states = ablator.run(
        [prefixes[i] for i in unknown],
        [placed[i] for i in unknown],
        [reads[i] for i in unknown],
        ablations,
        REPRESENTATION_LAYERS,
    )
    position = {keys[index][:3]: number for number, index in enumerate(unknown)}
    reference = {key: index for index, key in enumerate(keys)}
    for name, result in states.items():
        for row in assignments:
            for number in range(len(pairs.get((row.first_noun, row.second_noun), []))):
                for layer in REPRESENTATION_LAYERS:
                    lean = {}
                    for cue in ("first", "second"):
                        vector = result[f"layer-{layer}"][position[(row.unknown_word, number, cue)]]
                        lean[cue] = cosine(
                            vector,
                            references[f"layer-{layer}"][
                                reference[(row.unknown_word, number, cue, "second")]
                            ],
                        ) - cosine(
                            vector,
                            references[f"layer-{layer}"][
                                reference[(row.unknown_word, number, cue, "first")]
                            ],
                        )
                    rows.append(
                        {
                            "configuration": name,
                            "measure": f"cue_effect_{layer}",
                            "unknown_word": row.unknown_word,
                            "kind": row.kind,
                            "second_exposure": row.second_exposure,
                            "token_count": row.token_count,
                            "pair": number,
                            "cue": "both",
                            "value": float(lean["second"] - lean["first"]),
                        }
                    )
    return rows


def scan(run: str) -> None:
    model = LanguageModel.load(
        ModelSource(repository=str(weights(run, "final"))), batch_size=BATCH_SIZE
    )
    ablator = HeadAblator(model)
    ablations = single_head_ablations(ablator.layers, ablator.heads)
    rows = measure_addition(model, ablator, run, ablations)
    (OUTPUT / "scan").mkdir(parents=True, exist_ok=True)
    pandas.DataFrame(rows).assign(run=run).to_parquet(OUTPUT / "scan" / f"{run}.parquet")
    print(f"scan {run}: {len(rows)} 行", flush=True)


def context_effects(table: pandas.DataFrame) -> pandas.DataFrame:
    """未知語ごとの、課題1の文脈の効果と、表現での文脈の効果（止め方ごと）。"""
    keys = ["run", "configuration", "unknown_word", "kind", "second_exposure", "token_count"]
    shares = table[table["measure"] == "second_share"]
    wide = shares.pivot_table(index=[*keys, "pair"], columns="cue", values="value")
    choice = (wide["second"] - wide["first"]).groupby(level=keys).mean().rename("context_effect")
    representation = (
        table[table["measure"] != "second_share"]
        .pivot_table(index=[*keys, "pair"], columns="measure", values="value")
        .groupby(level=keys)
        .mean()
    )
    return pandas.concat([choice, representation], axis=1).reset_index()


def addition_scores(effects: pandas.DataFrame) -> pandas.Series:
    """head ごとの、止めたときの文脈の効果（別のカテゴリ・1000 回）の下がり方（大きい順）。"""
    effects = effects[(effects["kind"] == "other-category") & (effects["second_exposure"] == 1000)]
    wide = effects.pivot_table(
        index=["run", "unknown_word"], columns="configuration", values="context_effect"
    )
    drops = wide.drop(columns="none").rsub(wide["none"], axis=0).mean()
    return drops.sort_values(ascending=False)


def head_sets(run: str, number_of_heads: int) -> dict[str, Ablation]:
    natural_table = pandas.read_parquet(NATURAL / "natural.parquet")
    natural_ranking = natural_scores(natural_table, natural_readout_layer(natural_table))
    others = [
        pandas.read_parquet(path)
        for path in sorted((OUTPUT / "scan").glob("*.parquet"))
        if path.stem != run
    ]
    addition_ranking = addition_scores(context_effects(pandas.concat(others)))
    generator = np.random.default_rng(SEED)
    sets: dict[str, Ablation] = {"none": {}}
    for label, ranking in (("natural", natural_ranking), ("addition", addition_ranking)):
        for size in SET_SIZES:
            top = [head_of(name) for name in ranking.index[:size]]
            sets[f"{label}-{size}"] = as_ablation(top)
            for draw in range(RANDOM_DRAWS):
                sets[f"random-{label}-{size}-{draw}"] = as_ablation(
                    random_like(top, number_of_heads, generator)
                )
    return sets


def measure_sets(run: str) -> None:
    rows = []
    for checkpoint in ("before", "final"):
        model = LanguageModel.load(
            ModelSource(repository=str(weights(run, checkpoint))), batch_size=BATCH_SIZE
        )
        ablator = HeadAblator(model)
        sets = head_sets(run, ablator.heads)
        part = measure_addition(model, ablator, run, sets)
        rows += [row | {"checkpoint": checkpoint} for row in part]
        print(f"sets {run} {checkpoint}: {len(part)} 行", flush=True)
        del model, ablator
        torch.cuda.empty_cache()
    (OUTPUT / "sets").mkdir(parents=True, exist_ok=True)
    pandas.DataFrame(rows).assign(run=run).to_parquet(OUTPUT / "sets" / f"{run}.parquet")


def bootstrap(values: np.ndarray, generator: np.random.Generator, draws: int = 10000) -> str:
    means = generator.choice(values, (draws, len(values))).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return f"{values.mean():+.3f} [{low:+.3f}, {high:+.3f}]"


def summarize() -> None:
    generator = np.random.default_rng(SEED)
    lines = []
    natural_table = pandas.read_parquet(NATURAL / "natural.parquet")
    readout = natural_readout_layer(natural_table)
    baseline = natural_table[natural_table["configuration"] == "none"]
    profile = baseline.groupby("layer")["distinction"].mean()
    lines += [
        "## 自然な多義語（RAW-C、学習前の 1B）",
        f"語の数: {baseline['word'].nunique()}。語義の判別が最も大きい層: {readout}",
        "層ごとの語義の判別（止めないとき）: "
        + ", ".join(f"{layer}: {value:.3f}" for layer, value in profile.items()),
    ]
    natural_ranking = natural_scores(natural_table, readout)
    lines.append(
        "止めたときに判別が最も下がった head（層-head: 下がり方）: "
        + ", ".join(f"{name}: {value:.4f}" for name, value in natural_ranking.head(16).items())
    )
    words = sorted(baseline["word"].unique())
    even = natural_scores(natural_table[natural_table["word"].isin(words[0::2])], readout)
    odd = natural_scores(natural_table[natural_table["word"].isin(words[1::2])], readout)
    lines.append(
        "半分の語どうしの、head の下がり方の順位相関: "
        f"{even.rank().corr(odd.reindex(even.index).rank(), method='spearman'):.3f}"
    )
    natural_sets = pandas.read_parquet(NATURAL / "natural_sets.parquet")
    natural_sets = natural_sets[natural_sets["layer"] == readout]
    lines.append("残り半分の語で止めたときの、語義の判別の下がり方（語を単位にした 95% 区間）:")
    for size in SET_SIZES:
        wide = natural_sets.pivot_table(
            index=["held_out", "word"], columns="configuration", values="distinction"
        )
        chosen = (wide["none"] - wide[f"natural-{size}"]).to_numpy()
        randoms = [c for c in wide.columns if c.startswith(f"random-natural-{size}-")]
        control = (wide["none"].to_numpy()[:, None] - wide[randoms].to_numpy()).mean(axis=1)
        lines.append(
            f"  k={size}: 選んだ head {bootstrap(chosen, generator)}、"
            f"同じ層の無作為 {bootstrap(control, generator)}、"
            f"差 {bootstrap(chosen - control, generator)}"
        )
    # 2つ目の語義を足した未知語
    scans = context_effects(
        pandas.concat(
            pandas.read_parquet(path) for path in sorted((OUTPUT / "scan").glob("*.parquet"))
        )
    )
    addition_ranking = addition_scores(scans)
    matched = addition_ranking.reindex(natural_ranking.index)
    lines += [
        "",
        "## 2つ目の語義を足した未知語（別のカテゴリ・1000 回、2 の後の重み）",
        "止めたときに文脈の効果が最も下がった head: "
        + ", ".join(f"{name}: {value:.4f}" for name, value in addition_ranking.head(16).items()),
        "自然な多義語との、head の下がり方の順位相関（256 head）: "
        f"{natural_ranking.corr(matched, method='spearman'):.3f}",
        "上位 16 の重なり: "
        f"{len(set(natural_ranking.index[:16]) & set(addition_ranking.index[:16]))} 個",
    ]
    sets = pandas.concat(
        pandas.read_parquet(path) for path in sorted((OUTPUT / "sets").glob("*.parquet"))
    )
    for checkpoint in ("final", "before"):
        effects = context_effects(sets[sets["checkpoint"] == checkpoint])
        lines.append(
            f"\n### head の組をまとめて止めたとき（{checkpoint}。未知語を単位にした 95% 区間）"
        )
        for kind, level in (
            ("other-category", 1000),
            ("other-category", 0),
            ("same-category", 1000),
        ):
            subset = effects[(effects["kind"] == kind) & (effects["second_exposure"] == level)]
            for measure in ("context_effect", "cue_effect_12", "cue_effect_16"):
                wide = subset.pivot_table(
                    index=["run", "unknown_word"], columns="configuration", values=measure
                )
                lines.append(
                    f"{kind}・{level} 回・{measure}: 止めないとき {wide['none'].mean():.3f}"
                )
                for label in ("natural", "addition"):
                    for size in SET_SIZES:
                        chosen = (wide["none"] - wide[f"{label}-{size}"]).to_numpy()
                        randoms = [
                            c for c in wide.columns if c.startswith(f"random-{label}-{size}-")
                        ]
                        control = (
                            wide["none"].to_numpy()[:, None] - wide[randoms].to_numpy()
                        ).mean(axis=1)
                        lines.append(
                            f"  {label}-{size}: 下がり方 {bootstrap(chosen, generator)}、無作為 "
                            f"{bootstrap(control, generator)}、"
                            f"差 {bootstrap(chosen - control, generator)}"
                        )
    text = "\n".join(lines) + "\n"
    (RESULTS / "heads_summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    command, *values = sys.argv[1:]
    {"natural": natural, "scan": scan, "sets": measure_sets, "summarize": summarize}[command](
        *values
    )
