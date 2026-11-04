"""採用の基準を満たす head と MLP を選び、分類する（select-significant-heads-and-mlps の続き）。

ターゲット語の位置での入れ替えの効果（対照を引いたもの）で、足す語義の種類ごとに、次の3つを満たす
構成要素を採用する。
1. N が最大（4）の条件で、効果が正で、FDR の補正で有意（効果の地図の significant）
2. N に対する傾きが正: N（1・2・4）ごとの効果の平均に直線を当てはめた傾きが、片側で有意
   （ターゲット語と最小対を単位にした2段のブートストラップの標準誤差で、正規近似）
3. 半分ずつで再現する: ターゲット語18語を、カテゴリとトークン数がそろうように9語ずつに分け、
   片方で見つけた（その半分の中での FDR の補正で有意で、効果が正）構成要素が、もう片方でも
   片側で有意。どちらの向き（前半で見つけて後半で確かめる、その逆）でもよい
種類ごとの状態（adopted: 採用 / significant: 1だけ満たす / none: どちらでもない）から、
設計書 2-2 の表に従って分類する。
- sense-selection: 3つの種類のすべてで採用（属性によらない、語義選択そのものの機構）
- category: 別カテゴリと形容詞で採用、同カテゴリでは none（カテゴリの判定の候補）
- part-of-speech: 形容詞だけで採用、ほかは none（品詞の判定の候補）
- within-category: 同カテゴリだけで採用、ほかは none（同じカテゴリの中での細かい区別の候補）
- other: それ以外で、どこかで採用（個別に記述する）
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from lexical_acquisition_mechanisms.bootstrap_statistics import (
    BOOTSTRAP_SAMPLES,
    benjamini_hochberg,
    bootstrap_weights,
    mean_and_error,
    normal_p_values,
    positive_p_values,
    weighted_means,
)
from lexical_acquisition_mechanisms.sense_addition_assignments import KINDS, NUMBERS_OF_SENSES
from lexical_acquisition_mechanisms.word_selection import SelectedWord

LARGEST = max(NUMBERS_OF_SENSES)
SIGNIFICANCE_LEVEL = 0.05
FALSE_DISCOVERY_RATE = 0.05


@dataclass
class ComponentSelection:
    kind: str
    layer: int
    head: int
    # N が最大の条件での、対照を引いた効果と、効果の地図で有意か
    effect: float
    significant: bool
    # N に対する傾きと、傾きが正という片側の p 値
    slope: float
    slope_p_value: float
    # ターゲット語の前半・後半での片側の p 値と、その半分で見つけたか（FDR の補正で有意）
    first_half_p_value: float
    second_half_p_value: float
    found_in_first_half: bool
    found_in_second_half: bool
    replicated: bool
    adopted: bool


@dataclass
class ComponentClass:
    layer: int
    head: int
    # 種類ごとの状態（adopted / significant / none）
    same_category: str
    other_category: str
    adjective: str
    component_class: str


def target_halves(words: Sequence[SelectedWord], categories: Sequence[str]) -> dict[str, int]:
    """ターゲット語の、前半（0）か後半（1）か。カテゴリの番号とトークン数の和の偶奇で分ける。

    ターゲット語はカテゴリ × トークン数（1・2・3）に1語ずつなので、各トークン数で3語ずつ、
    各カテゴリで 2語と1語に分かれる。
    """
    order = {name: index for index, name in enumerate(categories)}
    return {
        word.word: (order[word.category] + word.token_count) % 2
        for word in words
        if word.role == "target"
    }


def select_components(
    differences: dict[tuple[str, int], tuple[list[tuple[str, str, str]], np.ndarray]],
    significant: dict[tuple[str, int], np.ndarray],
    halves: dict[str, int],
    layers: np.ndarray,
    heads: np.ndarray,
    seed: int,
) -> list[ComponentSelection]:
    """種類ごと・構成要素ごとの、採用の基準の値。

    differences は条件ごとの（最小対の（ターゲット語, donor, 文）, 最小対 × 構成要素の、対照を引いた
    効果）。significant は条件ごとの、効果の地図で有意で効果が正か（構成要素ごと）。
    """
    random_numbers = np.random.default_rng(seed)
    selections = []
    for kind in KINDS:
        if (kind, LARGEST) not in differences:
            continue
        slope, slope_p_values = _slopes(kind, differences, random_numbers)
        keys, values = differences[kind, LARGEST]
        half_results = []
        for half in (0, 1):
            chosen = [index for index, key in enumerate(keys) if halves[key[0]] == half]
            effect, errors = mean_and_error(
                [keys[index][0] for index in chosen], values[chosen], random_numbers
            )
            q_values = benjamini_hochberg(normal_p_values(effect, errors))
            found = (effect > 0) & (q_values < FALSE_DISCOVERY_RATE)
            half_results.append((positive_p_values(effect, errors), found))
        (first_p, first_found), (second_p, second_found) = half_results
        replicated = (first_found & (second_p < SIGNIFICANCE_LEVEL)) | (
            second_found & (first_p < SIGNIFICANCE_LEVEL)
        )
        positive = significant[kind, LARGEST]
        adopted = positive & (slope_p_values < SIGNIFICANCE_LEVEL) & replicated
        selections += [
            ComponentSelection(
                kind,
                int(layers[index]),
                int(heads[index]),
                float(values[:, index].mean()),
                bool(positive[index]),
                float(slope[index]),
                float(slope_p_values[index]),
                float(first_p[index]),
                float(second_p[index]),
                bool(first_found[index]),
                bool(second_found[index]),
                bool(replicated[index]),
                bool(adopted[index]),
            )
            for index in range(len(layers))
        ]
    return selections


def classify(selections: Sequence[ComponentSelection]) -> list[ComponentClass]:
    """どこかの種類で採用した構成要素の、種類ごとの状態と分類。"""
    states: dict[tuple[int, int], dict[str, str]] = {}
    for row in selections:
        state = "adopted" if row.adopted else "significant" if row.significant else "none"
        states.setdefault((row.layer, row.head), dict.fromkeys(KINDS, "none"))[row.kind] = state
    classes = []
    for (layer, head), state in states.items():
        adopted = {kind for kind in KINDS if state[kind] == "adopted"}
        none = {kind for kind in KINDS if state[kind] == "none"}
        if not adopted:
            continue
        if adopted == set(KINDS):
            name = "sense-selection"
        elif adopted == {"other-category", "adjective"} and "same-category" in none:
            name = "category"
        elif adopted == {"adjective"} and none == {"same-category", "other-category"}:
            name = "part-of-speech"
        elif adopted == {"same-category"} and none == {"other-category", "adjective"}:
            name = "within-category"
        else:
            name = "other"
        classes.append(ComponentClass(layer, head, *(state[kind] for kind in KINDS), name))
    return classes


def _slopes(
    kind: str,
    differences: dict[tuple[str, int], tuple[list[tuple[str, str, str]], np.ndarray]],
    random_numbers: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """N に対する傾きと、傾きが正という片側の p 値（構成要素ごと）。

    最小対は、N の条件をまたいで同じ単位として復元抽出する（足す語義は入れ子なので、同じ最小対が
    複数の N の条件にある）。
    """
    numbers = [number for number in NUMBERS_OF_SENSES if (kind, number) in differences]
    keys = sorted({key for number in numbers for key in differences[kind, number][0]})
    position = {key: index for index, key in enumerate(keys)}
    weights = bootstrap_weights([key[0] for key in keys], BOOTSTRAP_SAMPLES, random_numbers)
    means, samples = [], []
    for number in numbers:
        number_keys, values = differences[kind, number]
        size = values.shape[1]
        table = np.zeros((len(keys), size))
        present = np.zeros(len(keys), dtype=bool)
        indices = [position[key] for key in number_keys]
        table[indices] = values
        present[indices] = True
        means.append(values.mean(axis=0))
        samples.append(weighted_means(weights, table, present))
    centered = np.array(numbers, dtype=float) - np.mean(numbers)
    scale = (centered**2).sum()
    slope = np.tensordot(centered, np.array(means), axes=1) / scale
    sampled = np.tensordot(centered, np.array(samples), axes=(0, 0)) / scale
    return slope, positive_p_values(slope, sampled.std(axis=0))
