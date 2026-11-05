"""属性を判定する head を、3つの基準で決めて保存する（test-attribute-directions の続き）。

採用した head × 属性（カテゴリ・品詞）ごとに、次の3つを満たすものを「属性を判定する head」とする。
1. 符号化: 属性のプローブが、(a) 確認用の自然な語と、(b) 属性が違う条件の run（カテゴリは
   別カテゴリ、品詞は形容詞。足す語義の数は 4）の最小対で、偶然の水準を有意に超える
   （train-attribute-probes）
2. 因果: (b) と同じ run の課題2で、属性の方向の入れ替えの効果（s_Y の正解の対数確率の増え方）が、
   2つの比較の基準（同じ層の無作為な head・同じ次元数の無作為な方向）のどちらよりも大きい
3. 特異性: 課題1（m の変化）で、属性の方向の入れ替えが比較の基準より効く条件が、属性が違う条件と
   合う（その属性が違う条件でだけ効く）。カテゴリは別カテゴリと形容詞で効き、同カテゴリでは効かない。
   品詞は形容詞で効き、同カテゴリと別カテゴリでは効かない
「比較の基準より大きい」は、最小対の各方向での差（属性の方向 − 基準）を最小対の両方向で平均し、
ターゲット語と最小対を単位にした2段のブートストラップで、片側 p < 0.05 とする。
出力は outputs/results/sense-addition/<モデル>/ の attribute_direction_tests.csv（run × 課題
ごとの比較）・attribute_judging_heads.csv（3つの基準）・attribute_judging_heads.npz（属性を判定する
head の、属性の方向の正規直交基底 "{層}-{head}-{属性}-basis" と、自然な語での head の出力の平均
"{層}-{head}-{属性}-mean"。接続で使う）。
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lexical_acquisition_mechanisms.attribute_directions import attribute_basis
from lexical_acquisition_mechanisms.attribute_probes import (
    CUES,
    AttributeProbeAccuracy,
    AttributeProbeTransfer,
    read_attribute_probes,
)
from lexical_acquisition_mechanisms.bootstrap_statistics import mean_and_error, positive_p_values
from lexical_acquisition_mechanisms.component_adoption import LARGEST
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.task_accuracy import SIGNIFICANCE_LEVEL

# 属性ごとの、属性が違う条件の run と、効くべき種類・効かないべき種類
RELEVANT_RUNS = {"category": f"other-category-{LARGEST}", "part-of-speech": f"adjective-{LARGEST}"}
EFFECTIVE_KINDS = {"category": ("other-category", "adjective"), "part-of-speech": ("adjective",)}
INEFFECTIVE_KINDS = {
    "category": ("same-category",),
    "part-of-speech": ("same-category", "other-category"),
}
BASELINES = ("random-head", "random-direction")


@dataclass
class AttributeDirectionTest:
    layer: int
    head: int
    attribute: str
    run: str
    # sense（課題1）か、課題2（category か part-of-speech）
    task: str
    minimal_pairs: int
    # 入れ替えごとの、s_Y の側への動きの平均と、s_Y の答えになった割合
    attribute_change: float
    random_head_change: float
    random_direction_change: float
    attribute_switch_rate: float
    random_head_switch_rate: float
    random_direction_switch_rate: float
    # 属性の方向の効果が、それぞれの基準より大きいという片側の p 値（基準がなければ nan）
    random_head_p_value: float
    random_direction_p_value: float
    exceeds_baselines: bool


@dataclass
class AttributeJudgingHead:
    layer: int
    head: int
    attribute: str
    held_out_accuracy: float
    # (b) の、donor の語義を示す文と、元の語義を示す文での正解率
    donor_cue_accuracy: float
    target_cue_accuracy: float
    encoding: bool
    causal: bool
    specificity: bool
    judging: bool


def direction_tests(rows: Sequence[dict[str, object]], seed: int) -> list[AttributeDirectionTest]:
    """head × 属性 × run × 課題ごとの、属性の方向と比較の基準の効果の比べ。"""
    grouped: dict[tuple, dict[str, dict[tuple, dict[str, object]]]] = {}
    for row in rows:
        key = (row["layer"], row["head"], row["attribute"], row["run"], row["task"])
        pair = (row["target"], row["donor"], row["donor_sentence"], row["direction"])
        grouped.setdefault(key, {}).setdefault(row["patch"], {})[pair] = row
    random_numbers = np.random.default_rng(seed)
    tests = []
    for (layer, head, attribute, run, task), patches in grouped.items():
        pairs = sorted(patches["attribute"])
        minimal_pairs = sorted({pair[:3] for pair in pairs})
        means, rates, p_values = {}, {}, {}
        for patch in ("attribute", *BASELINES):
            values = patches.get(patch, {})
            means[patch] = _mean(
                [values[pair]["metric_change"] for pair in pairs if pair in values]
            )
            rates[patch] = _mean([values[pair]["switched"] for pair in pairs if pair in values])
        for baseline in BASELINES:
            if baseline not in patches:
                p_values[baseline] = float("nan")
                continue
            # 最小対の両方向で平均した、属性の方向 − 基準
            differences = np.array(
                [
                    np.mean(
                        [
                            patches["attribute"][pair]["metric_change"]
                            - patches[baseline][pair]["metric_change"]
                            for pair in pairs
                            if pair[:3] == minimal_pair
                        ]
                    )
                    for minimal_pair in minimal_pairs
                ]
            )[:, None]
            targets = [minimal_pair[0] for minimal_pair in minimal_pairs]
            effect, error = mean_and_error(targets, differences, random_numbers)
            p_values[baseline] = float(positive_p_values(effect, error)[0])
        exceeds = all(
            np.isnan(p_values[baseline]) or p_values[baseline] < SIGNIFICANCE_LEVEL
            for baseline in BASELINES
        ) and not all(np.isnan(p_values[baseline]) for baseline in BASELINES)
        tests.append(
            AttributeDirectionTest(
                layer,
                head,
                attribute,
                run,
                task,
                len(minimal_pairs),
                *(means[patch] for patch in ("attribute", *BASELINES)),
                *(rates[patch] for patch in ("attribute", *BASELINES)),
                *(p_values[baseline] for baseline in BASELINES),
                exceeds,
            )
        )
    return tests


def judging_heads(
    tests: Sequence[AttributeDirectionTest],
    accuracy: Sequence[AttributeProbeAccuracy],
    transfer: Sequence[AttributeProbeTransfer],
) -> list[AttributeJudgingHead]:
    """採用した head × 属性ごとの、3つの基準と、属性を判定する head か。"""
    held_out = {(row.layer, row.head, row.attribute): row for row in accuracy if not row.baseline}
    transferred = {(row.layer, row.head, row.attribute, row.run, row.cue): row for row in transfer}
    exceeds = {(test.layer, test.head, test.attribute, test.run, test.task): test for test in tests}
    results = []
    for (layer, head, attribute), probe_accuracy in sorted(held_out.items()):
        relevant = RELEVANT_RUNS[attribute]
        moved = [transferred.get((layer, head, attribute, relevant, cue)) for cue in CUES]
        # 手がかりが示す語義の属性を当てる: どちらの手がかりの文でも、偶然の水準を超える
        encoding = probe_accuracy.significant and all(
            row is not None and row.significant for row in moved
        )
        test = exceeds.get((layer, head, attribute, relevant, attribute))
        causal = test is not None and test.exceeds_baselines
        # 課題1で、比較の基準より効く種類
        effective = {
            kind
            for kind in (*EFFECTIVE_KINDS[attribute], *INEFFECTIVE_KINDS[attribute])
            if (sense := exceeds.get((layer, head, attribute, f"{kind}-{LARGEST}", "sense")))
            and sense.exceeds_baselines
        }
        specificity = set(EFFECTIVE_KINDS[attribute]) <= effective and not (
            set(INEFFECTIVE_KINDS[attribute]) & effective
        )
        results.append(
            AttributeJudgingHead(
                layer,
                head,
                attribute,
                probe_accuracy.held_out_accuracy,
                *(row.accuracy if row is not None else float("nan") for row in moved),
                encoding,
                causal,
                specificity,
                encoding and causal and specificity,
            )
        )
    return results


def write_attribute_judgments(
    rows: Sequence[dict[str, object]],
    input_paths: tuple[Path, Path, Path],
    outputs: tuple[Path, Path, Path],
    seed: int,
) -> None:
    accuracy_path, transfer_path, probes_path = input_paths
    tests_path, heads_path, directions_path = outputs
    tests = direction_tests(rows, seed)
    write_rows(tests_path, AttributeDirectionTest, tests)
    heads = judging_heads(
        tests,
        read_rows(accuracy_path, AttributeProbeAccuracy),
        read_rows(transfer_path, AttributeProbeTransfer),
    )
    write_rows(heads_path, AttributeJudgingHead, heads)
    probes = read_attribute_probes(probes_path) if heads else {}
    arrays = {}
    for row in heads:
        if row.judging:
            probe = probes[row.layer, row.head, row.attribute]
            prefix = f"{row.layer}-{row.head}-{row.attribute}"
            arrays[f"{prefix}-basis"] = attribute_basis(probe)
            # プローブの特徴の平均は、学習用の自然な語での、その head の出力の平均
            arrays[f"{prefix}-mean"] = probe.mean
    np.savez(directions_path, **arrays)


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if len(values) else float("nan")
