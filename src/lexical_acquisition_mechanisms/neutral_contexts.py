"""中立文脈を選ぶ（evaluation-data select-neutral-contexts）。

中立文脈は、語の意味を示さず、直後に名詞が来やすい文の書き出し（例: "Do you know about the"）。
文脈なしと合わせた条件で、未知語の指標や課題を測る。
条件: 学習前のモデルで、どの文脈も、特定のカテゴリの語を予測しやすくしていないこと。
1. 公開されている各モデルで、候補の文脈ごとに、自然な名詞（words select が選んだもの）を直後に
   置いたときの対数確率を出し、カテゴリごとに平均する。
2. 文脈ごとに、全カテゴリの平均を引く（文脈による、名詞全体の出やすさの違いを除く）。
3. その値の、全候補での平均からの差が、どのモデル・どのカテゴリでも上限以下の候補を、前から
   順に使う。
"""

import argparse
import gc
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from lexical_acquisition_mechanisms import word_selection
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.settings import ModelSource, Settings
from lexical_acquisition_mechanisms.word_selection import SelectedWord

COMMAND = "evaluation-data select-neutral-contexts"
OUTPUT_DIRECTORY = Path("data/evaluation")
CONTEXTS_PATH = OUTPUT_DIRECTORY / "neutral_contexts.csv"
CHECKS_PATH = OUTPUT_DIRECTORY / "neutral_context_checks.csv"
BATCH_SIZE = 64


@dataclass
class NeutralContext:
    # 条件の番号（0 は文脈なし）と、文脈の文（文脈なしは空）
    number: int
    context: str


@dataclass
class NeutralContextCheck:
    context: str
    model: str
    category: str
    # 直後に置いた自然な名詞の対数確率の、カテゴリでの平均
    mean_log_probability: float
    # その平均から全カテゴリの平均を引いた値の、全候補での平均からの差
    deviation: float
    # この候補が、全モデル・全カテゴリで条件を満たしたか
    passed: bool = False


def select_neutral_contexts(settings: Settings, arguments: argparse.Namespace) -> None:
    included = {
        "neutral_context_candidates",
        "number_of_neutral_contexts",
        "maximum_neutral_context_deviation",
    }
    record_settings = {
        "evaluation": settings.evaluation.model_dump(include=included),
        "models": {name: source.model_dump() for name, source in settings.models.items()},
    }
    step = Step(
        COMMAND,
        inputs={word_selection.OUTPUT_PATH: word_selection.COMMAND},
        outputs=[CONTEXTS_PATH, CHECKS_PATH],
        settings=record_settings,
    )
    run_unless_unchanged(
        step,
        lambda: build_neutral_contexts(
            settings, _load_model, word_selection.OUTPUT_PATH, CONTEXTS_PATH, CHECKS_PATH
        ),
        arguments.recompute,
    )


def build_neutral_contexts(
    settings: Settings,
    load_model: Callable[[ModelSource], LanguageModel],
    selected_words_path: Path,
    contexts_path: Path,
    checks_path: Path,
) -> None:
    evaluation = settings.evaluation
    candidates = evaluation.neutral_context_candidates
    categories = [category.name for category in settings.words.categories]
    nouns = [
        word for word in read_rows(selected_words_path, SelectedWord) if word.role == "natural-noun"
    ]
    checks = []
    for name, source in settings.models.items():
        model = load_model(source)
        means = _category_means(model, candidates, nouns, categories)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        centered = means - means.mean(axis=1, keepdims=True)
        deviations = centered - centered.mean(axis=0, keepdims=True)
        for context_index, context in enumerate(candidates):
            for category_index, category in enumerate(categories):
                checks.append(
                    NeutralContextCheck(
                        context,
                        name,
                        category,
                        float(means[context_index, category_index]),
                        float(deviations[context_index, category_index]),
                    )
                )
    failed = {
        check.context
        for check in checks
        if abs(check.deviation) > evaluation.maximum_neutral_context_deviation
    }
    for check in checks:
        check.passed = check.context not in failed
    write_rows(checks_path, NeutralContextCheck, checks)
    chosen = [context for context in candidates if context not in failed]
    if len(chosen) < evaluation.number_of_neutral_contexts:
        raise CommandStopped(
            f"条件に合う中立文脈が {len(chosen)} 種しかありません"
            f"（{evaluation.number_of_neutral_contexts} 種必要）。結果は {checks_path} にあります。"
            "settings.yaml の evaluation.neutral_context_candidates に、"
            "同じ形の文を足してください。",
            rerun_from=COMMAND,
        )
    rows = [NeutralContext(0, "")]
    rows += [
        NeutralContext(number, context)
        for number, context in enumerate(chosen[: evaluation.number_of_neutral_contexts], start=1)
    ]
    write_rows(contexts_path, NeutralContext, rows)


def _category_means(
    model: LanguageModel,
    contexts: Sequence[str],
    nouns: Sequence[SelectedWord],
    categories: Sequence[str],
) -> np.ndarray:
    """文脈の直後に置いた名詞の対数確率の、カテゴリごとの平均（形は 文脈 × カテゴリ）。"""
    inputs = []
    spans = []
    for context in contexts:
        for noun in nouns:
            noun_ids = [int(token) for token in noun.token_ids.split()]
            token_ids, part_spans = model.tokenizer.encode_with_spans([context, noun_ids])
            inputs.append(token_ids)
            spans.append(part_spans[1])
    log_probabilities = model.token_log_probabilities(inputs)
    noun_log_probabilities = np.array(
        [
            values[span.start : span.stop].sum()
            for values, span in zip(log_probabilities, spans, strict=True)
        ]
    ).reshape(len(contexts), len(nouns))
    noun_categories = np.array([noun.category for noun in nouns])
    means = [
        noun_log_probabilities[:, noun_categories == category].mean(axis=1)
        for category in categories
    ]
    return np.stack(means, axis=1)


def _load_model(source: ModelSource) -> LanguageModel:
    return LanguageModel.load(source, batch_size=BATCH_SIZE)
