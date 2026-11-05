"""課題を解かせる部品と、学習前のモデルでの課題の確認。

- answer_questions: 問題文（語は {word}）の後に来る選択肢の語の対数確率を比べ、最も大きいものを
  答えにする。
- 課題の正解率（data/evaluation/task_accuracy.csv）: 自然な語での確認の問題をモデルごとに解かせ、
  偶然の水準（1 ÷ 選択肢の数）を超えるかを、片側の二項検定（有意水準 SIGNIFICANCE_LEVEL）で
  確かめる。超えない課題は、そのモデルでは使わない（設計書「下流課題の共通形式」）。
- 言い換えの語の確認（target_paraphrase_checks.csv）: 最小対のターゲット語の手がかりの文で、
  ターゲット語の課題1を、言い換えの語と donor の2択で解かせ、言い換えの語を選ぶ割合をターゲット語
  ごとに出す（元の語義の代表語が、学習前のモデルで元の語に対して選ばれることの確認）。
"""

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.minimal_pairs import BLANK, MinimalPairCandidate
from lexical_acquisition_mechanisms.task_prompts import (
    WORD,
    TaskItem,
    choice_token_ids,
    encode_prompt,
    prompt_template,
)

SIGNIFICANCE_LEVEL = 0.05


@dataclass
class TaskAccuracy:
    model: str
    task: str
    questions: int
    correct: int
    accuracy: float
    # 偶然の水準と、それを超えるかの片側の二項検定の p 値
    chance: float
    p_value: float
    usable: bool


def answer_questions(
    model: LanguageModel,
    templates: Sequence[str],
    word_ids: Sequence[Sequence[int]],
    choices: Sequence[Sequence[str]],
) -> list[str]:
    """問題ごとに、"Answer:" の次に来る対数確率が最も大きい選択肢。"""
    inputs = [
        encode_prompt(template, ids, model.tokenizer)
        for template, ids in zip(templates, word_ids, strict=True)
    ]
    candidates = sorted({choice for question in choices for choice in question})
    candidate_ids = choice_token_ids(candidates, model.tokenizer)
    positions = [len(token_ids) - 1 for token_ids in inputs]
    values = model.next_token_log_probabilities(inputs, positions, candidate_ids)
    column = {choice: index for index, choice in enumerate(candidates)}
    return [
        max(question, key=lambda choice: values[row, column[choice]])
        for row, question in enumerate(choices)
    ]


def binomial_p_value(successes: int, trials: int, chance: float) -> float:
    """偶然の水準 chance で、trials 回のうち successes 回以上当たる確率（片側の二項検定）。"""
    if trials == 0:
        return 1.0
    terms = [
        math.exp(
            math.lgamma(trials + 1)
            - math.lgamma(count + 1)
            - math.lgamma(trials - count + 1)
            + count * math.log(chance)
            + (trials - count) * math.log1p(-chance)
        )
        for count in range(successes, trials + 1)
    ]
    return min(1.0, sum(terms))


def natural_task_accuracy(
    model: LanguageModel, model_name: str, questions: Sequence[Mapping[str, object]]
) -> list[TaskAccuracy]:
    """自然な語での確認の問題の、課題ごとの正解率。"""
    answers = answer_questions(
        model,
        [question["prompt_template"] for question in questions],
        [question["word_token_ids"] for question in questions],
        [question["choices"] for question in questions],
    )
    results = []
    for task in sorted({question["task"] for question in questions}):
        rows = [
            (question, answer)
            for question, answer in zip(questions, answers, strict=True)
            if question["task"] == task
        ]
        correct = sum(question["answer"] == answer for question, answer in rows)
        chance = float(np.mean([1 / len(question["choices"]) for question, _ in rows]))
        p_value = binomial_p_value(correct, len(rows), chance)
        results.append(
            TaskAccuracy(
                model_name,
                task,
                len(rows),
                correct,
                correct / len(rows),
                chance,
                p_value,
                p_value < SIGNIFICANCE_LEVEL,
            )
        )
    return results


def paraphrase_choices(
    model: LanguageModel,
    candidates: Sequence[MinimalPairCandidate],
    target_ids: Mapping[str, list[int]],
    paraphrases: Mapping[str, str],
    examples: Sequence[tuple[TaskItem, str]],
    number_of_examples: int,
    seed: int,
    indices: Sequence[int] | None = None,
) -> np.ndarray:
    """候補ごとに、ターゲット語の手がかりの文の課題1で、言い換えの語を選んだか。

    indices は、候補の、全ての候補の中での番号（例と選択肢の並びを決める乱数に使う。一部の候補だけ
    解かせても、同じ候補には同じ問題になるように）。
    """
    templates = []
    choices = []
    numbers = range(len(candidates)) if indices is None else indices
    for index, candidate in zip(numbers, candidates, strict=True):
        random_numbers = random.Random(f"{seed}:{index}")
        options = [paraphrases[candidate.target], candidate.donor]
        random_numbers.shuffle(options)
        sentence = candidate.target_sentence.replace(BLANK, WORD)
        item = TaskItem("sense", sentence, tuple(options), paraphrases[candidate.target])
        templates.append(
            prompt_template(random_numbers.sample(list(examples), number_of_examples), item)
        )
        choices.append(options)
    answers = answer_questions(
        model, templates, [target_ids[candidate.target] for candidate in candidates], choices
    )
    return np.array(
        [
            answer == paraphrases[candidate.target]
            for candidate, answer in zip(candidates, answers, strict=True)
        ]
    )
