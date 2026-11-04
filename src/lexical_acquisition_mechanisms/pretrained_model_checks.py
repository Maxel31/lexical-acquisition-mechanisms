"""公開されている OLMo 2 の3モデルで、評価データを確かめる。

evaluation-data check-with-pretrained-models の処理。
最小対の手がかりが語義を決めているか: 空欄の前までの文の後に置いた語の対数確率を比べる。donor の
手がかりの文（X）とターゲット語の手がかりの文（Y）で、donor は X の方が、ターゲット語は Y の方が
出やすい（全てのモデルで）候補だけを使う。語義の組ごとに、候補の順に
evaluation.minimal_pairs_per_sense_pair 組まで残し（data/evaluation/checked_minimal_pairs.csv）、
組ごとに1組を、人が確かめる minimal_pairs_review.csv に書く（空欄にはターゲット語を入れる。語義を
足したモデルで使う文なので）。
自然な語での課題の正解率（task_accuracy.csv）と、ターゲット語の言い換えの語の確認
（target_paraphrase_checks.csv）も書く（task_accuracy）。
"""

import argparse
import gc
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet
import torch

from lexical_acquisition_mechanisms import minimal_pairs, task_questions, word_finalization
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.minimal_pairs import BLANK, MinimalPairCandidate
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.settings import ModelSource, Settings
from lexical_acquisition_mechanisms.task_accuracy import (
    TaskAccuracy,
    natural_task_accuracy,
    paraphrase_choices,
)
from lexical_acquisition_mechanisms.task_prompts import TaskItem
from lexical_acquisition_mechanisms.task_questions import Paraphrase, TaskExample
from lexical_acquisition_mechanisms.word_selection import SelectedWord

COMMAND = "evaluation-data check-with-pretrained-models"
OUTPUT_DIRECTORY = Path("data/evaluation")
CHECKED_PAIRS_PATH = OUTPUT_DIRECTORY / "checked_minimal_pairs.csv"
REVIEW_PATH = OUTPUT_DIRECTORY / "minimal_pairs_review.csv"
TASK_ACCURACY_PATH = OUTPUT_DIRECTORY / "task_accuracy.csv"
PARAPHRASE_CHECKS_PATH = OUTPUT_DIRECTORY / "target_paraphrase_checks.csv"
REVIEWED_PAIRS_PER_SENSE_PAIR = 1
BATCH_SIZE = 64


@dataclass
class CheckedMinimalPair(MinimalPairCandidate):
    # 全てのモデルでの最小値: donor の対数確率の差（X − Y）と、ターゲット語の差（Y − X）
    donor_cue_effect: float = 0.0
    target_cue_effect: float = 0.0


@dataclass
class MinimalPairReviewRow:
    target: str
    donor: str
    kind: str
    # それぞれの文で決まるはずの語義（donor の語義と、ターゲット語の元の語義）
    donor_sense: str
    target_sense: str
    # 空欄にターゲット語を入れた2文
    donor_sentence: str
    target_sentence: str
    # 人が記入する（それぞれの文でターゲット語の語義がその行の語義に決まれば 1、そうでなければ 0）
    approved: str = ""


@dataclass
class ParaphraseCheck:
    model: str
    target: str
    paraphrase: str
    # 確認に通った最小対のターゲット語の手がかりの文の数と、そのうち言い換えの語を選んだ割合
    sentences: int
    paraphrase_share: float


@dataclass(frozen=True)
class CheckPaths:
    """この処理が読むファイルと書くファイル。"""

    words: Path = word_finalization.OUTPUT_PATH
    candidates: Path = minimal_pairs.OUTPUT_PATH
    natural_questions: Path = task_questions.NATURAL_QUESTIONS_PATH
    examples: Path = task_questions.EXAMPLES_PATH
    paraphrases: Path = task_questions.PARAPHRASES_PATH
    checked_pairs: Path = CHECKED_PAIRS_PATH
    review: Path = REVIEW_PATH
    task_accuracy: Path = TASK_ACCURACY_PATH
    paraphrase_checks: Path = PARAPHRASE_CHECKS_PATH


def check_with_pretrained_models(settings: Settings, arguments: argparse.Namespace) -> None:
    paths = CheckPaths()
    step = Step(
        COMMAND,
        inputs={
            paths.words: word_finalization.COMMAND,
            paths.candidates: minimal_pairs.COMMAND,
            paths.natural_questions: task_questions.COMMAND,
            paths.examples: task_questions.COMMAND,
            paths.paraphrases: task_questions.COMMAND,
        },
        outputs=[paths.checked_pairs, paths.review, paths.task_accuracy, paths.paraphrase_checks],
        settings={
            "minimal_pairs_per_sense_pair": settings.evaluation.minimal_pairs_per_sense_pair,
            "number_of_task_examples": settings.evaluation.number_of_task_examples,
            "seed": settings.words.seed,
            "models": {name: source.model_dump() for name, source in settings.models.items()},
        },
    )
    run_unless_unchanged(
        step, lambda: check_evaluation_data(settings, _load_model, paths), arguments.recompute
    )


def check_evaluation_data(
    settings: Settings, load_model: Callable[[ModelSource], LanguageModel], paths: CheckPaths
) -> None:
    words = {word.word: word for word in read_rows(paths.words, SelectedWord)}
    candidates = read_rows(paths.candidates, MinimalPairCandidate)
    questions = parquet.read_table(paths.natural_questions).to_pylist()
    examples = [
        (TaskItem(row.task, row.sentence, tuple(row.choices.split("|")), row.answer), row.word)
        for row in read_rows(paths.examples, TaskExample)
        if row.task == "sense"
    ]
    paraphrases = {row.word: row.paraphrase for row in read_rows(paths.paraphrases, Paraphrase)}
    target_ids = {
        word.word: [int(token) for token in word.token_ids.split()]
        for word in words.values()
        if word.role == "target"
    }
    effects = np.full((len(candidates), 2), np.inf)
    accuracy = []
    chose_paraphrase = {}
    for name, source in settings.models.items():
        model = load_model(source)
        effects = np.minimum(effects, cue_effects(model, candidates, words))
        accuracy += natural_task_accuracy(model, name, questions)
        chose_paraphrase[name] = paraphrase_choices(
            model,
            candidates,
            target_ids,
            paraphrases,
            examples,
            settings.evaluation.number_of_task_examples,
            settings.words.seed,
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
    write_rows(paths.task_accuracy, TaskAccuracy, accuracy)
    for row in accuracy:
        if not row.usable:
            print(
                f"{row.model} では、課題 {row.task} の正解率が偶然の水準を超えません"
                "（このモデルでは使いません）。"
            )
    passing = (effects > 0).all(axis=1)
    kept: dict[tuple[str, str], list[int]] = {}
    for index, candidate in enumerate(candidates):
        group = kept.setdefault((candidate.target, candidate.donor), [])
        if passing[index] and len(group) < settings.evaluation.minimal_pairs_per_sense_pair:
            group.append(index)
    checked = []
    for group in kept.values():
        for index in group:
            values = {
                field.name: getattr(candidates[index], field.name)
                for field in fields(MinimalPairCandidate)
            }
            checked.append(
                CheckedMinimalPair(
                    **values,
                    donor_cue_effect=float(effects[index, 0]),
                    target_cue_effect=float(effects[index, 1]),
                )
            )
    write_rows(paths.checked_pairs, CheckedMinimalPair, checked)
    _write_review(paths.review, [pair for pair in checked], words)
    checks = []
    for name, choices in chose_paraphrase.items():
        for target in sorted(target_ids):
            indices = [
                index
                for group in kept.values()
                for index in group
                if candidates[index].target == target
            ]
            share = float(np.mean(choices[indices])) if indices else 0.0
            checks.append(
                ParaphraseCheck(name, target, paraphrases.get(target, ""), len(indices), share)
            )
    write_rows(paths.paraphrase_checks, ParaphraseCheck, checks)
    short = [
        f"{target}/{donor}"
        for (target, donor), group in kept.items()
        if len(group) < settings.evaluation.minimal_pairs_per_sense_pair
    ]
    if short:
        print(f"確認に通った最小対が足りない語義の組があります（{len(short)} 組: {short[:10]}）。")


def _write_review(
    path: Path, checked: Sequence[CheckedMinimalPair], words: dict[str, SelectedWord]
) -> None:
    """語義の組ごとに最初の組を、人が確かめる表に書く。"""
    seen = {}
    for pair in checked:
        seen.setdefault((pair.target, pair.donor), []).append(pair)
    review = [
        MinimalPairReviewRow(
            pair.target,
            pair.donor,
            pair.kind,
            words[pair.donor].definition,
            words[pair.target].definition,
            pair.donor_sentence.replace(BLANK, pair.target),
            pair.target_sentence.replace(BLANK, pair.target),
        )
        for group in seen.values()
        for pair in group[:REVIEWED_PAIRS_PER_SENSE_PAIR]
    ]
    write_rows(path, MinimalPairReviewRow, review)


def cue_effects(
    model: LanguageModel,
    candidates: Sequence[MinimalPairCandidate],
    words: dict[str, SelectedWord],
) -> np.ndarray:
    """候補ごとの、donor の対数確率の差（X − Y）と、ターゲット語の差（Y − X）。"""
    prefixes = []
    word_ids = []
    for candidate in candidates:
        for sentence in (candidate.donor_sentence, candidate.target_sentence):
            for word in (candidate.donor, candidate.target):
                prefixes.append(sentence.split(BLANK)[0].removesuffix(" "))
                word_ids.append([int(token) for token in words[word].token_ids.split()])
    values = _word_log_probabilities(model, prefixes, word_ids).reshape(len(candidates), 2, 2)
    # values[候補, 文（X・Y）, 語（donor・ターゲット語）]
    donor_effect = values[:, 0, 0] - values[:, 1, 0]
    target_effect = values[:, 1, 1] - values[:, 0, 1]
    return np.stack([donor_effect, target_effect], axis=1)


def _word_log_probabilities(
    model: LanguageModel, prefixes: Sequence[str], word_ids: Sequence[list[int]]
) -> np.ndarray:
    """文の書き出しの後に置いた語（トークン列）の対数確率。"""
    inputs = []
    spans = []
    for prefix, ids in zip(prefixes, word_ids, strict=True):
        parts = [prefix, ids] if prefix else [ids]
        token_ids, part_spans = model.tokenizer.encode_with_spans(parts)
        inputs.append(token_ids)
        spans.append(part_spans[-1])
    log_probabilities = model.token_log_probabilities(inputs)
    return np.array(
        [
            values[span.start : span.stop].sum()
            for values, span in zip(log_probabilities, spans, strict=True)
        ]
    )


def _load_model(source: ModelSource) -> LanguageModel:
    return LanguageModel.load(source, batch_size=BATCH_SIZE)
