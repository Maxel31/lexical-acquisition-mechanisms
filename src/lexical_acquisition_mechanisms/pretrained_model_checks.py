"""公開されている OLMo 2 の3モデルで、評価データを確かめる。

evaluation-data check-with-pretrained-models の処理。
最小対の手がかりが語義を決めているか: 空欄の前までの文の後に置いた語の対数確率を比べる。donor の
手がかりの文（X）とターゲット語の手がかりの文（Y）で、donor は X の方が、ターゲット語は Y の方が
出やすい（全てのモデルで）候補だけを使う。語義の組ごとに、候補の順に
evaluation.minimal_pairs_per_sense_pair 組まで残し（data/evaluation/checked_minimal_pairs.csv）、
組ごとに1組を、人が確かめる minimal_pairs_review.csv に書く（空欄にはターゲット語を入れる。語義を
足したモデルで使う文なので）。確認に通った最小対が足りない語義の組は表示し、1つもない組があれば、
書き出した後に止まる（その語義が、後の全ての分析から黙って消えないように）。
自然な語での課題の正解率（task_accuracy.csv）と、ターゲット語の言い換えの語の確認
（target_paraphrase_checks.csv。元の語義を示す文で、言い換えの語を donor より選ぶ割合が、偶然の
水準 0.5 を超えるか。超えない組は表示する）も書く（task_accuracy）。言い換えの語の確認は、その
モデルで手がかりの条件を満たした候補だけで解かせる（残すのは、全てのモデルで満たした候補なので）。
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
from lexical_acquisition_mechanisms.bootstrap_statistics import SIGNIFICANCE_LEVEL
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_review_rows, write_rows
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.minimal_pairs import BLANK, MinimalPairCandidate
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.settings import ModelSource, Settings
from lexical_acquisition_mechanisms.task_accuracy import (
    TaskAccuracy,
    binomial_p_value,
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


def usable_tasks(model: str, path: Path = TASK_ACCURACY_PATH) -> set[str]:
    """モデルで、自然な語での正解率が偶然の水準を有意に超える課題（task_accuracy.csv）。"""
    return {row.task for row in read_rows(path, TaskAccuracy) if row.model == model and row.usable}


def require_usable_tasks(
    model: str, tasks: Sequence[str], command: str, path: Path = TASK_ACCURACY_PATH
) -> None:
    """モデルで使えない課題（偶然の水準を超えない課題）があれば止まる。

    設計書: 偶然の水準を有意に超えないモデルサイズでは、その課題を使わない。正解率の表がない、または
    表にそのモデルの行がないときは、check-with-pretrained-models から実行するように示して止まる。
    """
    if not path.exists() or model not in {row.model for row in read_rows(path, TaskAccuracy)}:
        raise CommandStopped(
            f"モデル {model} の、自然な語での課題の正解率がありません（{path}）。"
            "settings.yaml の models にそのモデルがあることを確かめて、実行してください。",
            rerun_from=COMMAND,
        )
    unusable = [task for task in tasks if task not in usable_tasks(model, path)]
    if unusable:
        raise CommandStopped(
            f"モデル {model} では、課題 {unusable} の自然な語での正解率が偶然の水準を超えない"
            f"（{path}）ので、この課題を使うコマンドは実行しません。",
            rerun_from=command,
        )


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
    # 割合が偶然の水準（donor との2択なので 0.5）を超えるかの、片側の二項検定
    p_value: float
    above_chance: bool


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
            # 課題が使えるか・言い換えの語が選ばれるかの判定の有意水準
            "significance_level": SIGNIFICANCE_LEVEL,
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
        model_effects = cue_effects(model, candidates, words)
        effects = np.minimum(effects, model_effects)
        accuracy += natural_task_accuracy(model, name, questions)
        checked = np.flatnonzero((model_effects > 0).all(axis=1))
        choices = np.zeros(len(candidates), dtype=bool)
        choices[checked] = paraphrase_choices(
            model,
            [candidates[index] for index in checked],
            target_ids,
            paraphrases,
            examples,
            settings.evaluation.number_of_task_examples,
            settings.words.seed,
            indices=checked.tolist(),
        )
        chose_paraphrase[name] = choices
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
            chosen = int(np.sum(choices[indices]))
            share = chosen / len(indices) if indices else 0.0
            p_value = binomial_p_value(chosen, len(indices), 0.5)
            checks.append(
                ParaphraseCheck(
                    name,
                    target,
                    paraphrases.get(target, ""),
                    len(indices),
                    share,
                    p_value,
                    p_value < SIGNIFICANCE_LEVEL,
                )
            )
    write_rows(paths.paraphrase_checks, ParaphraseCheck, checks)
    weak = [f"{check.target}（{check.model}）" for check in checks if not check.above_chance]
    if weak:
        print(
            f"元の語義を示す文で、言い換えの語が偶然の水準を超えて選ばれない組があります"
            f"（{len(weak)} 組: {weak[:10]}）。{paths.paraphrase_checks} を見て、言い換えの語を"
            "替えてください。"
        )
    short = [
        f"{target}/{donor}"
        for (target, donor), group in kept.items()
        if len(group) < settings.evaluation.minimal_pairs_per_sense_pair
    ]
    if short:
        print(f"確認に通った最小対が足りない語義の組があります（{len(short)} 組: {short[:10]}）。")
    empty = [f"{target}/{donor}" for (target, donor), group in kept.items() if not group]
    if empty:
        raise CommandStopped(
            f"確認に通った最小対が1つもない語義の組があります（{len(empty)} 組: {empty[:10]}）。"
            "settings.yaml の evaluation.minimal_pair_candidate_factor を大きくしてください。",
            rerun_from=minimal_pairs.COMMAND,
        )


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
    keys = ("target", "donor", "donor_sentence", "target_sentence")
    write_review_rows(path, MinimalPairReviewRow, review, keys)


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
