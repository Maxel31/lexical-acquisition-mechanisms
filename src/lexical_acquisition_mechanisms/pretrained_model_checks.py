"""公開されている OLMo 2 の3モデルで、評価データを確かめる。

evaluation-data check-with-pretrained-models の処理。
最小対の手がかりが語義を決めているか: 空欄の前までの文の後に置いた語の対数確率を比べる。donor の
手がかりの文（X）とターゲット語の手がかりの文（Y）で、donor は X の方が、ターゲット語は Y の方が
出やすい（全てのモデルで）候補だけを使う。語義の組ごとに、候補の順に
evaluation.minimal_pairs_per_sense_pair 組まで残し（data/evaluation/checked_minimal_pairs.csv）、
組ごとに1組を、人が確かめる minimal_pairs_review.csv に書く（空欄にはターゲット語を入れる。語義を
足したモデルで使う文なので）。
自然な語での課題の正解率と、ターゲット語の言い換えの語の確認は、task_accuracy。
"""

import argparse
import gc
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

from lexical_acquisition_mechanisms import minimal_pairs, word_finalization
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.minimal_pairs import BLANK, MinimalPairCandidate
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.settings import ModelSource, Settings
from lexical_acquisition_mechanisms.word_selection import SelectedWord

COMMAND = "evaluation-data check-with-pretrained-models"
OUTPUT_DIRECTORY = Path("data/evaluation")
CHECKED_PAIRS_PATH = OUTPUT_DIRECTORY / "checked_minimal_pairs.csv"
REVIEW_PATH = OUTPUT_DIRECTORY / "minimal_pairs_review.csv"
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


def check_with_pretrained_models(settings: Settings, arguments: argparse.Namespace) -> None:
    step = Step(
        COMMAND,
        inputs={
            word_finalization.OUTPUT_PATH: word_finalization.COMMAND,
            minimal_pairs.OUTPUT_PATH: minimal_pairs.COMMAND,
        },
        outputs=[CHECKED_PAIRS_PATH, REVIEW_PATH],
        settings={
            "minimal_pairs_per_sense_pair": settings.evaluation.minimal_pairs_per_sense_pair,
            "models": {name: source.model_dump() for name, source in settings.models.items()},
        },
    )
    run_unless_unchanged(
        step,
        lambda: check_minimal_pairs(
            settings,
            _load_model,
            word_finalization.OUTPUT_PATH,
            minimal_pairs.OUTPUT_PATH,
            (CHECKED_PAIRS_PATH, REVIEW_PATH),
        ),
        arguments.recompute,
    )


def check_minimal_pairs(
    settings: Settings,
    load_model: Callable[[ModelSource], LanguageModel],
    words_path: Path,
    candidates_path: Path,
    output_paths: tuple[Path, Path],
) -> None:
    checked_path, review_path = output_paths
    words = {word.word: word for word in read_rows(words_path, SelectedWord)}
    candidates = read_rows(candidates_path, MinimalPairCandidate)
    effects = np.full((len(candidates), 2), np.inf)
    for source in settings.models.values():
        model = load_model(source)
        effects = np.minimum(effects, cue_effects(model, candidates, words))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    kept: dict[tuple[str, str], list[CheckedMinimalPair]] = {}
    for candidate, (donor_effect, target_effect) in zip(candidates, effects, strict=True):
        group = kept.setdefault((candidate.target, candidate.donor), [])
        if (
            donor_effect > 0
            and target_effect > 0
            and (len(group) < settings.evaluation.minimal_pairs_per_sense_pair)
        ):
            values = {field.name: getattr(candidate, field.name) for field in fields(candidate)}
            group.append(
                CheckedMinimalPair(
                    **values,
                    donor_cue_effect=float(donor_effect),
                    target_cue_effect=float(target_effect),
                )
            )
    write_rows(
        checked_path, CheckedMinimalPair, [pair for group in kept.values() for pair in group]
    )
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
        for group in kept.values()
        for pair in group[:REVIEWED_PAIRS_PER_SENSE_PAIR]
    ]
    write_rows(review_path, MinimalPairReviewRow, review)
    short = [
        f"{target}/{donor}"
        for (target, donor), group in kept.items()
        if len(group) < settings.evaluation.minimal_pairs_per_sense_pair
    ]
    if short:
        print(f"確認に通った最小対が足りない語義の組があります（{len(short)} 組: {short[:10]}）。")


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
