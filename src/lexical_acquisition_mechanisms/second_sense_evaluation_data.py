"""2つ目の語義の追加の評価データ（最小対と課題1の問題）を作る
（evaluation-data generate-second-sense-minimal-pairs・build-second-sense-questions）。

2つ目の語義の追加の割り当て表の、1つ目の語義の名詞 X と2つ目の語義の名詞 Y の組ごとに、最小対
（手がかりの1語だけが違い、空欄に X と Y のどちらが合うかが変わる2文）を作る。X をターゲット
語、Y を donor として、語義の追加と同じ指示（generate-minimal-pairs）と判定（judge-minimal-pair）
を使う。
1. generate-second-sense-minimal-pairs（LLM）: 候補を作り、LLM の判定に合格した候補を残す。
2. build-second-sense-questions（GPU）: settings の全てのモデルで、手がかりが X と Y の出やすさを
   正しい向きに動かす候補を、組ごとに evaluation.minimal_pairs_per_sense_pair 個まで残し
   （check-with-pretrained-models と同じ基準）、その2文で課題1の問題を作る。空欄は未知語（問題文に
   {word} と書き、解析のときに未知語のトークン列を入れる）、選択肢は X と Y、正解は X の手がかりの
   文では X、Y の手がかりの文では Y。2文の問題は、例と選択肢の順番を同じにする（2文の違いが
   手がかりの1語だけになるように）。
結果は data/evaluation/second-sense-addition/ に書く。
"""

import argparse
import gc
import random
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet
import torch

from lexical_acquisition_mechanisms import (
    minimal_pair_judgment,
    minimal_pairs,
    second_sense_assignments,
    task_questions,
    word_finalization,
)
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.llm_client import ANSWERS_PATH, LlmClient, PromptTemplate
from lexical_acquisition_mechanisms.minimal_pairs import BLANK, MinimalPairCandidate, SensePair
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.pretrained_model_checks import (
    CheckedMinimalPair,
    cue_effects,
    stop_if_sense_pairs_lack_minimal_pairs,
)
from lexical_acquisition_mechanisms.second_sense_assignments import SecondSenseAssignment
from lexical_acquisition_mechanisms.settings import ModelSource, Settings
from lexical_acquisition_mechanisms.task_prompts import WORD, TaskItem, prompt_template
from lexical_acquisition_mechanisms.task_questions import TaskExample
from lexical_acquisition_mechanisms.word_selection import SelectedWord

GENERATE_COMMAND = "evaluation-data generate-second-sense-minimal-pairs"
QUESTIONS_COMMAND = "evaluation-data build-second-sense-questions"
OUTPUT_DIRECTORY = Path("data/evaluation") / second_sense_assignments.EXPERIMENT
BATCH_SIZE = 64


@dataclass
class SecondSensePaths:
    """この処理が読むファイルと書くファイル。"""

    words: Path = word_finalization.OUTPUT_PATH
    assignments: Path = second_sense_assignments.ASSIGNMENTS_PATH
    examples: Path = task_questions.EXAMPLES_PATH
    candidates: Path = OUTPUT_DIRECTORY / "minimal_pair_candidates.csv"
    judged: Path = OUTPUT_DIRECTORY / "judged_minimal_pair_candidates.csv"
    judgments: Path = OUTPUT_DIRECTORY / "minimal_pair_judgments.csv"
    checked: Path = OUTPUT_DIRECTORY / "checked_minimal_pairs.csv"
    questions: Path = OUTPUT_DIRECTORY / "questions.parquet"


def second_sense_pairs(words_path: Path, assignments_path: Path) -> list[SensePair]:
    """割り当て表の (X, Y, 種類) の組（重なりを除き、X・Y・種類の順に並べる）。"""
    words = {word.word: word for word in read_rows(words_path, SelectedWord)}
    rows = read_rows(assignments_path, SecondSenseAssignment)
    chosen = sorted({(row.first_noun, row.second_noun, row.kind) for row in rows})
    return [
        SensePair(
            first,
            words[first].category,
            words[first].token_count,
            second,
            words[second].role,
            kind,
        )
        for first, second, kind in chosen
    ]


def generate_second_sense_minimal_pairs(settings: Settings, arguments: argparse.Namespace) -> None:
    paths = SecondSensePaths()
    generation = PromptTemplate.load(minimal_pairs.PROMPT)
    judgment = PromptTemplate.load(minimal_pair_judgment.PROMPT)
    step = Step(
        GENERATE_COMMAND,
        inputs={
            paths.words: word_finalization.COMMAND,
            paths.assignments: second_sense_assignments.COMMAND,
        },
        outputs=[paths.candidates, paths.judged, paths.judgments],
        settings={
            "generation": minimal_pairs.record_settings(settings, generation),
            "judgment": minimal_pair_judgment.record_settings(settings, judgment),
        },
    )

    client = LlmClient(settings.llm, ANSWERS_PATH, rerun_from=GENERATE_COMMAND)
    run_unless_unchanged(
        step,
        lambda: write_second_sense_minimal_pairs(settings, (generation, judgment), client, paths),
        arguments.recompute,
    )


def write_second_sense_minimal_pairs(
    settings: Settings,
    prompts: tuple[PromptTemplate, PromptTemplate],
    client: LlmClient,
    paths: SecondSensePaths,
) -> None:
    """(X, Y) の組ごとに最小対の候補を作り（prompts の1つ目）、LLM が判定する（2つ目）。"""
    generation, judgment = prompts
    paths.candidates.parent.mkdir(parents=True, exist_ok=True)
    minimal_pairs.write_minimal_pair_candidates(
        settings,
        generation,
        client,
        paths.words,
        second_sense_pairs(paths.words, paths.assignments),
        paths.candidates,
        command=GENERATE_COMMAND,
    )
    minimal_pair_judgment.write_judged_minimal_pairs(
        judgment, client, paths.words, paths.candidates, (paths.judged, paths.judgments)
    )


def build_second_sense_questions(settings: Settings, arguments: argparse.Namespace) -> None:
    paths = SecondSensePaths()
    step = Step(
        QUESTIONS_COMMAND,
        inputs={
            paths.words: word_finalization.COMMAND,
            paths.assignments: second_sense_assignments.COMMAND,
            paths.judged: GENERATE_COMMAND,
            paths.examples: task_questions.COMMAND,
        },
        outputs=[paths.checked, paths.questions],
        settings={
            "minimal_pairs_per_sense_pair": settings.evaluation.minimal_pairs_per_sense_pair,
            "allow_sense_pairs_without_minimal_pairs": (
                settings.evaluation.allow_sense_pairs_without_minimal_pairs
            ),
            "number_of_task_examples": settings.evaluation.number_of_task_examples,
            "seed": settings.words.seed,
            "models": {name: source.model_dump() for name, source in settings.models.items()},
        },
    )
    run_unless_unchanged(
        step,
        lambda: write_second_sense_questions(settings, _load_model, paths),
        arguments.recompute,
    )


def write_second_sense_questions(
    settings: Settings, load_model: Callable[[ModelSource], LanguageModel], paths: SecondSensePaths
) -> None:
    words = {word.word: word for word in read_rows(paths.words, SelectedWord)}
    candidates = read_rows(paths.judged, MinimalPairCandidate)
    # 全てのモデルでの最小値（どのモデルでも、手がかりが正しい向きに動かす候補だけを残す）
    effects = np.full((len(candidates), 2), np.inf)
    for source in settings.models.values():
        model = load_model(source)
        effects = np.minimum(effects, cue_effects(model, candidates, words))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    passing = (effects > 0).all(axis=1)
    kept: dict[tuple[str, str], list[int]] = {
        (pair.target, pair.donor): [] for pair in second_sense_pairs(paths.words, paths.assignments)
    }
    for index, candidate in enumerate(candidates):
        group = kept[(candidate.target, candidate.donor)]
        if passing[index] and len(group) < settings.evaluation.minimal_pairs_per_sense_pair:
            group.append(index)
    checked = [
        CheckedMinimalPair(
            **{
                field.name: getattr(candidates[index], field.name)
                for field in fields(MinimalPairCandidate)
            },
            donor_cue_effect=float(effects[index, 0]),
            target_cue_effect=float(effects[index, 1]),
        )
        for group in kept.values()
        for index in group
    ]
    write_rows(paths.checked, CheckedMinimalPair, checked)
    counts = [len(group) for group in kept.values()]
    print(
        f"確認に通った最小対: {len(checked)} / {len(candidates)} 組（語義の組 {len(kept)} 組、"
        f"組ごとの中央値 {np.median(counts):.0f}）。"
    )
    empty = [f"{first}/{second}" for (first, second), group in kept.items() if not group]
    stop_if_sense_pairs_lack_minimal_pairs(
        empty, settings.evaluation.allow_sense_pairs_without_minimal_pairs
    )
    examples = [
        (TaskItem(row.task, row.sentence, tuple(row.choices.split("|")), row.answer), row.word)
        for row in read_rows(paths.examples, TaskExample)
        if row.task == "sense"
    ]
    rows = []
    for number, pair in enumerate(checked):
        random_numbers = random.Random(f"{settings.words.seed}:{pair.target}:{pair.donor}:{number}")
        choices = [pair.target, pair.donor]
        random_numbers.shuffle(choices)
        chosen = random_numbers.sample(examples, settings.evaluation.number_of_task_examples)
        for cue, sentence, answer in (
            ("first", pair.target_sentence, pair.target),
            ("second", pair.donor_sentence, pair.donor),
        ):
            item = TaskItem("sense", sentence.replace(BLANK, WORD), tuple(choices), answer)
            rows.append(
                {
                    "first_noun": pair.target,
                    "second_noun": pair.donor,
                    "kind": pair.kind,
                    "pair": number,
                    "cue": cue,
                    "prompt_template": prompt_template(chosen, item),
                    "choices": choices,
                    "answer": answer,
                }
            )
    parquet.write_table(arrow.Table.from_pylist(rows), paths.questions)


def _load_model(source: ModelSource) -> LanguageModel:
    return LanguageModel.load(source, batch_size=BATCH_SIZE)
