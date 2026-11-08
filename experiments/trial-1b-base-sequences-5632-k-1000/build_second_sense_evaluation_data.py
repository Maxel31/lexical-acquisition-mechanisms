"""2つ目の語義の追加の評価データを、試行のために作る（試行の作業。本来のコマンドは、試行の
結果を見てから作る）。

割り当て表（data/assignments/second-sense-addition/assignments.csv）の、1つ目の語義の名詞 X と
2つ目の語義の名詞 Y の組ごとに、次の3つの段階を順に行う。結果は
outputs/partial-results/second-sense-evaluation/ に書く。
- generate（LLM のサーバが要る）: 最小対の候補を、evaluation-data generate-minimal-pairs と
  同じ指示と処理で作り（X をターゲット語、Y を donor として渡す）、judge-minimal-pairs と同じ
  判定で絞る。
- check（GPU 1枚）: 学習前のモデルで、手がかりが X と Y の出やすさを正しい向きに動かす候補を、
  組ごとに evaluation.minimal_pairs_per_sense_pair 個まで残す（check-with-pretrained-models の
  cue_effects と同じ基準）。
- questions: 残った最小対の2文で、課題1の問題を作る。空欄は未知語（{word} で書く）、選択肢は
  X と Y、正解は、X の手がかりの文では X、Y の手がかりの文では Y。2文の問題は、例と選択肢の
  順番を同じにする。
使い方（リポジトリの最上位のディレクトリで）:
  uv run python \
    experiments/trial-1b-base-sequences-5632-k-1000/build_second_sense_evaluation_data.py generate
"""

import random
import sys
from pathlib import Path

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet

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
from lexical_acquisition_mechanisms.minimal_pairs import BLANK, MinimalPairCandidate
from lexical_acquisition_mechanisms.pretrained_model_checks import cue_effects
from lexical_acquisition_mechanisms.second_sense_assignments import SecondSenseAssignment
from lexical_acquisition_mechanisms.sense_addition_assignments import SenseAdditionAssignment
from lexical_acquisition_mechanisms.settings import load_settings
from lexical_acquisition_mechanisms.task_prompts import WORD, TaskItem, prompt_template
from lexical_acquisition_mechanisms.task_questions import TaskExample
from lexical_acquisition_mechanisms.word_selection import SelectedWord

DIRECTORY = Path("outputs/partial-results/second-sense-evaluation")
PAIRS_PATH = DIRECTORY / "sense_pairs.csv"
CANDIDATES_PATH = DIRECTORY / "minimal_pair_candidates.csv"
JUDGED_PATH = DIRECTORY / "judged_minimal_pair_candidates.csv"
JUDGMENTS_PATH = DIRECTORY / "minimal_pair_judgments.csv"
CHECKED_PATH = DIRECTORY / "checked_minimal_pairs.csv"
QUESTIONS_PATH = DIRECTORY / "questions.parquet"
MODEL = "OLMo-2-0425-1B"


def sense_pairs(words: dict[str, SelectedWord]) -> list[SenseAdditionAssignment]:
    """割り当て表の (X, Y, 種類) の組を、generate-minimal-pairs が読む形にしたもの。"""
    rows = read_rows(second_sense_assignments.ASSIGNMENTS_PATH, SecondSenseAssignment)
    pairs = sorted({(row.first_noun, row.second_noun, row.kind) for row in rows})
    return [
        SenseAdditionAssignment(
            # generate-minimal-pairs は、足す語義の数が最大（4）の run の行だけを読む
            f"{kind}-4",
            first,
            words[first].category,
            words[first].token_count,
            second,
            "noun",
            words[second].category,
            1,
        )
        for first, second, kind in pairs
    ]


def generate() -> None:
    settings = load_settings(Path("settings.yaml"))
    words = {word.word: word for word in read_rows(word_finalization.OUTPUT_PATH, SelectedWord)}
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    write_rows(PAIRS_PATH, SenseAdditionAssignment, sense_pairs(words))
    client = LlmClient(settings.llm, ANSWERS_PATH, rerun_from=minimal_pairs.COMMAND)
    minimal_pairs.build_minimal_pair_candidates(
        settings,
        PromptTemplate.load(minimal_pairs.PROMPT),
        client,
        word_finalization.OUTPUT_PATH,
        PAIRS_PATH,
        CANDIDATES_PATH,
    )
    minimal_pair_judgment.write_judged_minimal_pairs(
        PromptTemplate.load(minimal_pair_judgment.PROMPT),
        LlmClient(settings.llm, ANSWERS_PATH, rerun_from=minimal_pair_judgment.COMMAND),
        word_finalization.OUTPUT_PATH,
        CANDIDATES_PATH,
        (JUDGED_PATH, JUDGMENTS_PATH),
    )


def check() -> None:
    settings = load_settings(Path("settings.yaml"))
    words = {word.word: word for word in read_rows(word_finalization.OUTPUT_PATH, SelectedWord)}
    candidates = read_rows(JUDGED_PATH, MinimalPairCandidate)
    model = LanguageModel.load(settings.models[MODEL], batch_size=16)
    effects = cue_effects(model, candidates, words)
    passing = (effects > 0).all(axis=1)
    kept: dict[tuple[str, str], list[MinimalPairCandidate]] = {}
    for candidate, passed in zip(candidates, passing, strict=True):
        group = kept.setdefault((candidate.target, candidate.donor), [])
        if passed and len(group) < settings.evaluation.minimal_pairs_per_sense_pair:
            group.append(candidate)
    checked = [candidate for group in kept.values() for candidate in group]
    write_rows(CHECKED_PATH, MinimalPairCandidate, checked)
    counts = np.array([len(group) for group in kept.values()])
    print(
        f"確認に通った最小対: {len(checked)} / {len(candidates)} 組。"
        f"語義の組 {len(kept)} 組のうち、0 組 {int((counts == 0).sum())}、"
        f"20 組に届かない {int((counts < 20).sum())}。"
    )


def questions() -> None:
    settings = load_settings(Path("settings.yaml"))
    examples = [
        (TaskItem(row.task, row.sentence, tuple(row.choices.split("|")), row.answer), row.word)
        for row in read_rows(task_questions.EXAMPLES_PATH, TaskExample)
        if row.task == "sense"
    ]
    rows = []
    for index, pair in enumerate(read_rows(CHECKED_PATH, MinimalPairCandidate)):
        random_numbers = random.Random(f"{settings.words.seed}:{pair.target}:{pair.donor}:{index}")
        choices = [pair.target, pair.donor]
        random_numbers.shuffle(choices)
        chosen = random_numbers.sample(examples, settings.evaluation.number_of_task_examples)
        for cue, sentence, answer in [
            ("first", pair.target_sentence, pair.target),
            ("second", pair.donor_sentence, pair.donor),
        ]:
            item = TaskItem("sense", sentence.replace(BLANK, WORD), tuple(choices), answer)
            rows.append(
                {
                    "first_noun": pair.target,
                    "second_noun": pair.donor,
                    "kind": pair.kind,
                    "pair": index,
                    "cue": cue,
                    "prompt_template": prompt_template(chosen, item),
                    "choices": choices,
                    "answer": answer,
                }
            )
    parquet.write_table(arrow.Table.from_pylist(rows), QUESTIONS_PATH)
    print(f"{QUESTIONS_PATH} に {len(rows)} 問を書きました。")


if __name__ == "__main__":
    {"generate": generate, "check": check, "questions": questions}[sys.argv[1]]()
