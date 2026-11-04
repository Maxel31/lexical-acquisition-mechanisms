"""評価に使う最小対と、語義の追加の課題の問題を書き出す（evaluation-data finalize）。

1. 人が確認用の CSV（check-with-pretrained-models が書く minimal_pairs_review.csv）の approved 列に
   記入した値を読み、0 の最小対を除いて、data/evaluation/minimal_pairs.csv に書く。1・0 でない値や、
   確認に通った最小対と合わない行があれば止まる。
2. 語義の追加の問題（sense_addition_questions.parquet）: 足す語義の種類と数 N の条件ごとに、
   ターゲット語の最小対の2文（空欄はターゲット語。{word} で書く）で、
   - 課題1: 選択肢は、ターゲット語の元の語義の代表語（言い換えの語）と、その条件で足す N 語
   - 課題2: 名詞を足す条件はカテゴリ（6択）、形容詞を足す条件は品詞（noun / adjective）
   を作る。正解は、donor の手がかりの文では donor（の属性）、ターゲット語の手がかりの文では元の語義
   （の属性）。対照の run は、対応する条件の問題で測る。
   最小対の2文の問題は、例と選択肢の順番を同じにし、手がかりの語だけが違うようにする（head と MLP の
   出力を、位置ごとにもう一方の文での出力に入れ替えるため）。
"""

import argparse
import random
from collections.abc import Sequence
from pathlib import Path

import pyarrow as arrow
import pyarrow.parquet as parquet

from lexical_acquisition_mechanisms import (
    pretrained_model_checks,
    sense_addition_assignments,
    task_questions,
    word_finalization,
)
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.minimal_pairs import BLANK
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.pretrained_model_checks import (
    CheckedMinimalPair,
    MinimalPairReviewRow,
)
from lexical_acquisition_mechanisms.sense_addition_assignments import SenseAdditionAssignment
from lexical_acquisition_mechanisms.settings import Settings
from lexical_acquisition_mechanisms.task_prompts import (
    PARTS_OF_SPEECH,
    WORD,
    TaskItem,
    prompt_template,
)
from lexical_acquisition_mechanisms.task_questions import Paraphrase, TaskExample
from lexical_acquisition_mechanisms.word_selection import SelectedWord

COMMAND = "evaluation-data finalize"
OUTPUT_DIRECTORY = Path("data/evaluation")
MINIMAL_PAIRS_PATH = OUTPUT_DIRECTORY / "minimal_pairs.csv"
QUESTIONS_PATH = OUTPUT_DIRECTORY / "sense_addition_questions.parquet"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--review-csv",
        type=Path,
        required=True,
        help="approved 列に記入した確認用の CSV（data/evaluation/minimal_pairs_review.csv）",
    )


def finalize_evaluation_data(settings: Settings, arguments: argparse.Namespace) -> None:
    review_path: Path = arguments.review_csv
    inputs = {
        word_finalization.OUTPUT_PATH: word_finalization.COMMAND,
        sense_addition_assignments.ASSIGNMENTS_PATH: sense_addition_assignments.COMMAND,
        pretrained_model_checks.CHECKED_PAIRS_PATH: pretrained_model_checks.COMMAND,
        review_path: pretrained_model_checks.COMMAND,
        task_questions.EXAMPLES_PATH: task_questions.COMMAND,
        task_questions.PARAPHRASES_PATH: task_questions.COMMAND,
    }
    step = Step(
        COMMAND,
        inputs=inputs,
        outputs=[MINIMAL_PAIRS_PATH, QUESTIONS_PATH],
        settings={
            "number_of_task_examples": settings.evaluation.number_of_task_examples,
            "categories": [category.name for category in settings.words.categories],
            "seed": settings.words.seed,
        },
    )
    outputs = (MINIMAL_PAIRS_PATH, QUESTIONS_PATH)
    run_unless_unchanged(
        step,
        lambda: write_final_evaluation_data(settings, list(inputs), review_path, outputs),
        arguments.recompute,
    )


def write_final_evaluation_data(
    settings: Settings,
    input_paths: Sequence[Path],
    review_path: Path,
    output_paths: tuple[Path, Path],
) -> None:
    words_path, assignments_path, checked_path, _, examples_path, paraphrases_path = input_paths
    words = {word.word: word for word in read_rows(words_path, SelectedWord)}
    checked = read_rows(checked_path, CheckedMinimalPair)
    review = read_rows(review_path, MinimalPairReviewRow)
    rerun = f"{COMMAND} --review-csv {review_path}"
    filled = {
        (pair.target, pair.donor, pair.donor_sentence.replace(BLANK, pair.target)): pair
        for pair in checked
    }
    unknown = [row for row in review if (row.target, row.donor, row.donor_sentence) not in filled]
    if unknown:
        raise CommandStopped(
            f"{review_path} に、確認に通った最小対と合わない行があります（{len(unknown)} 行）。"
            "書き出し直して、確認し直してください。",
            rerun_from=pretrained_model_checks.COMMAND,
        )
    unfilled = [row for row in review if row.approved.strip() not in {"0", "1"}]
    if unfilled:
        raise CommandStopped(
            f"{review_path} の approved 列が 1・0 でない行が {len(unfilled)} 行あります。"
            "記入してください。",
            rerun_from=rerun,
        )
    rejected = {
        id(filled[(row.target, row.donor, row.donor_sentence)])
        for row in review
        if row.approved.strip() == "0"
    }
    pairs = [pair for pair in checked if id(pair) not in rejected]
    write_rows(output_paths[0], CheckedMinimalPair, pairs)
    examples = {}
    for row in read_rows(examples_path, TaskExample):
        item = TaskItem(row.task, row.sentence, tuple(row.choices.split("|")), row.answer)
        examples.setdefault(row.task, []).append((item, row.word))
    paraphrases = {row.word: row.paraphrase for row in read_rows(paraphrases_path, Paraphrase)}
    assignments = read_rows(assignments_path, SenseAdditionAssignment)
    questions = sense_addition_questions(settings, words, pairs, assignments, paraphrases, examples)
    parquet.write_table(arrow.Table.from_pylist(questions), output_paths[1])


def sense_addition_questions(
    settings: Settings,
    words: dict[str, SelectedWord],
    pairs: Sequence[CheckedMinimalPair],
    assignments: Sequence[SenseAdditionAssignment],
    paraphrases: dict[str, str],
    examples: dict[str, list[tuple[TaskItem, str]]],
) -> list[dict[str, object]]:
    """条件（足す語義の種類と数）ごとの、最小対の文の課題1と課題2の問題。"""
    categories = [category.name for category in settings.words.categories]
    count = settings.evaluation.number_of_task_examples
    seed = settings.words.seed
    donors: dict[tuple[str, str], list[str]] = {}
    for row in sorted(assignments, key=lambda row: row.order):
        donors.setdefault((row.run, row.target), []).append(row.donor)
    questions = []
    for (run, target), run_donors in donors.items():
        kind, _, number = run.rpartition("-")
        senses = [paraphrases[target], *run_donors]
        run_pairs = [pair for pair in pairs if pair.target == target and pair.donor in run_donors]
        for pair_number, pair in enumerate(run_pairs):
            for cue, sentence in (("donor", pair.donor_sentence), ("target", pair.target_sentence)):
                # 2文で同じ乱数を使い、例と選択肢の順番をそろえる
                random_numbers = random.Random(f"{seed}:{run}:{target}:{pair_number}")
                template = sentence.replace(BLANK, WORD)
                sense_choices = list(senses)
                random_numbers.shuffle(sense_choices)
                answer = pair.donor if cue == "donor" else paraphrases[target]
                items = [TaskItem("sense", template, tuple(sense_choices), answer)]
                if kind == "adjective":
                    parts = list(PARTS_OF_SPEECH)
                    random_numbers.shuffle(parts)
                    part = "adjective" if cue == "donor" else "noun"
                    items.append(TaskItem("part-of-speech", template, tuple(parts), part))
                else:
                    shuffled = list(categories)
                    random_numbers.shuffle(shuffled)
                    meant = pair.donor if cue == "donor" else target
                    category = words[meant].category
                    items.append(TaskItem("category", template, tuple(shuffled), category))
                for item in items:
                    chosen = random_numbers.sample(examples[item.task], count)
                    questions.append(
                        {
                            "kind": kind,
                            "number_of_senses": int(number),
                            "target": target,
                            "donor": pair.donor,
                            "donor_sentence": pair.donor_sentence,
                            "cue": cue,
                            "task": item.task,
                            "prompt_template": prompt_template(chosen, item),
                            "choices": list(item.choices),
                            "answer": item.answer,
                        }
                    )
    return questions
