"""課題1の選択肢の確率を、問題ごとに保存する（試行の診断。GPU 1枚）。

measure-sense-accuracy は最後の重みで答え（最も確率の大きい選択肢）だけを残すので、この試行だけ、
選択肢ごとの対数確率を、次の2つの場合について保存する。結果は
outputs/partial-results/sense-choice-probabilities/ に書く（本来の解析の結果とは別の場所）。

- checkpoints <モデル> <チェックポイント>: 語義の追加の11の run の、学習の途中の重み
  （checkpoint-25・checkpoint-50・checkpoint-75・final）で、各 run の条件の課題1を解かせる
  （対照の run は、対応する条件の全ての問題）。
- context <名前> <リポジトリ> <版>: 学習前のモデルで、全ての課題1を、空欄にターゲット語・非語
  （blicket）・donor をそれぞれ入れて解かせる。非語では、答えを決める手がかりは文だけなので、
  手がかりの文によって答えが動くかで、そのモデルが文を読んで課題1を解けるかがわかる。

使い方（リポジトリの最上位のディレクトリで）:
  CUDA_VISIBLE_DEVICES=0 uv run python \
    experiments/trial-1b-base-sequences-5632-k-1000/save_sense_choice_probabilities.py \
    checkpoints OLMo-2-0425-1B checkpoint-25
  CUDA_VISIBLE_DEVICES=4 uv run python \
    experiments/trial-1b-base-sequences-5632-k-1000/save_sense_choice_probabilities.py \
    context OLMo-2-1124-7B allenai/OLMo-2-1124-7B c0371f4281bf2376207646c6b62ddc6c442c7577
"""

import gc
import sys
from collections.abc import Sequence
from pathlib import Path

import pyarrow as arrow
import pyarrow.parquet as parquet
import torch

from lexical_acquisition_mechanisms import (
    evaluation_finalization,
    sense_accuracy,
    sense_addition_assignments,
    word_finalization,
)
from lexical_acquisition_mechanisms.csv_tables import read_rows
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.settings import ModelSource
from lexical_acquisition_mechanisms.task_prompts import choice_token_ids, encode_prompt
from lexical_acquisition_mechanisms.word_selection import SelectedWord

OUTPUT_DIRECTORY = Path("outputs/partial-results/sense-choice-probabilities")
CHECKPOINTS_DIRECTORY = Path("outputs/checkpoints") / sense_addition_assignments.EXPERIMENT
NONWORD = "blicket"
BATCH_SIZE = 16


def sense_questions() -> list[tuple[int, dict]]:
    questions = parquet.read_table(evaluation_finalization.QUESTIONS_PATH).to_pylist()
    return [
        (index, question) for index, question in enumerate(questions) if question["task"] == "sense"
    ]


def word_ids(words: dict[str, SelectedWord], word: str) -> list[int]:
    return [int(token) for token in words[word].token_ids.split()]


def choice_log_probabilities(
    model: LanguageModel, questions: Sequence[dict], ids: Sequence[Sequence[int]]
) -> list[list[float]]:
    """問題ごとに、"Answer:" の次に来る各選択肢（問題の choices の順）の対数確率。"""
    inputs = [
        encode_prompt(question["prompt_template"], word, model.tokenizer)
        for question, word in zip(questions, ids, strict=True)
    ]
    candidates = sorted({choice for question in questions for choice in question["choices"]})
    column = {choice: index for index, choice in enumerate(candidates)}
    values = model.next_token_log_probabilities(
        inputs,
        [len(tokens) - 1 for tokens in inputs],
        choice_token_ids(candidates, model.tokenizer),
    )
    return [
        [float(row[column[choice]]) for choice in question["choices"]]
        for question, row in zip(questions, values, strict=True)
    ]


def write(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(arrow.Table.from_pylist(rows), path)
    print(f"{path} に {len(rows)} 行を書きました。", flush=True)


def save_checkpoints(model_name: str, checkpoint: str) -> None:
    words = {word.word: word for word in read_rows(word_finalization.OUTPUT_PATH, SelectedWord)}
    questions = sense_questions()
    rows = []
    for run in sense_addition_assignments.run_names():
        chosen = [
            (index, question)
            for index, question in questions
            if sense_accuracy.evaluated_in(run, question)
        ]
        directory = CHECKPOINTS_DIRECTORY / model_name / run / checkpoint
        model = LanguageModel.load(ModelSource(repository=str(directory)), batch_size=BATCH_SIZE)
        values = choice_log_probabilities(
            model,
            [question for _, question in chosen],
            [word_ids(words, question["target"]) for _, question in chosen],
        )
        rows += [
            {"run": run, "checkpoint": checkpoint, "question": index, "log_probabilities": value}
            for (index, _), value in zip(chosen, values, strict=True)
        ]
        print(f"{run} {checkpoint}: {len(chosen)} 問", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    write(rows, OUTPUT_DIRECTORY / "checkpoints" / model_name / f"{checkpoint}.parquet")


def save_context(name: str, repository: str, revision: str) -> None:
    words = {word.word: word for word in read_rows(word_finalization.OUTPUT_PATH, SelectedWord)}
    model = LanguageModel.load(
        ModelSource(repository=repository, revision=revision), batch_size=BATCH_SIZE
    )
    nonword = model.tokenizer.word_token_ids(NONWORD)
    questions = sense_questions()
    rows = []
    for variant in ("target", "nonword", "donor"):
        ids = [
            nonword if variant == "nonword" else word_ids(words, question[variant])
            for _, question in questions
        ]
        values = choice_log_probabilities(model, [question for _, question in questions], ids)
        rows += [
            {"model": name, "word": variant, "question": index, "log_probabilities": value}
            for (index, _), value in zip(questions, values, strict=True)
        ]
        print(f"{name} {variant}: {len(questions)} 問", flush=True)
    write(rows, OUTPUT_DIRECTORY / "context" / f"{name}.parquet")


if __name__ == "__main__":
    mode, *values = sys.argv[1:]
    {"checkpoints": save_checkpoints, "context": save_context}[mode](*values)
