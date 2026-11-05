"""run ごとの学習の設定ファイル。

training write-run-settings --experiment <実験> --model <モデル> が書く。

outputs/runs/<実験>/<モデル>/<run>.yaml に、その run の学習に要るものを全て書く。training run は、
このファイルだけを読んで学習する（後で settings.yaml を変えても、書いた run の設定は変わらない）。
"""

import argparse
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationError

from lexical_acquisition_mechanisms import (
    base_data,
    sense_addition_assignments,
    sense_addition_training_data,
    unknown_word_assignments,
    unknown_word_training_data,
    word_finalization,
)
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.csv_tables import read_rows
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.settings import (
    ModelSource,
    Settings,
    SettingsError,
    SettingsSection,
    describe_problems,
)
from lexical_acquisition_mechanisms.word_selection import SelectedWord

RUNS_DIRECTORY = Path("outputs/runs")
CHECKPOINTS_DIRECTORY = Path("outputs/checkpoints")
EXPERIMENT_RUNS = {
    unknown_word_assignments.EXPERIMENT: [
        *unknown_word_assignments.run_names(),
        unknown_word_training_data.CONTROL_RUN,
    ],
    sense_addition_assignments.EXPERIMENT: sense_addition_assignments.run_names(),
}
# 実験ごとの、学習データ（training-data build が書く）のディレクトリ
TRAINING_DATA_DIRECTORIES = {
    unknown_word_assignments.EXPERIMENT: unknown_word_training_data.OUTPUT_DIRECTORY,
    sense_addition_assignments.EXPERIMENT: sense_addition_training_data.OUTPUT_DIRECTORY,
}


class RunSettings(SettingsSection):
    """1つの run の学習の設定。"""

    experiment: str
    run: str
    # --model で指定した名前と、その取得元
    model: str
    model_source: ModelSource
    # 学習データ（training-data build が書いたもの）と、差し込んだ用例の始まりの位置、検証データ
    training_tokens: Path
    passage_starts: Path
    # 差し込んだ用例1件の長さ（後に付けた区切りを含む）
    passage_length: int = Field(gt=0)
    validation_sequences: Path
    # 重みと学習の記録を書くディレクトリ
    output_directory: Path
    learning_rate: float = Field(gt=0)
    batch_size_in_tokens: int = Field(gt=0)
    sequence_length: int = Field(gt=0)
    warmup_fraction: float = Field(ge=0, lt=1)
    sequences_per_device: int = Field(gt=0)
    optimizer_betas: tuple[float, float]
    optimizer_epsilon: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    gradient_clipping: float = Field(gt=0)
    z_loss_weight: float = Field(ge=0)
    checkpoint_fractions: list[float]
    checkpoint_number_type: Literal["bfloat16", "float32"]
    # 埋め込みを初期化する新しいトークン番号（1トークンの未知語）と、乱数の倍率
    new_token_ids: list[int]
    new_token_noise_scale: float = Field(ge=0)
    seed: int
    weights_and_biases_project: str


def run_settings_path(
    experiment: str, model: str, run: str, directory: Path = RUNS_DIRECTORY
) -> Path:
    return directory / experiment / model / f"{run}.yaml"


def load_run_settings(path: Path) -> RunSettings:
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SettingsError(f"run の設定ファイル {path} がありません。") from error
    try:
        return RunSettings.model_validate(content)
    except ValidationError as error:
        raise SettingsError(describe_problems(path, error)) from error


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--experiment", choices=list(EXPERIMENT_RUNS), required=True, help="学習する実験"
    )
    parser.add_argument(
        "--model", required=True, help="学習するモデル（settings.yaml の models の名前）"
    )


def write_run_settings(settings: Settings, arguments: argparse.Namespace) -> None:
    experiment, model = arguments.experiment, arguments.model
    if model not in settings.models:
        raise CommandStopped(
            f"モデル {model} は settings.yaml の models にありません（{list(settings.models)}）。",
            rerun_from=f"training write-run-settings --experiment {experiment} --model {model}",
        )
    paths = [run_settings_path(experiment, model, run) for run in EXPERIMENT_RUNS[experiment]]
    step = Step(
        f"training write-run-settings --experiment {experiment} --model {model}",
        inputs={word_finalization.OUTPUT_PATH: word_finalization.COMMAND},
        outputs=paths,
        settings={
            "training": settings.training.model_dump(mode="json"),
            "tokens_per_passage": settings.passages.tokens_per_passage,
            "model": settings.models[model].model_dump(),
            "seed": settings.words.seed,
        },
    )
    run_unless_unchanged(
        step,
        lambda: build_run_settings(settings, experiment, model, word_finalization.OUTPUT_PATH),
        arguments.recompute,
    )


def build_run_settings(
    settings: Settings,
    experiment: str,
    model: str,
    words_path: Path,
    directory: Path = RUNS_DIRECTORY,
) -> list[Path]:
    """実験の全ての run の設定ファイルを書き、そのパスを返す。"""
    training = settings.training
    new_token_ids = []
    if experiment == unknown_word_assignments.EXPERIMENT:
        new_token_ids = [
            int(word.token_ids)
            for word in read_rows(words_path, SelectedWord)
            if word.role == "unknown" and word.token_count == 1
        ]
    paths = []
    for run in EXPERIMENT_RUNS[experiment]:
        data_directory = TRAINING_DATA_DIRECTORIES[experiment] / run
        run_settings = RunSettings(
            experiment=experiment,
            run=run,
            model=model,
            model_source=settings.models[model],
            training_tokens=data_directory / unknown_word_training_data.TOKENS_FILE_NAME,
            passage_starts=data_directory / unknown_word_training_data.PASSAGE_STARTS_FILE_NAME,
            passage_length=settings.passages.tokens_per_passage + 1,
            validation_sequences=base_data.output_paths()["validation"],
            output_directory=CHECKPOINTS_DIRECTORY / experiment / model / run,
            learning_rate=training.learning_rate_by_model[model],
            batch_size_in_tokens=training.batch_size_in_tokens,
            sequence_length=training.sequence_length,
            warmup_fraction=training.warmup_fraction,
            sequences_per_device=training.sequences_per_device_by_model[model],
            optimizer_betas=training.optimizer_betas,
            optimizer_epsilon=training.optimizer_epsilon,
            weight_decay=training.weight_decay,
            gradient_clipping=training.gradient_clipping,
            z_loss_weight=training.z_loss_weight,
            checkpoint_fractions=training.checkpoint_fractions,
            checkpoint_number_type=training.checkpoint_number_type,
            new_token_ids=new_token_ids,
            new_token_noise_scale=training.new_token_embedding_initialization.noise_scale,
            seed=settings.words.seed,
            weights_and_biases_project=training.weights_and_biases_project,
        )
        path = run_settings_path(experiment, model, run, directory)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = run_settings.model_dump(mode="json")
        path.write_text(
            yaml.safe_dump(content, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        paths.append(path)
    return paths
