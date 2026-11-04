"""実在語らしさのプローブを学習する（unknown-word-acquisition train-real-word-probes --model）。

学習前のモデル（settings.yaml の版）で、プローブの語
（evaluation-data build-probe-and-normalization-words が書いたもの）を中立文脈の各条件の直後に置き、
語の最後のトークンの位置の各層の残差ストリームを取り出す。トークン数
（1・2・3）ごと・層ごとに、実在語（正例）とそれ以外（負例: 非語・学習していない新しいトークン）を
分けるロジスティック回帰を、学習用の語の全ての条件で学習し、確認用の語での正解率を出す。
学習していない新しいトークンは、学習のときと同じ方法（既存の埋め込みの平均 + 乱数）で初期化する。
乱数の種は学習のときからずらす（同じ種だと、乱数の先頭が1トークンの未知語の初期値と同じになり、
露出0回の未知語の埋め込みが、プローブの負例そのものになるため）。
結果は outputs/probes/<モデル>/ の real_word_probes.npz と real_word_probe_accuracy.csv に書く。
"""

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lexical_acquisition_mechanisms import neutral_contexts, probe_and_normalization_words
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.continued_pretraining import initialize_token_embeddings
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.logistic_probes import LogisticProbe, train_logistic_probe
from lexical_acquisition_mechanisms.neutral_contexts import NeutralContext
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.probe_and_normalization_words import ProbeWord
from lexical_acquisition_mechanisms.settings import ModelSource, Settings
from lexical_acquisition_mechanisms.word_representations import word_representations

PROBES_DIRECTORY = Path("outputs/probes")
# 学習していないトークンの初期値の乱数の種に足す数（未知語の初期値と同じ乱数にしないため）
UNTRAINED_TOKEN_SEED_OFFSET = 1
BATCH_SIZE = 32


@dataclass
class ProbeAccuracy:
    token_count: int
    layer: int
    training_accuracy: float
    held_out_accuracy: float
    # 学習用の語の、正例と負例の数
    positives: int
    negatives: int


def probe_paths(model: str) -> tuple[Path, Path]:
    directory = PROBES_DIRECTORY / model
    return directory / "real_word_probes.npz", directory / "real_word_probe_accuracy.csv"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="settings.yaml の models の名前")


def train_real_word_probes(settings: Settings, arguments: argparse.Namespace) -> None:
    model_name = arguments.model
    if model_name not in settings.models:
        raise CommandStopped(
            f"モデル {model_name} は settings.yaml の models にありません。",
            rerun_from=f"unknown-word-acquisition train-real-word-probes --model {model_name}",
        )
    step = Step(
        f"unknown-word-acquisition train-real-word-probes --model {model_name}",
        inputs={
            probe_and_normalization_words.PROBE_WORDS_PATH: probe_and_normalization_words.COMMAND,
            neutral_contexts.CONTEXTS_PATH: neutral_contexts.COMMAND,
        },
        outputs=list(probe_paths(model_name)),
        settings={
            "model": settings.models[model_name].model_dump(),
            "noise_scale": settings.training.new_token_embedding_initialization.noise_scale,
            "seed": settings.words.seed,
            "untrained_token_seed_offset": UNTRAINED_TOKEN_SEED_OFFSET,
        },
    )
    run_unless_unchanged(
        step,
        lambda: write_real_word_probes(
            settings,
            model_name,
            lambda source: LanguageModel.load(source, batch_size=BATCH_SIZE),
            probe_and_normalization_words.PROBE_WORDS_PATH,
            neutral_contexts.CONTEXTS_PATH,
        ),
        arguments.recompute,
    )


def write_real_word_probes(
    settings: Settings,
    model_name: str,
    load_model: Callable[[ModelSource], LanguageModel],
    probe_words_path: Path,
    contexts_path: Path,
    output_paths: tuple[Path, Path] | None = None,
) -> None:
    probes_path, accuracy_path = output_paths or probe_paths(model_name)
    words = read_rows(probe_words_path, ProbeWord)
    contexts = [row.context for row in read_rows(contexts_path, NeutralContext)]
    model = load_model(settings.models[model_name])
    untrained = [int(row.token_ids) for row in words if row.kind == "untrained-token"]
    if untrained:
        # 使われていない最初のトークン番号が、トークナイザの語彙の大きさ
        vocabulary_size = model.tokenizer.unused_token_ids(1)[0]
        initialize_token_embeddings(
            model.model,
            untrained,
            vocabulary_size,
            settings.training.new_token_embedding_initialization.noise_scale,
            settings.words.seed + UNTRAINED_TOKEN_SEED_OFFSET,
        )
    states = word_representations(
        model, [[int(token) for token in row.token_ids.split()] for row in words], contexts
    )
    arrays = {}
    accuracy = []
    for token_count in sorted({row.token_count for row in words}):
        indices = [index for index, row in enumerate(words) if row.token_count == token_count]
        for layer in range(states.shape[2]):
            probe, result = _train_layer(words, indices, states[:, :, layer], token_count, layer)
            arrays[f"{token_count}-{layer}-weights"] = probe.weights
            arrays[f"{token_count}-{layer}-biases"] = probe.biases
            arrays[f"{token_count}-{layer}-mean"] = probe.mean
            arrays[f"{token_count}-{layer}-scale"] = probe.scale
            accuracy.append(result)
    probes_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(probes_path, **arrays)
    write_rows(accuracy_path, ProbeAccuracy, accuracy)


def _train_layer(
    words: Sequence[ProbeWord],
    indices: Sequence[int],
    states: np.ndarray,
    token_count: int,
    layer: int,
) -> tuple[LogisticProbe, ProbeAccuracy]:
    """1つのトークン数・層のプローブ（実在語がクラス 1）と、その正解率。"""

    def samples(split: str) -> tuple[np.ndarray, np.ndarray]:
        chosen = [index for index in indices if words[index].split == split]
        features = states[chosen].reshape(-1, states.shape[-1])
        labels = np.repeat(
            [int(words[index].kind == "real-word") for index in chosen], states.shape[1]
        )
        return features, labels

    training_features, training_labels = samples("training")
    probe = train_logistic_probe(training_features, training_labels, 2)
    held_out_features, held_out_labels = samples("held-out")
    held_out = probe.accuracy(held_out_features, held_out_labels) if len(held_out_labels) else 0.0
    positives = int(training_labels.sum()) // states.shape[1]
    negatives = len(training_labels) // states.shape[1] - positives
    training = probe.accuracy(training_features, training_labels)
    result = ProbeAccuracy(token_count, layer, training, held_out, positives, negatives)
    return probe, result
