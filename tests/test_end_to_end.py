"""割り当てから全結果の集計までの全てのコマンドを、GPU と LLM のサーバなしで続けて実行するテスト。

データ作成の段階（語と用例。LLM・Pile・基データのダウンロードが要る）の生成物は end_to_end_world で
小さく作る。モデルは極小の OLMo 2（CPU）、LLM は決まった答えを返す部品に置き換える。手作業
（確認用の CSV の記入）は、全ての行に 1 を記入する。
語義の追加の分析と接続は、無作為な重みのモデルでは基準を満たす head がないので、基準を緩めて
（有意水準を 1 にするなど）最後まで通す。
"""

import csv
import shutil
import socket
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from end_to_end_world import CUES, SUBJECTS, build_world, write_world

from lexical_acquisition_mechanisms import (
    attribute_judgments,
    attribute_probes,
    component_adoption,
    dependence_regression,
    pretrained_model_checks,
    second_sense_evaluation_data,
    significant_heads_and_mlps,
    task_questions,
)
from lexical_acquisition_mechanisms.command_line import main
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.llm_client import LlmClient
from lexical_acquisition_mechanisms.settings import ModelSource, load_settings
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_LENGTH = 128
MODEL = "tiny"


def write_settings(directory: Path, model: ModelSource) -> None:
    """リポジトリの settings.yaml を、極小の規模にしたもの。"""
    content = yaml.safe_load((REPOSITORY_ROOT / "settings.yaml").read_text(encoding="utf-8"))
    content["models"] = {MODEL: {"repository": model.repository}}
    content["unknown_word_acquisition"]["exposure_levels"] = [0, 1, 2, 3, 4, 5, 6, 8]
    content["sense_addition"]["passages_per_added_sense"] = 2
    content["second_sense_addition"] = {
        "minimum_first_exposure": 4,
        "second_exposure_levels": [0, 1, 2, 3],
    }
    content["passages"] |= {"tokens_per_passage": 24, "evaluation_passages_per_word": 2}
    content["evaluation"] |= {
        "sentences_per_natural_word": 3,
        "passages_per_natural_word": 4,
        "probe_words_per_token_count": 6,
        "normalization_nouns_per_token_count": 3,
        "number_of_task_examples": 1,
        "minimal_pairs_per_sense_pair": 2,
        "minimal_pair_candidate_factor": 2,
        "number_of_neutral_contexts": 2,
    }
    content["training"] |= {
        "learning_rate_by_model": {MODEL: 1.0e-4},
        "batch_size_in_tokens": 4 * SEQUENCE_LENGTH,
        "sequence_length": SEQUENCE_LENGTH,
        "sequences_per_device_by_model": {MODEL: 2},
        "gradient_checkpointing_by_model": {MODEL: False},
        "checkpoint_fractions": [],
        # 2つ目の語義の追加では、基データの残りの系列を使う
        "number_of_base_sequences_used": 48,
    }
    (directory / "settings.yaml").write_text(
        yaml.safe_dump(content, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def free_port() -> int:
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        return connection.getsockname()[1]


def run(*arguments: str) -> None:
    assert main(list(arguments)) == 0, f"止まったコマンド: {' '.join(arguments)}"


def approve_all(path: Path) -> None:
    """確認用の CSV の approved 列に、全て 1 を記入する。"""
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["approved"] = "1"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_all(experiment: str) -> None:
    run("training", "write-run-settings", "--experiment", experiment, "--model", MODEL)
    for path in sorted(Path("outputs/runs", experiment, MODEL).glob("*.yaml")):
        run("training", "run", "--run-settings", str(path))


@pytest.mark.slow
def test_all_commands_run_to_the_end(
    tmp_path: Path, tiny_model_source: ModelSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    shutil.copytree(REPOSITORY_ROOT / "llm_prompts", tmp_path / "llm_prompts")
    write_settings(tmp_path, tiny_model_source)
    settings = load_settings()
    tokenizer = WordTokenizer.from_model(tiny_model_source)
    world = build_world(settings, tokenizer, tiny_model_source.repository)
    write_world(settings, tokenizer, world)
    # GPU の代わりに CPU、LLM の代わりに決まった答え、学習は1つのプロセスで
    load = LanguageModel.load.__func__

    def load_on_cpu(cls, source, device="cuda", dtype=torch.float32, batch_size=16):
        return load(cls, source, device="cpu", dtype=dtype, batch_size=batch_size)

    monkeypatch.setattr(LanguageModel, "load", classmethod(load_on_cpu))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("MASTER_PORT", str(free_port()))
    monkeypatch.setenv("WANDB_MODE", "disabled")

    def answer(self, requests):
        answers = []
        for request in requests:
            if request.prompt.name == "list-paraphrases":
                answers.append({"paraphrases": world.paraphrases})
            elif request.prompt.name == "choose-paraphrase":
                first = request.values["candidates"].split(", ")[0]
                answers.append({"reason": "", "choice": first})
            elif request.prompt.name == "generate-minimal-pairs":
                pairs = []
                for number in range(int(request.values["number"])):
                    subject = SUBJECTS[number % len(SUBJECTS)]
                    cues = CUES[number % len(CUES)], CUES[(number + 1) % len(CUES)]
                    pairs.append(
                        {
                            "sentence": f"{subject} {cues[0]} the ___ today.",
                            "first_cue": cues[0],
                            "second_cue": cues[1],
                        }
                    )
                answers.append({"pairs": pairs})
            elif request.prompt.name == "judge-minimal-pair":
                answers.append({"reason": "", "sentence_1_fits": True, "sentence_2_fits": True})
            else:
                raise AssertionError(f"想定していない LLM への問い合わせ: {request.prompt.name}")
        return answers

    monkeypatch.setattr(LlmClient, "ask_all", answer)
    # 作った語の synset は WordNet にないので、語義を指せる語はないものとする
    monkeypatch.setattr(task_questions, "sense_words", lambda synset: set())
    # 無作為な重みでは、最小対の手がかりが効かず、課題の正解率も偶然の水準を超えないので、
    # 全ての候補を通し、全ての課題を使う
    for module in (pretrained_model_checks, second_sense_evaluation_data):
        monkeypatch.setattr(
            module, "cue_effects", lambda model, candidates, words: np.ones((len(candidates), 2))
        )
    monkeypatch.setattr(
        pretrained_model_checks,
        "usable_tasks",
        lambda *arguments: {"category", "part-of-speech", "sense"},
    )

    run("assignments", "build", "--experiment", "unknown-word-acquisition")
    run("assignments", "build", "--experiment", "sense-addition")
    run("assignments", "build", "--experiment", "second-sense-addition")
    run("training-data", "build", "--experiment", "unknown-word-acquisition")
    run("training-data", "build", "--experiment", "second-sense-addition")
    run("evaluation-data", "extract-natural-word-sentences")
    run("evaluation-data", "build-probe-and-normalization-words")
    run("evaluation-data", "generate-minimal-pairs")
    run("evaluation-data", "judge-minimal-pairs")
    run("evaluation-data", "build-task-questions")
    run("evaluation-data", "check-with-pretrained-models")
    review = Path("data/evaluation/minimal_pairs_review.csv")
    approve_all(review)
    run("evaluation-data", "finalize", "--review-csv", str(review))
    run("evaluation-data", "generate-second-sense-minimal-pairs")
    run("evaluation-data", "build-second-sense-questions")

    train_all("unknown-word-acquisition")
    train_all("second-sense-addition")
    run("second-sense-addition", "measure-context-effects", "--model", MODEL)
    for action in (
        "train-real-word-probes",
        "measure-indicators",
        "analyze-acquisition-order",
    ):
        run("unknown-word-acquisition", action, "--model", MODEL)
    run("unknown-word-acquisition", "make-report")

    run("training-data", "build", "--experiment", "sense-addition")
    train_all("sense-addition")
    # 無作為な重みでは基準を満たさないので、基準を緩めて最後まで通す
    monkeypatch.setattr(significant_heads_and_mlps, "FALSE_DISCOVERY_RATE", 1.0)
    monkeypatch.setattr(component_adoption, "FALSE_DISCOVERY_RATE", 1.0)
    monkeypatch.setattr(component_adoption, "SIGNIFICANCE_LEVEL", 1.0)
    monkeypatch.setattr(attribute_probes, "SIGNIFICANCE_LEVEL", 1.0)
    monkeypatch.setattr(attribute_judgments, "SIGNIFICANCE_LEVEL", 1.0)
    monkeypatch.setattr(
        attribute_judgments, "INEFFECTIVE_KINDS", {"category": (), "part-of-speech": ()}
    )
    monkeypatch.setattr(dependence_regression, "MINIMUM_KNOWN_DROP", float("-inf"))
    for action in (
        "measure-sense-accuracy",
        "measure-head-and-mlp-effects",
        "select-significant-heads-and-mlps",
        "train-attribute-probes",
        "test-attribute-directions",
    ):
        run("sense-addition", action, "--model", MODEL)
    run("sense-addition", "make-report")

    for action in ("measure-dependence", "fit-regression", "compare-head-maps"):
        run("dependence-on-category-judging-heads", action, "--model", MODEL)
    run("dependence-on-category-judging-heads", "make-report")
    run("results", "collect")

    assert Path("outputs/results/all_results.parquet").stat().st_size > 0
    assert Path("outputs/figures/model_sizes.png").stat().st_size > 0
