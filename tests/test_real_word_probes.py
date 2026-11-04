"""実在語らしさのプローブの学習のテスト。極小のモデルを CPU で使う。"""

from pathlib import Path

import numpy as np

from lexical_acquisition_mechanisms.continued_pretraining import initialize_token_embeddings
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.neutral_contexts import NeutralContext
from lexical_acquisition_mechanisms.probe_and_normalization_words import ProbeWord
from lexical_acquisition_mechanisms.real_word_probes import ProbeAccuracy, write_real_word_probes
from lexical_acquisition_mechanisms.settings import ModelSource, load_settings
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_probes_are_trained_for_each_token_count_and_layer(
    tmp_path: Path, tiny_model_source: ModelSource
) -> None:
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    settings = settings.model_copy(update={"models": {"tiny": tiny_model_source}})
    tokenizer = WordTokenizer.from_model(tiny_model_source)
    words = []
    for index, word in enumerate(
        ["pasta", "bread", "cheese", "soup", "giraffe", "lasagna", "dumpling", "crocodile"]
    ):
        ids = tokenizer.word_token_ids(word)
        split = "held-out" if index % 4 == 3 else "training"
        words.append(ProbeWord(len(ids), "real-word", word, " ".join(map(str, ids)), split))
    for index, token_id in enumerate(range(100340, 100344)):
        words.append(
            ProbeWord(
                1,
                "untrained-token",
                f"<untrained_{index}>",
                str(token_id),
                "training" if index < 3 else "held-out",
            )
        )
    giraffe, lasagna = tokenizer.word_token_ids("giraffe"), tokenizer.word_token_ids("lasagna")
    for split, ids in (
        ("training", [giraffe[0], lasagna[1]]),
        ("held-out", [lasagna[0], giraffe[1]]),
    ):
        words.append(
            ProbeWord(2, "nonword", tokenizer.decode(ids).strip(), " ".join(map(str, ids)), split)
        )
    write_rows(tmp_path / "probe_words.csv", ProbeWord, words)
    contexts = [NeutralContext(0, ""), NeutralContext(1, "Do you know about the")]
    write_rows(tmp_path / "contexts.csv", NeutralContext, contexts)
    outputs = (tmp_path / "probes.npz", tmp_path / "accuracy.csv")

    write_real_word_probes(
        settings,
        "tiny",
        lambda source: LanguageModel.load(source, device="cpu"),
        tmp_path / "probe_words.csv",
        tmp_path / "contexts.csv",
        outputs,
    )

    accuracy = read_rows(outputs[1], ProbeAccuracy)
    # 1・2トークン × 埋め込みと2つの層
    assert [(row.token_count, row.layer) for row in accuracy] == [
        (count, layer) for count in (1, 2) for layer in range(3)
    ]
    one_token = next(row for row in accuracy if row.token_count == 1)
    assert (one_token.positives, one_token.negatives) == (3, 3)
    assert all(row.training_accuracy >= 0.5 for row in accuracy)
    probes = np.load(outputs[0])
    assert probes["1-2-weights"].shape == (2, 32)


def test_untrained_tokens_do_not_copy_the_unknown_word_initialization(
    tmp_path: Path, tiny_model_source: ModelSource
) -> None:
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    settings = settings.model_copy(update={"models": {"tiny": tiny_model_source}})
    token_ids = list(range(100278, 100282))
    words = [
        ProbeWord(1, kind, f"<{kind}_{index}>", str(token_id), split)
        for index, token_id in enumerate(token_ids)
        for kind, split in (("untrained-token", "training"),)
    ]
    real = WordTokenizer.from_model(tiny_model_source).word_token_ids("pasta")
    words.append(ProbeWord(1, "real-word", "pasta", str(real[0]), "training"))
    write_rows(tmp_path / "probe_words.csv", ProbeWord, words)
    write_rows(tmp_path / "contexts.csv", NeutralContext, [NeutralContext(0, "")])
    loaded = []

    def load_model(source: ModelSource) -> LanguageModel:
        loaded.append(LanguageModel.load(source, device="cpu"))
        return loaded[-1]

    write_real_word_probes(
        settings,
        "tiny",
        load_model,
        tmp_path / "probe_words.csv",
        tmp_path / "contexts.csv",
        (tmp_path / "probes.npz", tmp_path / "accuracy.csv"),
    )

    # 学習で1トークンの未知語に使う初期値（同じ番号・学習と同じ種）とは、違う埋め込みにする
    training = LanguageModel.load(tiny_model_source, device="cpu")
    vocabulary_size = training.tokenizer.unused_token_ids(1)[0]
    noise_scale = settings.training.new_token_embedding_initialization.noise_scale
    initialize_token_embeddings(
        training.model, token_ids, vocabulary_size, noise_scale, settings.words.seed
    )
    probe_rows = loaded[0].model.get_input_embeddings().weight[token_ids]
    training_rows = training.model.get_input_embeddings().weight[token_ids]
    assert not np.allclose(probe_rows.detach().numpy(), training_rows.detach().numpy())
