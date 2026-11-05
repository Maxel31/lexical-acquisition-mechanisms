"""語の内部表現と、ロジスティック回帰のプローブのテスト。"""

import numpy as np
import pytest

from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.logistic_probes import train_logistic_probe
from lexical_acquisition_mechanisms.settings import ModelSource
from lexical_acquisition_mechanisms.word_representations import word_representations


def test_representations_of_words_in_contexts(tiny_model_source: ModelSource) -> None:
    model = LanguageModel.load(tiny_model_source, device="cpu")
    pasta = model.tokenizer.word_token_ids("pasta")
    zubbet = model.tokenizer.word_token_ids("zubbet")

    states = word_representations(model, [pasta, zubbet], ["", "Do you know about the"])

    assert states.shape == (2, 2, model.number_of_layers + 1, 32)
    alone = model.residual_streams([zubbet], [len(zubbet) - 1])[0]
    assert np.allclose(states[1, 0], alone, atol=1e-5)
    in_context = model.tokenizer.encode(["Do you know about the", zubbet])
    expected = model.residual_streams([in_context], [len(in_context) - 1])[0]
    assert np.allclose(states[1, 1], expected, atol=1e-5)


@pytest.mark.parametrize("number_of_classes", [2, 3])
def test_probe_separates_clusters(number_of_classes: int) -> None:
    generator = np.random.default_rng(0)
    centers = generator.normal(0, 3, (number_of_classes, 8))
    labels = np.repeat(np.arange(number_of_classes), 50)
    features = centers[labels] + generator.normal(0, 1, (len(labels), 8))

    probe = train_logistic_probe(features, labels, number_of_classes)

    assert probe.accuracy(features, labels) > 0.95
    probabilities = probe.probabilities(features)
    assert probabilities.shape == (len(labels), number_of_classes)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
