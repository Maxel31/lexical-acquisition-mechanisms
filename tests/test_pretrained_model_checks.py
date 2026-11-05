"""学習前のモデルでの最小対の確認のテスト。モデルは、決まった対数確率を返す部品で置き換える。"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.minimal_pairs import MinimalPairCandidate
from lexical_acquisition_mechanisms.pretrained_model_checks import (
    CheckedMinimalPair,
    MinimalPairReviewRow,
    check_minimal_pairs,
)
from lexical_acquisition_mechanisms.settings import ModelSource, load_settings
from lexical_acquisition_mechanisms.word_selection import SelectedWord
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# 語の最後のトークンの対数確率は、手がかりに合えば -1、合わなければ -3
FITS = {"pasta": ("cooked", "ate"), "car": ("parked", "drove")}


class ModelFollowingCues:
    def __init__(self, tokenizer: WordTokenizer) -> None:
        self.tokenizer = tokenizer

    def token_log_probabilities(self, inputs: Sequence[Sequence[int]]) -> list[np.ndarray]:
        results = []
        for token_ids in inputs:
            words = self.tokenizer.decode(token_ids).split()
            values = np.zeros(len(token_ids), dtype=np.float32)
            values[-1] = -1.0 if any(cue in words for cue in FITS[words[-1]]) else -3.0
            results.append(values)
        return results


def candidate(donor_sentence: str, target_sentence: str) -> MinimalPairCandidate:
    return MinimalPairCandidate(
        "car",
        "vehicle",
        1,
        "pasta",
        "noun",
        "other-category",
        donor_sentence,
        target_sentence,
        "",
        "",
    )


def test_pairs_whose_cues_decide_the_sense_are_kept(
    tmp_path: Path, tiny_model_source: ModelSource
) -> None:
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    evaluation = settings.evaluation.model_copy(update={"minimal_pairs_per_sense_pair": 2})
    settings = settings.model_copy(
        update={"models": {"tiny": tiny_model_source}, "evaluation": evaluation}
    )
    tokenizer = WordTokenizer.from_model(tiny_model_source)
    words = [
        SelectedWord(
            word,
            role,
            definition=f"the {word}",
            token_ids=" ".join(map(str, tokenizer.word_token_ids(word))),
        )
        for word, role in (("car", "target"), ("pasta", "noun"))
    ]
    write_rows(tmp_path / "words.csv", SelectedWord, words)
    candidates = [
        candidate("She saw the ___ today.", "She met the ___ today."),
        candidate("She cooked the ___ today.", "She parked the ___ today."),
        candidate("He ate the ___ quickly.", "He drove the ___ quickly."),
        candidate("We cooked the ___ again.", "We parked the ___ again."),
    ]
    write_rows(tmp_path / "candidates.csv", MinimalPairCandidate, candidates)
    outputs = (tmp_path / "checked.csv", tmp_path / "review.csv")

    check_minimal_pairs(
        settings,
        lambda source: ModelFollowingCues(tokenizer),
        tmp_path / "words.csv",
        tmp_path / "candidates.csv",
        outputs,
    )

    # 手がかりが語義を決めない1組目は使わず、通った組を、語義の組ごとに2組まで残す
    checked = read_rows(outputs[0], CheckedMinimalPair)
    assert [pair.donor_sentence for pair in checked] == [
        "She cooked the ___ today.",
        "He ate the ___ quickly.",
    ]
    assert all(pair.donor_cue_effect == 2.0 and pair.target_cue_effect == 2.0 for pair in checked)
    (review,) = read_rows(outputs[1], MinimalPairReviewRow)
    assert (review.donor_sentence, review.target_sentence) == (
        "She cooked the car today.",
        "She parked the car today.",
    )
    assert (review.donor_sense, review.target_sense, review.approved) == (
        "the pasta",
        "the car",
        "",
    )
