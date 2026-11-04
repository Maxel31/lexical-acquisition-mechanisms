"""用例の窓の切り出しのテスト。"""

from lexical_acquisition_mechanisms.passage_replacement import WINDOW_MARGIN
from lexical_acquisition_mechanisms.passage_windows import (
    PassageWindow,
    WindowSettings,
    cut_windows,
)
from lexical_acquisition_mechanisms.settings import ModelSource

TEXT = "We had some pasta for dinner, and then we talked about the weather for a long while."
SETTINGS = WindowSettings(
    tokens_per_passage=8, context_characters=70, seed=0, managed_words=frozenset({"pasta"})
)


def occurrence(start: int) -> dict:
    return {
        "word": "pasta",
        "shard": 0,
        "row": start,
        "start": start,
        "context_start": 0,
        "context": TEXT,
    }


def test_windows_are_the_same_with_several_processes(tiny_model_source: ModelSource) -> None:
    occurrences = [occurrence(TEXT.index("pasta"))] * 3

    alone = cut_windows(occurrences, tiny_model_source, SETTINGS, number_of_processes=1)
    together = cut_windows(occurrences, tiny_model_source, SETTINGS, number_of_processes=2)

    assert alone == together
    assert all(isinstance(window, PassageWindow) for window in alone)
    assert all(len(window.token_ids) == 8 + WINDOW_MARGIN for window in alone)
