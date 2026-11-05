"""用例の窓の切り出しのテスト。"""

import pytest

from lexical_acquisition_mechanisms import passage_windows
from lexical_acquisition_mechanisms.passage_windows import (
    PassageWindow,
    WindowSettings,
    cut_windows,
)
from lexical_acquisition_mechanisms.settings import ModelSource

TEXT = "We had some pasta for dinner, and then we talked about the weather for a long while."
SETTINGS = WindowSettings(
    tokens_per_passage=8,
    context_characters=70,
    seed=0,
    other_forms={"pasta": frozenset({"pastas", "pastaes"}), "cherry": frozenset({"cherries"})},
    managed_forms={"pasta": "pasta", "pastas": "pasta", "pastaes": "pasta"},
)


def occurrence(start: int, word: str = "pasta", text: str = TEXT) -> dict:
    return {
        "word": word,
        "shard": 0,
        "row": start,
        "start": start,
        "context_start": 0,
        "context": text,
    }


def test_windows_are_the_same_with_several_processes(
    tiny_model_source: ModelSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1件ずつのまとまりに分け、別々の出現（行が違うので、窓の位置も違いうる）の順が保たれるか
    monkeypatch.setattr(passage_windows, "CHUNK_SIZE", 1)
    occurrences = [occurrence(TEXT.index("pasta")) | {"row": row} for row in range(6)]

    alone = cut_windows(occurrences, tiny_model_source, SETTINGS, number_of_processes=1)
    together = cut_windows(occurrences, tiny_model_source, SETTINGS, number_of_processes=2)

    assert alone == together
    assert all(isinstance(window, PassageWindow) for window in alone)
    assert all(len(window.token_ids) == 8 for window in alone)


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        # 語の別の形（規則にない複数形も、inflected_forms が作った形を使う）
        ("Some cherry pie and two cherries were left.", "語の標準の形でない形を含む"),
        # ほかの管理対象の語の別の形
        ("Some cherry pie and two pastas were left.", "ほかの管理対象の語を含む"),
    ],
)
def test_windows_with_other_forms_are_not_used(
    tiny_model_source: ModelSource, text: str, reason: str
) -> None:
    occurrences = [occurrence(text.index("cherry"), "cherry", text)]

    assert cut_windows(occurrences, tiny_model_source, SETTINGS, number_of_processes=1) == [reason]
