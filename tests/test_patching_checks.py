"""勾配による近似を、実際の入れ替えで確かめる処理のテスト。"""

import numpy as np
import pytest

from lexical_acquisition_mechanisms import patching_checks
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.patching_checks import check_estimates, summarize_checks
from lexical_acquisition_mechanisms.settings import ModelSource


def test_largest_and_random_components_are_checked(
    tiny_model_source: ModelSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(patching_checks, "CHECKED_COMPONENTS", 3)
    monkeypatch.setattr(patching_checks, "CHECKED_PAIRS", 1)
    model = LanguageModel.load(tiny_model_source, device="cpu")
    first = model.tokenizer.encode(["She cooked the pasta. Answer:"])
    second = model.tokenizer.encode(["She parked the pasta. Answer:"])
    choice = tuple(model.tokenizer.word_token_ids(word)[0] for word in ("food", "car"))
    rows = [{}, {}]
    # 構成要素は 2層 × (4 head + MLP)。全ての組の平均の絶対値で順位を決める
    estimates = np.zeros((2, 10))
    estimates[0, 7] = 3.0
    estimates[1, [7, 2, 5]] = [3.0, -4.0, 2.0]

    checks = check_estimates(
        model, "other-category-1", 10, rows, estimates, lambda row: (first, second, [3], choice), 0
    )

    largest = [(check.layer, check.head) for check in checks if check.chosen == "largest"]
    assert largest == [(1, 2), (0, 2), (1, 0)]
    randomly = [(check.layer, check.head) for check in checks if check.chosen == "random"]
    assert len(set(randomly)) == 3 and not set(randomly) & set(largest)
    assert len({check.pair for check in checks}) == 1 and checks[0].pair in {10, 11}
    summary = summarize_checks(checks)
    assert [(row.chosen, row.components, row.pairs) for row in summary] == [
        ("largest", 3, 1),
        ("random", 3, 1),
    ]
