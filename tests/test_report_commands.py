"""図と表を作るコマンド（make-report）が、結果のないときに止まるかのテスト。"""

import argparse
from pathlib import Path

import pytest

from lexical_acquisition_mechanisms import (
    dependence_report,
    sense_addition_report,
    unknown_word_report,
)
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.settings import load_settings

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("module", "previous"),
    [
        (unknown_word_report, "analyze-acquisition-order"),
        (sense_addition_report, "test-attribute-directions"),
        (dependence_report, "fit-regression"),
    ],
)
def test_reports_stop_when_no_model_has_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module, previous: str
) -> None:
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(CommandStopped, match=f"{previous} を終えたモデルがない"):
        module.make_report(settings, argparse.Namespace(recompute=False))

    # 中身のない図のディレクトリを、終えた処理として記録しない
    assert not Path("outputs").exists()
