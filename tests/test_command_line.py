"""コマンドの入口のテスト。"""

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

from lexical_acquisition_mechanisms.command_line import Action, build_parser, main
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.settings import Settings

SETTINGS_TEXT = "models:\n  tiny-model:\n    repository: ./tiny-model\n"


@pytest.fixture
def in_repository_like_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """settings.yaml がある作業ディレクトリで実行する。"""
    (tmp_path / "settings.yaml").write_text(SETTINGS_TEXT, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def do_nothing(settings: Settings, arguments: argparse.Namespace) -> None:
    pass


def add_review_csv_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--review-csv", type=Path, required=True)


EXAMPLE_ACTIONS = [
    Action("words", "select", "候補から語を選ぶ", do_nothing),
    Action("words", "finalize", "語の一覧を書き出す", do_nothing, add_review_csv_argument),
    Action("passages", "collect", "用例の候補を切り出す", do_nothing),
]


def test_help_lists_stages_and_actions(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"], EXAMPLE_ACTIONS)
    assert exit_info.value.code == 0
    stage_help = capsys.readouterr().out
    assert "実験で操作する語を選ぶ" in stage_help
    assert "Pile から用例を切り出し、語義を判定する" in stage_help

    with pytest.raises(SystemExit):
        main(["words", "--help"], EXAMPLE_ACTIONS)
    action_help = capsys.readouterr().out
    assert "候補から語を選ぶ" in action_help
    assert "語の一覧を書き出す" in action_help


def test_selected_action_runs_with_settings_and_arguments(
    in_repository_like_directory: Path,
) -> None:
    calls = []

    def record_call(settings: Settings, arguments: argparse.Namespace) -> None:
        calls.append((settings.models["tiny-model"].repository, arguments.review_csv))

    action = Action("words", "finalize", "語の一覧を書き出す", record_call, add_review_csv_argument)

    exit_code = main(["words", "finalize", "--review-csv", "data/words/review.csv"], [action])

    assert exit_code == 0
    assert calls == [("./tiny-model", Path("data/words/review.csv"))]


def test_every_action_accepts_recompute(in_repository_like_directory: Path) -> None:
    flags = []

    def record_flag(settings: Settings, arguments: argparse.Namespace) -> None:
        flags.append(arguments.recompute)

    actions = [
        Action("words", "select", "", record_flag),
        Action("passages", "collect", "", record_flag),
    ]

    main(["words", "select"], actions)
    main(["passages", "collect", "--recompute"], actions)

    assert flags == [False, True]


@pytest.mark.parametrize(
    "command_line_arguments", [[], ["words"], ["unknown-stage", "select"], ["words", "x"]]
)
def test_incomplete_or_unknown_command_is_a_usage_error(command_line_arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(command_line_arguments, EXAMPLE_ACTIONS)
    assert exit_info.value.code == 2


def test_stopped_action_shows_what_to_fix_and_where_to_rerun(
    in_repository_like_directory: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def stop(settings: Settings, arguments: argparse.Namespace) -> None:
        raise CommandStopped("approved が 0 の語があります。", rerun_from="words select")

    exit_code = main(["words", "finalize"], [Action("words", "finalize", "", stop)])

    assert exit_code == 1
    message = capsys.readouterr().err
    assert "approved が 0 の語があります。" in message
    assert "uv run lexical-acquisition-mechanisms words select" in message


def test_missing_settings_file_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = main(["words", "select"], EXAMPLE_ACTIONS)

    assert exit_code == 1
    message = capsys.readouterr().err
    assert "settings.yaml がありません" in message
    assert "uv run lexical-acquisition-mechanisms words select" in message


def test_action_in_a_stage_not_in_the_readme_is_rejected() -> None:
    with pytest.raises(ValueError, match="段階 word は README にありません"):
        build_parser([Action("word", "select", "", do_nothing)])


@pytest.mark.parametrize(
    "command",
    [
        [sys.executable, "-m", "lexical_acquisition_mechanisms"],
        [str(Path(sys.executable).parent / "lexical-acquisition-mechanisms")],
    ],
    ids=["python-module", "installed-command"],
)
def test_entry_points_show_help(command: list[str]) -> None:
    completed = subprocess.run(
        [*command, "--help"], capture_output=True, text=True, check=False, timeout=60
    )

    assert completed.returncode == 0, completed.stderr
    assert "lexical-acquisition-mechanisms" in completed.stdout
