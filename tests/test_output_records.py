"""生成物の記録と、処理の省略のテスト。"""

import json
import subprocess
from pathlib import Path

import pytest

from lexical_acquisition_mechanisms import output_records
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.output_records import (
    Step,
    edited_file_settings,
    hash_path,
    record_path_of,
    run_unless_unchanged,
)

WRITING_COMMAND = "words generate-noun-candidates"


def write_as_output(path: Path, text: str) -> None:
    """前の処理（WRITING_COMMAND）の生成物として、記録つきで書き出す。"""
    step = Step(command=WRITING_COMMAND, inputs={}, outputs=[path], settings={"text": text})
    run_unless_unchanged(step, lambda: path.write_text(text, encoding="utf-8"))


class CountingStep:
    """入力を大文字にして書き出す処理。実行した回数を数える。"""

    def __init__(self, directory: Path) -> None:
        self.input_path = directory / "words.txt"
        self.output_path = directory / "upper_words.txt"
        write_as_output(self.input_path, "pasta\n")
        self.number_of_runs = 0

    def step(self, settings: dict | None = None) -> Step:
        return Step(
            command="words select",
            inputs={self.input_path: WRITING_COMMAND},
            outputs=[self.output_path],
            settings={"number_of_nouns": 72} if settings is None else settings,
        )

    def compute(self) -> None:
        self.number_of_runs += 1
        text = self.input_path.read_text(encoding="utf-8")
        self.output_path.write_text(text.upper(), encoding="utf-8")

    def run(self, settings: dict | None = None, recompute: bool = False) -> bool:
        return run_unless_unchanged(self.step(settings), self.compute, recompute=recompute)


@pytest.fixture
def counting_step(tmp_path: Path) -> CountingStep:
    return CountingStep(tmp_path)


def test_second_run_with_same_input_and_settings_is_skipped(
    counting_step: CountingStep, capsys: pytest.CaptureFixture[str]
) -> None:
    assert counting_step.run() is True
    assert counting_step.run() is False

    assert counting_step.number_of_runs == 1
    assert "省略しました: words select" in capsys.readouterr().out


def test_record_keeps_settings_and_hashes(counting_step: CountingStep) -> None:
    counting_step.run()

    record = json.loads(record_path_of(counting_step.output_path).read_text(encoding="utf-8"))
    assert record["command"] == "words select"
    assert record["settings"] == {"number_of_nouns": 72}
    assert record["inputs"] == {str(counting_step.input_path): hash_path(counting_step.input_path)}
    assert record["output"]["hash"] == hash_path(counting_step.output_path)
    assert record["output"]["size"] == len("PASTA\n")
    assert record["code_version"]
    assert record["created_at"]


# 前回の実行の後に、何かを変える。run() に渡す引数を返す
def change_input(counting_step: CountingStep) -> dict:
    write_as_output(counting_step.input_path, "grape\n")
    return {}


def change_settings(counting_step: CountingStep) -> dict:
    return {"settings": {"number_of_nouns": 60}}


def ask_to_recompute(counting_step: CountingStep) -> dict:
    return {"recompute": True}


def delete_output(counting_step: CountingStep) -> dict:
    counting_step.output_path.unlink()
    return {}


def delete_record(counting_step: CountingStep) -> dict:
    record_path_of(counting_step.output_path).unlink()
    return {}


def break_record(counting_step: CountingStep) -> dict:
    record_path_of(counting_step.output_path).write_text("{", encoding="utf-8")
    return {}


@pytest.mark.parametrize(
    "change",
    [change_input, change_settings, ask_to_recompute, delete_output, delete_record, break_record],
)
def test_step_runs_again_when_something_changed(counting_step: CountingStep, change) -> None:
    counting_step.run()

    assert counting_step.run(**change(counting_step)) is True
    assert counting_step.number_of_runs == 2


def test_edited_output_is_kept_while_the_input_is_unchanged(counting_step: CountingStep) -> None:
    """人が記入した確認用の CSV のように、生成物を書き換えても、入力が同じなら上書きしない。"""
    counting_step.run()
    counting_step.output_path.write_text("PASTA,1\n", encoding="utf-8")

    assert counting_step.run() is False
    assert counting_step.output_path.read_text(encoding="utf-8") == "PASTA,1\n"


def test_edited_output_is_copied_before_it_is_overwritten(counting_step: CountingStep) -> None:
    counting_step.run()
    counting_step.output_path.write_text("PASTA,1\n", encoding="utf-8")

    counting_step.run(**change_input(counting_step))

    copies = list(counting_step.output_path.parent.glob("upper_words.edited-*.txt"))
    assert [copy.read_text(encoding="utf-8") for copy in copies] == ["PASTA,1\n"]
    assert counting_step.output_path.read_text(encoding="utf-8") == "GRAPE\n"


def test_unedited_output_is_overwritten_without_a_copy(counting_step: CountingStep) -> None:
    counting_step.run()

    counting_step.run(recompute=True)

    assert not list(counting_step.output_path.parent.glob("upper_words.edited-*"))


def test_missing_input_stops_with_the_command_that_writes_it(tmp_path: Path) -> None:
    step = Step(
        command="words finalize --review-csv data/words/review.csv",
        inputs={tmp_path / "review.csv": "words measure-hidden-state-similarity"},
        outputs=[tmp_path / "words.csv"],
    )

    with pytest.raises(CommandStopped) as stopped:
        run_unless_unchanged(step, lambda: None)

    assert str(tmp_path / "review.csv") in stopped.value.message
    assert stopped.value.rerun_from == "words measure-hidden-state-similarity"


def test_input_left_by_a_failed_step_is_not_used(counting_step: CountingStep) -> None:
    """前の処理が途中で失敗して残した生成物（記録がない）を、次の処理は使わない。"""
    counting_step.run()

    def fail() -> None:
        counting_step.output_path.write_text("half", encoding="utf-8")
        raise RuntimeError("途中で失敗")

    with pytest.raises(RuntimeError, match="途中で失敗"):
        run_unless_unchanged(counting_step.step(settings={"number_of_nouns": 60}), fail)

    next_step = Step(
        command="words finalize",
        inputs={counting_step.output_path: "words select"},
        outputs=[counting_step.output_path.with_name("final.txt")],
    )
    with pytest.raises(CommandStopped) as stopped:
        run_unless_unchanged(next_step, lambda: None)
    assert stopped.value.rerun_from == "words select"
    assert counting_step.run() is True


def test_step_that_does_not_write_its_output_is_an_error(tmp_path: Path) -> None:
    step = Step(command="words select", inputs={}, outputs=[tmp_path / "never_written.txt"])

    with pytest.raises(RuntimeError, match="never_written.txt"):
        run_unless_unchanged(step, lambda: None)


def test_unchanged_input_is_not_read_again(
    counting_step: CountingStep, monkeypatch: pytest.MonkeyPatch
) -> None:
    """大きな重みを、省くかどうかを決めるたびに読み直さない。"""
    counting_step.run()

    def fail_to_read(path: Path) -> str:
        raise AssertionError(f"{path} を読み直した")

    monkeypatch.setattr(output_records, "_hash_file", fail_to_read)
    assert counting_step.run() is False


def test_directory_hash_follows_its_files_but_not_their_records(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    (directory / "weights.bin").write_bytes(b"\x00\x01")
    original_hash = hash_path(directory)

    (directory / "weights.bin.record.json").write_text("{}", encoding="utf-8")
    assert hash_path(directory) == original_hash

    (directory / "weights.bin").write_bytes(b"\x00\x02")
    assert hash_path(directory) != original_hash


def test_code_version_counts_new_source_files_but_not_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def git(*arguments: str) -> None:
        subprocess.run(
            ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *arguments],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    (tmp_path / "settings.yaml").write_text("models: {}\n", encoding="utf-8")
    git("init", "--quiet")
    git("add", "settings.yaml")
    git("commit", "--quiet", "-m", "最初のコミット")
    monkeypatch.chdir(tmp_path)

    (tmp_path / "settings.yaml").write_text("models: {tiny: ./tiny}\n", encoding="utf-8")
    assert "コミットしていない変更あり" not in output_records._code_version()

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "new_module.py").write_text("", encoding="utf-8")
    first = output_records._code_version()
    assert "コミットしていない変更あり" in first
    # コミットしないまま、もう一度直すと、別の版になる
    (tmp_path / "src" / "new_module.py").write_text("VALUE = 1\n", encoding="utf-8")
    assert output_records._code_version() not in (first, "不明")


def test_files_people_fill_in_are_recorded_by_their_content(tmp_path: Path) -> None:
    # 確認用の CSV を、別の名前で保存した（記録はない）
    review = tmp_path / "review_filled.csv"
    review.write_text("word,approved\npasta,1\n", encoding="utf-8")
    recorded = edited_file_settings(review, "words measure-hidden-state-similarity")
    review.write_text("word,approved\npasta,0\n", encoding="utf-8")

    assert recorded["path"] == str(review)
    assert edited_file_settings(review, "")["hash"] != recorded["hash"]
    with pytest.raises(CommandStopped, match="がありません") as stopped:
        edited_file_settings(tmp_path / "missing.csv", "words measure-hidden-state-similarity")
    assert stopped.value.rerun_from == "words measure-hidden-state-similarity"
