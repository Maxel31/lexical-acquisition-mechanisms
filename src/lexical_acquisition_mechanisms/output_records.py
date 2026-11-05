"""生成物の記録と、入力と設定が前回と同じ処理の省略。

処理が書き出した生成物ごとに、隣に記録（<生成物の名前>.record.json）を書く。記録には、処理の
コマンド・処理が使った設定の値・入力のハッシュ・生成物のハッシュ・コードの版を残す。
同じ処理をもう一度実行したとき、生成物がそろっていて、記録のコマンド・設定・入力が今と同じなら省く。

- 記録のない入力（書き出す途中で止まった生成物など）は使わずに止まる。
- 記録の後に書き換えられた生成物（人が記入した確認用の CSV など）は、上書きする前に写しを残す。
- 人が記入して渡すファイル（--review-csv。別の名前で保存したものもある）は、入力ではなく、パスと
  中身のハッシュを設定として記録する（edited_file_settings）。
"""

import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from lexical_acquisition_mechanisms.command_stopped import CommandStopped

RECORD_SUFFIX = ".record.json"


@dataclass(frozen=True)
class Step:
    """1つの処理が読むもの・書くもの・使う設定。"""

    # この処理のコマンド（例: "assignments build --experiment sense-addition"）
    command: str
    # 入力のファイルと、それを書き出すコマンド（入力が使えないときに、実行するコマンドとして示す）
    inputs: Mapping[Path, str]
    outputs: Sequence[Path]
    # 処理が使う設定の値（JSON にできるもの）
    settings: Mapping[str, Any] = field(default_factory=dict)


def run_unless_unchanged(step: Step, compute: Callable[[], None], recompute: bool = False) -> bool:
    """入力と設定が前回と同じなら省き、そうでなければ compute を実行して記録を書く。

    実行したら True、省いたら False を返す。
    """
    if not step.outputs:
        raise ValueError(f"処理 {step.command} の生成物が指定されていません。")
    for path, writing_command in step.inputs.items():
        if _read_record(path) is None:
            raise CommandStopped(
                f"入力 {path} がないか、書き出す途中で止まっています。"
                "先に、これを書き出すコマンドを実行してください。",
                rerun_from=writing_command,
            )
    fingerprint = step_fingerprint(step)
    if not recompute and all(_read_fingerprint(path) == fingerprint for path in step.outputs):
        print(f"省略しました: {step.command}（入力と設定が前回と同じ。やり直すときは --recompute）")
        return False

    # 記録するのは、実行を始めたときのコードの版
    code_version = _code_version()
    for path in step.outputs:
        _keep_copy_if_edited(path)
        # 途中で失敗したときに、古い記録で「済み」と判断しないよう、先に記録を消す
        record_path_of(path).unlink(missing_ok=True)
    compute()
    missing_outputs = [str(path) for path in step.outputs if not path.exists()]
    if missing_outputs:
        raise RuntimeError(
            f"処理 {step.command} が、生成物 {missing_outputs} を書き出していません。"
        )
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    for path in step.outputs:
        record = {
            **fingerprint,
            "output": {"path": str(path), "hash": hash_path(path), **_file_state(path)},
            "code_version": code_version,
            "created_at": created_at,
        }
        _write_text_atomically(
            record_path_of(path), json.dumps(record, ensure_ascii=False, indent=2) + "\n"
        )
    return True


def step_fingerprint(step: Step) -> dict[str, Any]:
    """処理を省けるかを決める、コマンド・設定・入力のハッシュ。"""
    return {
        "command": step.command,
        "settings": json.loads(json.dumps(step.settings, sort_keys=True)),
        "inputs": {str(path): _current_hash(path) for path in step.inputs},
    }


def edited_file_settings(path: Path, writing_command: str) -> dict[str, str]:
    """人が記入して渡すファイルを、処理の設定として記録する値（パスと中身のハッシュ）。"""
    if not path.is_file():
        raise CommandStopped(
            f"{path} がありません。先に、これを書き出すコマンドを実行して、記入してください。",
            rerun_from=writing_command,
        )
    return {"path": str(path), "hash": hash_path(path)}


def record_path_of(path: Path) -> Path:
    return path.with_name(path.name + RECORD_SUFFIX)


def hash_path(path: Path) -> str:
    """ファイルの中身の SHA-256。

    ディレクトリは、中の全ファイル（記録を除く）の名前と中身から作る。
    """
    if path.is_file():
        return _hash_file(path)
    digest = hashlib.sha256()
    for file_path in _files_in_directory(path):
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8") + b"\0")
        digest.update(_hash_file(file_path).encode("ascii") + b"\n")
    return digest.hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files_in_directory(path: Path) -> list[Path]:
    return sorted(
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file() and not file_path.name.endswith(RECORD_SUFFIX)
    )


def _file_state(path: Path) -> dict[str, int]:
    """大きさと更新時刻（ディレクトリは、中の全ファイルの数・大きさの合計・最新の更新時刻）。"""
    files = [path] if path.is_file() else _files_in_directory(path)
    statistics = [file_path.stat() for file_path in files]
    return {
        "number_of_files": len(statistics),
        "size": sum(statistic.st_size for statistic in statistics),
        "modified_nanoseconds": max((statistic.st_mtime_ns for statistic in statistics), default=0),
    }


def _current_hash(path: Path) -> str:
    """今の中身のハッシュ。

    記録の後に大きさも更新時刻も変わっていなければ、記録のハッシュを使う（大きな重みを、
    省くかどうかを決めるたびに読み直さないため）。
    """
    recorded_output = (_read_record(path) or {}).get("output") or {}
    state = _file_state(path)
    if recorded_output.get("hash") and all(
        recorded_output.get(key) == value for key, value in state.items()
    ):
        return recorded_output["hash"]
    return hash_path(path)


def _keep_copy_if_edited(path: Path) -> None:
    """記録の後に書き換えられたファイルを、上書きする前に写す。"""
    recorded_hash = ((_read_record(path) or {}).get("output") or {}).get("hash")
    if not path.is_file() or recorded_hash is None or _current_hash(path) == recorded_hash:
        return
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    copy_path = path.with_name(f"{path.stem}.edited-{stamp}{path.suffix}")
    shutil.copy2(path, copy_path)
    print(f"記録の後に書き換えられた {path} を、上書きする前に {copy_path} に写しました。")


def _read_record(path: Path) -> dict[str, Any] | None:
    """生成物の記録。生成物か記録がないとき、または記録が壊れているときは None。"""
    record_path = record_path_of(path)
    if not path.exists() or not record_path.exists():
        return None
    try:
        return json.loads(record_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _read_fingerprint(path: Path) -> dict[str, Any] | None:
    record = _read_record(path)
    if record is None:
        return None
    return {key: record.get(key) for key in ("command", "settings", "inputs")}


def code_version() -> str:
    """今のコードの版（git のコミット。コミットしていない変更があれば、その旨を付ける）。"""
    return _code_version()


def _code_version() -> str:
    """git のコミット。コミットしていないコードの変更があれば、その旨を付ける。

    settings.yaml の変更は設定として別に記録するので、コードの変更に数えない。src/ と
    llm_prompts/ の、まだ git に加えていない新しいファイルは、コードの変更に数える。
    """
    try:
        commit = _run_git("rev-parse", "HEAD")
        changes = _run_git(
            "status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)settings.yaml"
        ) or _run_git("status", "--porcelain", "--untracked-files=all", "--", "src", "llm_prompts")
    except (OSError, subprocess.SubprocessError):
        return "不明"
    return commit + ("（コミットしていない変更あり）" if changes else "")


def _run_git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], capture_output=True, text=True, check=True, timeout=30
    )
    return completed.stdout.strip()


def _write_text_atomically(path: Path, text: str) -> None:
    # 複数のプロセス（torchrun で起動した学習など）が同時に書いても衝突しない一時ファイルに書く
    temporary_path = path.with_name(f"{path.name}.writing-{os.getpid()}")
    temporary_path.write_text(text, encoding="utf-8")
    os.replace(temporary_path, path)
