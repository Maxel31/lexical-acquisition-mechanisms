"""GPU で run ごとに測る長い処理の、run ごとの途中の結果。

測り終えた run の結果を <最初の生成物>.runs/<run>.pickle に、処理の指紋（コマンド・設定・入力の
ハッシュ。output_records.step_fingerprint）とともに書く。止まった後に同じ処理を実行し直すと、指紋が
同じ run の結果を読み、残りの run から測る。処理を終えたら消す（clear）。--recompute のときは、
残っている結果を使わない。
"""

import hashlib
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

from lexical_acquisition_mechanisms.output_records import Step, step_fingerprint


class RunResults:
    def __init__(self, step: Step, recompute: bool = False) -> None:
        self._directory = Path(f"{step.outputs[0]}.runs")
        fingerprint = json.dumps(step_fingerprint(step), sort_keys=True)
        self._key = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
        if recompute:
            self.clear()

    def get(self, run: str) -> Any | None:
        """測り終えた run の結果（なければ、または処理の指紋が違えば None）。"""
        path = self._directory / f"{run}.pickle"
        if not path.is_file():
            return None
        with path.open("rb") as file:
            key, value = pickle.load(file)
        return value if key == self._key else None

    def put(self, run: str, value: Any) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._directory / f"{run}.pickle"
        temporary = path.with_name(path.name + ".partial")
        with temporary.open("wb") as file:
            pickle.dump((self._key, value), file)
        temporary.replace(path)

    def clear(self) -> None:
        shutil.rmtree(self._directory, ignore_errors=True)
