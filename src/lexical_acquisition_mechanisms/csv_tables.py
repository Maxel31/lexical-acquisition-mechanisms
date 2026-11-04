"""dataclass の行の一覧と、CSV のファイルの読み書き。

列は dataclass の項目の順にする。読むときは、項目の型（str・int・float・bool）に直す。
人が表計算ソフトで記入した CSV も読めるように、先頭の BOM（UTF-8 の印）は無視する。
"""

import csv
import dataclasses
from collections.abc import Iterable
from pathlib import Path
from typing import Any, get_type_hints

CONVERTERS = {str: str, int: int, float: float, bool: lambda text: text == "True"}


def write_rows(path: Path, row_type: type, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [field.name for field in dataclasses.fields(row_type)]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(dataclasses.asdict(row) for row in rows)


def read_rows[Row](path: Path, row_type: type[Row]) -> list[Row]:
    types = get_type_hints(row_type)
    with path.open(encoding="utf-8-sig", newline="") as file:
        return [
            row_type(**{name: CONVERTERS[types[name]](value) for name, value in row.items()})
            for row in csv.DictReader(file)
        ]
