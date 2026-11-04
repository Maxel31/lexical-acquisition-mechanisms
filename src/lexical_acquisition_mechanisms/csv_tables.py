"""dataclass の行の一覧と、CSV のファイルの読み書き。

列は dataclass の項目の順にする。読むときは、項目の型（str・int・float・bool）に直す。
人が表計算ソフトで記入した CSV も読めるように、先頭の BOM（UTF-8 の印）は無視する。
人が approved 列を記入する確認用の CSV は write_review_rows で書く（記入の引き継ぎと BOM）。
"""

import csv
import dataclasses
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, get_type_hints

CONVERTERS = {str: str, int: int, float: float, bool: lambda text: text == "True"}


def write_rows(path: Path, row_type: type, rows: Iterable[Any], encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [field.name for field in dataclasses.fields(row_type)]
    with path.open("w", encoding=encoding, newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(dataclasses.asdict(row) for row in rows)


def read_rows[Row](path: Path, row_type: type[Row]) -> list[Row]:
    types = get_type_hints(row_type)
    try:
        with path.open(encoding="utf-8-sig", newline="") as file:
            return [
                row_type(**{name: CONVERTERS[types[name]](value) for name, value in row.items()})
                for row in csv.DictReader(file)
            ]
    except UnicodeDecodeError as error:
        raise ValueError(
            f"{path} を UTF-8 として読めません。表計算ソフトで「CSV UTF-8」の形式で保存し直して"
            "ください。"
        ) from error


def write_review_rows(
    path: Path, row_type: type, rows: Sequence[Any], key_fields: Sequence[str]
) -> None:
    """人が approved 列を記入する確認用の CSV を書く。

    書き直す前の CSV に、key_fields が同じ行があれば、記入済みの approved を引き継ぐ（語や用例を
    1つ差し替えるたびに、全ての行を確認し直さなくてよいように）。日本語の表計算ソフトが UTF-8 と
    して開けるように、先頭に BOM を付ける。
    """
    previous: dict[tuple, str] = {}
    if path.exists():
        try:
            for row in read_rows(path, row_type):
                previous[tuple(getattr(row, name) for name in key_fields)] = row.approved
        except (ValueError, TypeError, KeyError):
            # 列が変わった古い CSV は、引き継がない
            previous = {}
    for row in rows:
        if not row.approved:
            row.approved = previous.get(tuple(getattr(row, name) for name in key_fields), "")
    write_rows(path, row_type, rows, encoding="utf-8-sig")
