"""dataclass の行の一覧と、CSV のファイルの読み書き。

列は dataclass の項目の順にする。読むときは、項目の型（str・int・float・bool）に直す。読む CSV の
列が項目と合わない（ない列のうち既定値のない項目がある、項目にない列がある）ときは、その列を示して
止まる（古い版のコマンドが書いた CSV や、列を足した CSV など）。
人が表計算ソフトで記入した CSV も読めるように、先頭の BOM（UTF-8 の印）は無視する。
人が approved 列を記入する確認用の CSV は write_review_rows で書く（記入の引き継ぎと BOM）。
"""

import csv
import dataclasses
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, get_type_hints

CONVERTERS = {str: str, int: int, float: float, bool: lambda text: text == "True"}


class UnreadableEncodingError(ValueError):
    """CSV を UTF-8 として読めない（表計算ソフトで別の文字コードで保存したときなど）。"""


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
            reader = csv.DictReader(file)
            _check_columns(path, row_type, reader.fieldnames or [])
            return [
                row_type(**{name: CONVERTERS[types[name]](value) for name, value in row.items()})
                for row in reader
            ]
    except UnicodeDecodeError as error:
        raise UnreadableEncodingError(
            f"{path} を UTF-8 として読めません。表計算ソフトで「CSV UTF-8」の形式で保存し直して"
            "ください。"
        ) from error


def _check_columns(path: Path, row_type: type, columns: Sequence[str]) -> None:
    fields = dataclasses.fields(row_type)
    required = [
        field.name
        for field in fields
        if field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING
    ]
    missing = [name for name in required if name not in columns]
    unknown = [name for name in columns if name not in {field.name for field in fields}]
    if missing or unknown:
        raise ValueError(
            f"{path} の列が、{row_type.__name__} の項目と合いません（ない列: {missing}、"
            f"項目にない列: {unknown}）。この CSV を書いたコマンドを実行し直してください（人が"
            "記入する CSV なら、列を足したり消したりしないでください）。"
        )


def require_readable(path: Path) -> None:
    """CSV があれば、UTF-8 として読めるかを確かめる（読めなければ UnreadableEncodingError）。

    確認用の CSV を書き直すコマンドは、時間のかかる計算（GPU など）の前に呼ぶ。計算の後の
    write_review_rows で止まると、保存し直した後に計算を全てやり直すことになるので。
    """
    if not path.exists():
        return
    try:
        path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as error:
        raise UnreadableEncodingError(
            f"{path} を UTF-8 として読めません。表計算ソフトで「CSV UTF-8」の形式で保存し直して"
            "ください。"
        ) from error


def write_review_rows(
    path: Path, row_type: type, rows: Sequence[Any], key_fields: Sequence[str]
) -> None:
    """人が approved 列を記入する確認用の CSV を書く。

    書き直す前の CSV に、key_fields が同じ行があれば、記入済みの approved を引き継ぐ（語や用例を
    1つ差し替えるたびに、全ての行を確認し直さなくてよいように）。日本語の表計算ソフトが UTF-8 と
    して開けるように、先頭に BOM を付ける。書き直す前の CSV を UTF-8 として読めないときは、記入を
    捨てないように止まる。列が変わった古い CSV は、引き継がずに、そのことを表示する。
    """
    previous: dict[tuple, str] = {}
    if path.exists():
        try:
            for row in read_rows(path, row_type):
                previous[tuple(getattr(row, name) for name in key_fields)] = row.approved
        except UnreadableEncodingError:
            raise
        except (ValueError, TypeError, KeyError) as error:
            print(f"{path} の記入済みの approved を引き継ぎません（{error}）。")
            previous = {}
    for row in rows:
        if not row.approved:
            row.approved = previous.get(tuple(getattr(row, name) for name in key_fields), "")
    write_rows(path, row_type, rows, encoding="utf-8-sig")
