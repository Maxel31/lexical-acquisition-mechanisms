"""CSV の読み書きのテスト。"""

from dataclasses import dataclass
from pathlib import Path

import pytest

from lexical_acquisition_mechanisms.csv_tables import read_rows, write_review_rows


@dataclass
class Row:
    word: str
    note: str
    approved: str = ""


def test_review_rows_keep_earlier_approvals(tmp_path: Path) -> None:
    path = tmp_path / "review.csv"
    write_review_rows(path, Row, [Row("pasta", "a"), Row("bread", "b")], ("word",))
    # 人が記入した後、語を1つ差し替えて書き直す
    rows = read_rows(path, Row)
    rows[0].approved, rows[1].approved = "1", "0"
    write_review_rows(path, Row, rows, ("word",))

    write_review_rows(path, Row, [Row("pasta", "changed"), Row("cheese", "c")], ("word",))

    assert [(row.word, row.approved) for row in read_rows(path, Row)] == [
        ("pasta", "1"),
        ("cheese", ""),
    ]
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_csv_saved_in_another_encoding_gives_a_clear_message(tmp_path: Path) -> None:
    path = tmp_path / "review.csv"
    path.write_bytes("word,note,approved\nパスタ,a,1\n".encode("cp932"))

    with pytest.raises(ValueError, match="CSV UTF-8"):
        read_rows(path, Row)


def test_columns_that_do_not_match_the_fields_give_a_clear_message(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_text("word,approved,comment\npasta,1,ok\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"ない列: \['note'\]、項目にない列: \['comment'\]"):
        read_rows(path, Row)
