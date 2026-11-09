"""未知語の候補を作る処理のテスト。

infini-gram と ARC のダウンロードは、決まった答えを返す部品で置き換える。
"""

import csv
import json
from pathlib import Path

import httpx
import pytest

from lexical_acquisition_mechanisms import infini_gram
from lexical_acquisition_mechanisms.infini_gram import InfiniGramClient
from lexical_acquisition_mechanisms.settings import ModelSource, Settings, load_settings
from lexical_acquisition_mechanisms.unknown_word_candidates import build_unknown_word_candidates

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def arc_row(nonword: str, body_neighbours: int, words_with_onset: int, illegal: int) -> str:
    columns = ["0"] * 26
    columns[0], columns[5] = nonword, str(body_neighbours)
    columns[21], columns[23] = str(words_with_onset), str(illegal)
    return "\t".join(columns)


# 条件を満たす非語と、満たさない非語（語尾の部分がない・語頭の子音の並びがない・2文字の並びがない）
ARC_TSV = "\n".join(
    [arc_row(word, 3, 5, 0) for word in ["blolph", "skrint", "thwurg", "glimbrast", "prandulk"]]
    + [arc_row("zzorp", 0, 5, 0), arc_row("vlimp", 3, 0, 0), arc_row("bnast", 3, 5, 1)]
)


class StubServices:
    """ARC の TSV を返す。infini-gram は、最初の問い合わせの語だけ現れると答える。"""

    def __init__(self) -> None:
        self.first_query: str | None = None

    def handle(self, request: httpx.Request) -> httpx.Response:
        if "figshare" in str(request.url):
            return httpx.Response(200, text=ARC_TSV)
        query = json.dumps(json.loads(request.content)["query_ids"])
        self.first_query = self.first_query or query
        return httpx.Response(200, json={"count": int(query == self.first_query)})


@pytest.fixture
def settings(tiny_model_source: ModelSource, wordnet_directory: Path) -> Settings:
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    unknown_words = settings.unknown_words.model_copy(update={"candidates_per_group": 2})
    return settings.model_copy(
        update={"models": {"tiny": tiny_model_source}, "unknown_words": unknown_words}
    )


def test_candidates_are_absent_strings_of_two_and_three_tokens(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def return_immediately(seconds: float) -> None:
        pass

    monkeypatch.setattr(infini_gram.asyncio, "sleep", return_immediately)
    services = StubServices()
    transport = httpx.MockTransport(services.handle)
    client = InfiniGramClient("test", tmp_path / "counts.sqlite3", transport=transport)
    output_path = tmp_path / "unknown_word_candidates.csv"

    build_unknown_word_candidates(settings, client, output_path, arc_transport=transport)

    with output_path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    far_words = {row["word"] for row in rows if row["spelling_closeness"] == "far"}
    assert far_words <= {"blolph", "skrint", "thwurg", "glimbrast", "prandulk"}
    near = [row for row in rows if row["spelling_closeness"] == "near"]
    assert {int(row["token_count"]) for row in rows} <= {2, 3}
    for row in near:
        assert len(row["word"]) == len(row["source_word"])
        differences = sum(a != b for a, b in zip(row["word"], row["source_word"], strict=True))
        assert 1 <= differences <= 2
    for token_count in (2, 3):
        absent_near = [
            row
            for row in near
            if row["token_count"] == str(token_count) and row["occurrences"] == "0"
        ]
        assert len(absent_near) >= 2
    assert sum(row["occurrences"] != "0" for row in rows) == 1
