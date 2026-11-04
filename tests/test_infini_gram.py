"""infini-gram への問い合わせのテスト。API の代わりに、決まった回数を返す部品を使う。"""

import json
from pathlib import Path

import httpx
import pytest
from transformers import AutoTokenizer

from lexical_acquisition_mechanisms import infini_gram
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.infini_gram import (
    LLAMA_TOKENIZER,
    InfiniGramClient,
    word_forms,
)


@pytest.fixture(scope="module")
def llama_tokenizer():
    try:
        return AutoTokenizer.from_pretrained(LLAMA_TOKENIZER)
    except OSError as error:
        pytest.skip(f"Llama-2 のトークナイザを読み込めません: {error}")


class StubIndex:
    """語頭の形 "▁past|a"（pasta）が 7 回と、語の途中の形 "Bl|olph" が 1 回だけあるコーパス。"""

    def __init__(self, tokenizer, statuses: list[int] | None = None) -> None:
        self.counts = {
            tuple(tokenizer("pasta", add_special_tokens=False)["input_ids"]): 7,
            tuple(tokenizer("\nBlolph", add_special_tokens=False)["input_ids"][2:]): 1,
        }
        self.statuses = list(statuses or [])
        self.bodies: list[dict] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        if self.statuses:
            return httpx.Response(self.statuses.pop(0))
        if body["index"] == "v4_no-such-index_llama":
            return httpx.Response(200, json={"error": "Invalid index"})
        (forms,) = body["query_ids"]
        count = sum(self.counts.get(tuple(form), 0) for form in forms)
        return httpx.Response(200, json={"count": count, "approx": False})


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    async def return_immediately(seconds: float) -> None:
        pass

    monkeypatch.setattr(infini_gram.asyncio, "sleep", return_immediately)


def make_client(tmp_path: Path, index: StubIndex) -> InfiniGramClient:
    return InfiniGramClient(
        "words generate-unknown-word-candidates",
        tmp_path / "counts.sqlite3",
        transport=httpx.MockTransport(index.handle),
    )


def test_word_forms_cover_cases_and_word_internal_positions(llama_tokenizer) -> None:
    forms = word_forms("blolph", llama_tokenizer)

    assert len(forms) == 6
    decoded = {llama_tokenizer.decode(form) for form in forms}
    assert {"blolph", "Blolph", "BLOLPH"} <= decoded


def test_words_with_any_form_in_the_corpus_are_counted(tmp_path: Path, llama_tokenizer) -> None:
    index = StubIndex(llama_tokenizer)

    counts = make_client(tmp_path, index).occurrences(["pasta", "blolph", "zurvex"], "v4_test")

    assert counts == {"pasta": 7, "blolph": 1, "zurvex": 0}
    assert all(len(body["query_ids"][0]) <= 4 for body in index.bodies)


def test_saved_counts_are_reused(tmp_path: Path, llama_tokenizer) -> None:
    index = StubIndex(llama_tokenizer)
    make_client(tmp_path, index).occurrences(["pasta"], "v4_test")
    number_of_requests = len(index.bodies)

    assert make_client(tmp_path, index).occurrences(["pasta"], "v4_test") == {"pasta": 7}
    assert len(index.bodies) == number_of_requests


def test_rate_limits_are_retried(tmp_path: Path, llama_tokenizer) -> None:
    index = StubIndex(llama_tokenizer, statuses=[429, 503])

    assert make_client(tmp_path, index).occurrences(["pasta"], "v4_test")["pasta"] == 7


def test_persistent_failure_stops_with_the_command_to_rerun(
    tmp_path: Path, llama_tokenizer
) -> None:
    index = StubIndex(llama_tokenizer, statuses=[503] * 100)

    with pytest.raises(CommandStopped) as stopped:
        make_client(tmp_path, index).occurrences(["pasta"], "v4_test")

    assert stopped.value.rerun_from == "words generate-unknown-word-candidates"


def test_error_in_a_successful_response_is_raised(tmp_path: Path, llama_tokenizer) -> None:
    with pytest.raises(RuntimeError, match="Invalid index"):
        make_client(tmp_path, StubIndex(llama_tokenizer)).occurrences(
            ["pasta"], "v4_no-such-index_llama"
        )
