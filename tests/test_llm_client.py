"""LLM への問い合わせのテスト。サーバの代わりに、決まった答えを返す部品を使う。"""

import asyncio
import json
import sqlite3
import threading
from pathlib import Path

import httpx
import pytest

from lexical_acquisition_mechanisms import llm_client
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.llm_client import (
    LlmClient,
    LlmRequest,
    PromptTemplate,
    prompt_record_settings,
    with_maximum_items,
)
from lexical_acquisition_mechanisms.settings import LlmSettings

SETTINGS = LlmSettings(
    server_url="http://llm.test/v1", model="test-model", maximum_concurrent_requests=4
)
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"passed": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["passed", "reason"],
    "additionalProperties": False,
}
PROMPT = PromptTemplate("judge-word-sense", "Is $word used as food here?\n$passage\n")
# 待つ処理を差し替える前の asyncio.sleep（本当に待つテストに使う）
REAL_SLEEP = asyncio.sleep


class StubServer:
    """語が pasta なら合格と答える。受けた問い合わせを残す。

    statuses_by_word: 語ごとに、先に返す状態コードの列（尽きたら答える）。
    truncated_words: 長さの上限で切れた答えを返す語（上限が truncated_below 未満のとき）。
    """

    def __init__(
        self,
        statuses_by_word: dict[str, list[int]] | None = None,
        truncated_words: tuple[str, ...] = (),
        truncated_below: int = 10_000,
    ) -> None:
        self.bodies: list[dict] = []
        self.statuses_by_word = statuses_by_word or {}
        self.truncated_words = truncated_words
        self.truncated_below = truncated_below

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        content = body["messages"][0]["content"]
        word = content.split()[1]
        statuses = self.statuses_by_word.get(word, [])
        if statuses:
            return httpx.Response(statuses.pop(0), text="stub error")
        if word in self.truncated_words and body["max_tokens"] < self.truncated_below:
            choice = {"message": {"content": '{"passed": tr'}, "finish_reason": "length"}
            return httpx.Response(200, json={"choices": [choice]})
        answer = {"passed": word == "pasta", "reason": "stub"}
        choice = {"message": {"content": json.dumps(answer)}, "finish_reason": "stop"}
        return httpx.Response(200, json={"choices": [choice]})


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    async def return_immediately(seconds: float) -> None:
        pass

    monkeypatch.setattr(llm_client.asyncio, "sleep", return_immediately)


def make_client(tmp_path: Path, server: StubServer) -> LlmClient:
    return LlmClient(
        SETTINGS,
        tmp_path / "answers.sqlite3",
        rerun_from="passages judge-word-sense",
        transport=httpx.MockTransport(server.handle),
    )


def requests_for(*words: str) -> list[LlmRequest]:
    return [
        LlmRequest(PROMPT, {"word": word, "passage": f"I ate the {word}."}, ANSWER_SCHEMA)
        for word in words
    ]


def test_answers_come_back_in_the_order_of_the_requests(tmp_path: Path) -> None:
    answers = make_client(tmp_path, StubServer()).ask_all(requests_for("car", "pasta", "lamp"))

    assert [answer["passed"] for answer in answers] == [False, True, False]


def test_request_asks_for_the_answer_schema_at_temperature_zero(tmp_path: Path) -> None:
    server = StubServer()

    make_client(tmp_path, server).ask_all(requests_for("pasta"))

    body = server.bodies[0]
    assert body["model"] == "test-model"
    assert body["temperature"] == 0
    assert body["response_format"]["json_schema"]["schema"] == ANSWER_SCHEMA
    assert body["messages"][0]["content"] == "Is pasta used as food here?\nI ate the pasta.\n"


def test_saved_answers_are_reused(tmp_path: Path) -> None:
    server = StubServer()
    make_client(tmp_path, server).ask_all(requests_for("car", "pasta"))

    answers = make_client(tmp_path, server).ask_all(requests_for("pasta", "car", "pasta"))

    assert len(server.bodies) == 2
    assert [answer["passed"] for answer in answers] == [True, False, True]


def test_changed_prompt_is_asked_again_and_changes_the_record_settings(tmp_path: Path) -> None:
    server = StubServer()
    make_client(tmp_path, server).ask_all(requests_for("pasta"))
    changed_prompt = PromptTemplate(PROMPT.name, PROMPT.text + "Answer in JSON.\n")

    request = LlmRequest(changed_prompt, {"word": "pasta", "passage": "x"}, ANSWER_SCHEMA)
    make_client(tmp_path, server).ask_all([request])

    assert len(server.bodies) == 2
    assert prompt_record_settings(SETTINGS, [PROMPT]) != prompt_record_settings(
        SETTINGS, [changed_prompt]
    )
    assert prompt_record_settings(SETTINGS, [PROMPT])["llm_model"] == "test-model"


def test_requests_are_asked_in_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_client, "CHUNK_SIZE", 2)
    server = StubServer()

    answers = make_client(tmp_path, server).ask_all(requests_for("car", "pasta", "lamp"))

    assert [answer["passed"] for answer in answers] == [False, True, False]
    assert len(server.bodies) == 3


def test_server_errors_and_rate_limits_are_retried(tmp_path: Path) -> None:
    server = StubServer(statuses_by_word={"pasta": [503, 429]})

    answers = make_client(tmp_path, server).ask_all(requests_for("pasta"))

    assert answers == [{"passed": True, "reason": "stub"}]
    assert len(server.bodies) == 3


def test_truncated_answer_is_asked_again_with_a_larger_limit(tmp_path: Path) -> None:
    server = StubServer(truncated_words=("pasta",), truncated_below=1000)

    answers = make_client(tmp_path, server).ask_all(requests_for("pasta"))

    assert answers == [{"passed": True, "reason": "stub"}]
    assert [body["max_tokens"] for body in server.bodies] == [512, 1024]


def test_invalid_answer_becomes_none_and_is_asked_again_next_time(tmp_path: Path) -> None:
    server = StubServer(truncated_words=("lamp",), statuses_by_word={"car": [400]})

    answers = make_client(tmp_path, server).ask_all(requests_for("pasta", "lamp", "car"))

    assert answers[0] == {"passed": True, "reason": "stub"}
    assert answers[1:] == [None, None]
    number_of_requests = len(server.bodies)
    make_client(tmp_path, server).ask_all(requests_for("pasta", "lamp", "car"))
    assert len(server.bodies) == number_of_requests + 3  # lamp は2回（上限を2倍にして）、car は1回


def test_too_many_invalid_answers_stop_the_command(tmp_path: Path) -> None:
    words = [f"word{number}" for number in range(12)]
    server = StubServer(statuses_by_word={word: [404] for word in words})

    with pytest.raises(CommandStopped) as stopped:
        make_client(tmp_path, server).ask_all(requests_for(*words))

    assert "llm.model" in stopped.value.message
    assert "stub error" in stopped.value.message


def test_unreachable_server_stops_with_the_command_to_rerun(tmp_path: Path) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = LlmClient(
        SETTINGS,
        tmp_path / "answers.sqlite3",
        rerun_from="passages judge-word-sense",
        transport=httpx.MockTransport(refuse),
    )

    with pytest.raises(CommandStopped) as stopped:
        client.ask_all(requests_for("pasta"))

    assert "接続できません" in stopped.value.message
    assert stopped.value.rerun_from == "passages judge-word-sense"


def test_answers_received_before_a_stop_are_kept(tmp_path: Path) -> None:
    """1つの問い合わせでサーバが失敗し続けても、ほかの答えは保存され、次は問い合わせない。"""
    server = StubServer(statuses_by_word={"car": [503] * 100})

    with pytest.raises(CommandStopped):
        make_client(tmp_path, server).ask_all(requests_for("pasta", "car", "lamp"))

    server.statuses_by_word = {}
    number_of_requests = len(server.bodies)
    make_client(tmp_path, server).ask_all(requests_for("pasta", "car", "lamp"))
    assert len(server.bodies) == number_of_requests + 1


def test_prompt_template_fills_values_and_loads_from_a_directory(tmp_path: Path) -> None:
    (tmp_path / "rate-similarity.md").write_text("Rate $first and $second.", encoding="utf-8")

    prompt = PromptTemplate.load("rate-similarity", tmp_path)

    assert prompt.render({"first": "car", "second": "truck"}) == "Rate car and truck."
    with pytest.raises(KeyError):
        prompt.render({"first": "car"})


def test_lists_in_answers_are_capped_by_the_schema() -> None:
    schema = {
        "type": "object",
        "properties": {"nouns": {"type": "array", "items": {"type": "string"}}},
        "required": ["nouns"],
    }

    capped = with_maximum_items(schema, "nouns", 25)

    assert capped["properties"]["nouns"] == {
        "type": "array",
        "items": {"type": "string"},
        "maxItems": 25,
    }
    # 元の形は変えない
    assert "maxItems" not in schema["properties"]["nouns"]


def test_a_client_waits_while_another_command_writes_to_the_database(tmp_path: Path) -> None:
    # ほかのコマンドが書き込みの途中でも、"database is locked" で止まらず、待ってから書く
    server = StubServer()
    make_client(tmp_path, server).ask_all(requests_for("car"))
    holder = sqlite3.connect(tmp_path / "answers.sqlite3", check_same_thread=False)
    holder.execute("BEGIN IMMEDIATE")
    holder.execute("DELETE FROM answers WHERE prompt_name = 'nothing'")
    # sqlite の既定の待ち時間（5 秒）より長く書き込みを続ける
    releaser = threading.Timer(6.0, holder.commit)
    releaser.start()

    answers = make_client(tmp_path, server).ask_all(requests_for("pasta"))

    releaser.join()
    holder.close()
    assert answers[0]["passed"] is True


def test_the_write_lock_is_not_held_while_waiting_for_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 答えを待っている間は、ほかのコマンドがすぐに書き込みの鍵を取れる。2件ごとに書くので、
    # lamp を待つ間には、pasta と bread の答えを書き終えている（書いた後に鍵を離しているか）
    monkeypatch.setattr(llm_client, "COMMIT_EVERY", 2)
    path = tmp_path / "answers.sqlite3"
    make_client(tmp_path, StubServer()).ask_all(requests_for("car"))
    could_write = []
    server = StubServer()

    async def handle(request: httpx.Request) -> httpx.Response:
        # 本当に待って、前の答えを受け取る（書く）番を回す
        await REAL_SLEEP(0.05)
        if "lamp" in json.loads(request.content)["messages"][0]["content"]:
            other = sqlite3.connect(path, timeout=0)
            try:
                other.execute("BEGIN IMMEDIATE")
                other.rollback()
                could_write.append(True)
            except sqlite3.OperationalError:
                could_write.append(False)
            finally:
                other.close()
        return server.handle(request)

    settings = SETTINGS.model_copy(update={"maximum_concurrent_requests": 1})
    client = LlmClient(settings, path, rerun_from="test", transport=httpx.MockTransport(handle))
    client.ask_all(requests_for("pasta", "bread", "rice", "lamp"))

    assert could_write == [True]
    saved = sqlite3.connect(path).execute("SELECT COUNT(*) FROM answers").fetchone()[0]
    assert saved == 5
