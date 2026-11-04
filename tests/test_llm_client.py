"""LLM への問い合わせのテスト。サーバの代わりに、決まった答えを返す部品を使う。"""

import json
from pathlib import Path

import httpx
import pytest

from lexical_acquisition_mechanisms import llm_client
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.llm_client import LlmClient, LlmRequest, PromptTemplate
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


class StubServer:
    """語が pasta なら合格と答える。受けた問い合わせを残す。"""

    def __init__(self, failures_before_success: int = 0) -> None:
        self.bodies: list[dict] = []
        self.failures_before_success = failures_before_success

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        if self.failures_before_success > 0:
            self.failures_before_success -= 1
            return httpx.Response(503)
        content = body["messages"][0]["content"]
        answer = {"passed": "pasta" in content, "reason": "stub"}
        message = {"role": "assistant", "content": json.dumps(answer)}
        return httpx.Response(200, json={"choices": [{"message": message}]})


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
    server = StubServer()

    answers = make_client(tmp_path, server).ask_all(requests_for("car", "pasta", "lamp"))

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


def test_changed_prompt_is_asked_again(tmp_path: Path) -> None:
    server = StubServer()
    make_client(tmp_path, server).ask_all(requests_for("pasta"))
    changed_prompt = PromptTemplate(PROMPT.name, PROMPT.text + "Answer in JSON.\n")

    request = LlmRequest(changed_prompt, {"word": "pasta", "passage": "x"}, ANSWER_SCHEMA)
    make_client(tmp_path, server).ask_all([request])

    assert len(server.bodies) == 2
    assert changed_prompt.version != PROMPT.version


def test_requests_are_asked_in_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_client, "CHUNK_SIZE", 2)
    server = StubServer()

    answers = make_client(tmp_path, server).ask_all(requests_for("car", "pasta", "lamp"))

    assert [answer["passed"] for answer in answers] == [False, True, False]
    assert len(server.bodies) == 3


def test_server_errors_are_retried(tmp_path: Path) -> None:
    server = StubServer(failures_before_success=2)

    answers = make_client(tmp_path, server).ask_all(requests_for("pasta"))

    assert answers == [{"passed": True, "reason": "stub"}]
    assert len(server.bodies) == 3


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
    """途中でサーバが止まっても、それまでに得た答えは保存され、次は問い合わせない。"""
    server = StubServer()
    make_client(tmp_path, server).ask_all(requests_for("pasta"))
    server.failures_before_success = 100

    with pytest.raises(CommandStopped):
        make_client(tmp_path, server).ask_all(requests_for("pasta", "car"))

    server.failures_before_success = 0
    number_of_requests = len(server.bodies)
    make_client(tmp_path, server).ask_all(requests_for("pasta", "car"))
    assert len(server.bodies) == number_of_requests + 1


def test_prompt_template_fills_values_and_loads_from_a_directory(tmp_path: Path) -> None:
    (tmp_path / "rate-similarity.md").write_text("Rate $first and $second.", encoding="utf-8")

    prompt = PromptTemplate.load("rate-similarity", tmp_path)

    assert prompt.render({"first": "car", "second": "truck"}) == "Rate car and truck."
    with pytest.raises(KeyError):
        prompt.render({"first": "car"})
