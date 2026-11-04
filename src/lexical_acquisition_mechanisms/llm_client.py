"""LLM（vLLM のサーバ）への問い合わせ。

- 指示文は llm_prompts/<名前>.md に置き、中身のハッシュを版として記録する。
- 答えは JSON にし、形（JSON Schema）をサーバに渡して、その形でだけ答えさせる。温度は 0。
- 答えは SQLite のファイルに保存し、同じ問い合わせ（モデル・指示文・形が同じ）には保存した答えを返す
  （同じ入力には同じ結果を返し、途中で止まっても、答えを得た分は問い合わせ直さない）。
"""

import asyncio
import hashlib
import json
import sqlite3
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.settings import LlmSettings

PROMPT_DIRECTORY = Path("llm_prompts")
ANSWERS_PATH = Path("data/llm_answers.sqlite3")
NUMBER_OF_ATTEMPTS = 5
CHUNK_SIZE = 10_000


@dataclass(frozen=True)
class PromptTemplate:
    """指示文。$word のような箇所に、問い合わせごとの値を入れる。"""

    name: str
    text: str

    @classmethod
    def load(cls, name: str, directory: Path = PROMPT_DIRECTORY) -> "PromptTemplate":
        return cls(name, (directory / f"{name}.md").read_text(encoding="utf-8"))

    @property
    def version(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:12]

    def render(self, values: Mapping[str, str]) -> str:
        return string.Template(self.text).substitute(values)


@dataclass(frozen=True)
class LlmRequest:
    prompt: PromptTemplate
    values: Mapping[str, str]
    # 答えの形（JSON Schema）
    answer_schema: Mapping[str, Any]
    maximum_tokens: int = 512

    def body(self, model: str) -> dict[str, Any]:
        return {
            "model": model,
            "messages": [{"role": "user", "content": self.prompt.render(self.values)}],
            "temperature": 0,
            "seed": 0,
            "max_tokens": self.maximum_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": self.answer_schema, "strict": True},
            },
        }


class LlmClient:
    def __init__(
        self,
        settings: LlmSettings,
        answers_path: Path,
        rerun_from: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._answers_path = answers_path
        self._rerun_from = rerun_from
        self._transport = transport

    def ask_all(self, requests: Sequence[LlmRequest]) -> list[dict[str, Any]]:
        """問い合わせごとの答え（JSON を読んだもの）を、問い合わせの順に返す。

        メモリに収まるように、CHUNK_SIZE 件ずつ問い合わせる。
        """
        self._answers_path.parent.mkdir(parents=True, exist_ok=True)
        all_answers: list[dict[str, Any]] = []
        with sqlite3.connect(self._answers_path) as database:
            database.execute(
                "CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, prompt_name TEXT,"
                " prompt_version TEXT, body TEXT, answer TEXT, created_at TEXT)"
            )
            for start in range(0, len(requests), CHUNK_SIZE):
                chunk = requests[start : start + CHUNK_SIZE]
                bodies = [request.body(self._settings.model) for request in chunk]
                keys = [_key_of(body) for body in bodies]
                answers = _saved_answers(database, keys)
                missing = {key: index for index, key in enumerate(keys) if key not in answers}
                if missing:
                    print(
                        f"LLM に {len(missing)} 件を問い合わせます（{start + len(chunk)} / "
                        f"{len(requests)} 件まで。保存済み {len(answers)} 件）。"
                    )
                    asyncio.run(self._ask_missing(database, chunk, bodies, missing, answers))
                all_answers.extend(answers[key] for key in keys)
        return all_answers

    async def _ask_missing(
        self,
        database: sqlite3.Connection,
        requests: Sequence[LlmRequest],
        bodies: Sequence[dict[str, Any]],
        missing: Mapping[str, int],
        answers: dict[str, dict[str, Any]],
    ) -> None:
        limit = asyncio.Semaphore(self._settings.maximum_concurrent_requests)
        async with httpx.AsyncClient(
            base_url=self._settings.server_url, transport=self._transport, timeout=600
        ) as client:

            async def ask(key: str, index: int) -> tuple[str, int, dict[str, Any]]:
                async with limit:
                    return key, index, await self._post(client, bodies[index])

            tasks = [ask(key, index) for key, index in missing.items()]
            try:
                for number, finished in enumerate(asyncio.as_completed(tasks), start=1):
                    key, index, answer = await finished
                    answers[key] = answer
                    request = requests[index]
                    database.execute(
                        "INSERT OR REPLACE INTO answers VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            key,
                            request.prompt.name,
                            request.prompt.version,
                            json.dumps(bodies[index], ensure_ascii=False),
                            json.dumps(answer, ensure_ascii=False),
                            datetime.now().astimezone().isoformat(timespec="seconds"),
                        ),
                    )
                    if number % 1000 == 0:
                        database.commit()
            finally:
                # 途中で止まっても、答えを得た分は残す
                database.commit()

    async def _post(self, client: httpx.AsyncClient, body: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(NUMBER_OF_ATTEMPTS):
            try:
                response = await client.post("/chat/completions", json=body)
            except httpx.TransportError as error:
                problem = f"LLM のサーバ {self._settings.server_url} に接続できません（{error}）"
            else:
                if response.status_code < 500:
                    response.raise_for_status()
                    content = response.json()["choices"][0]["message"]["content"]
                    return json.loads(content)
                problem = f"LLM のサーバがエラーを返しました（{response.status_code}）"
            await asyncio.sleep(2**attempt)
        raise CommandStopped(
            f"{problem}。README のとおりにサーバを起動してから、実行し直してください。",
            rerun_from=self._rerun_from,
        )


def _key_of(body: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


def _saved_answers(database: sqlite3.Connection, keys: Sequence[str]) -> dict[str, dict]:
    answers = {}
    unique_keys = list(dict.fromkeys(keys))
    for start in range(0, len(unique_keys), 500):
        chunk = unique_keys[start : start + 500]
        placeholders = ",".join("?" * len(chunk))
        rows = database.execute(
            f"SELECT key, answer FROM answers WHERE key IN ({placeholders})", chunk
        )
        answers.update({key: json.loads(answer) for key, answer in rows})
    return answers
