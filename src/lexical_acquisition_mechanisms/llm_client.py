"""LLM（vLLM のサーバ）への問い合わせ。

- 指示文は llm_prompts/<名前>.md に置き、中身のハッシュを版として記録する。
- 答えは JSON にし、形（JSON Schema）をサーバに渡して、その形でだけ答えさせる。温度は 0。
- 答えは SQLite のファイルに保存し、同じ問い合わせ（モデル・指示文・入れた値・形・長さの上限が同じ）
  には保存した答えを返す（同じ入力には同じ結果を返し、途中で止まっても、得た答えは問い合わせ直さない）。
- 形に合う答えが得られなかった問い合わせ（長さの上限で切れた、サーバが 400 番台を返したなど）は、
  答えを None にして先へ進む（保存はしないので、次に実行したときに問い合わせ直す）。
  多すぎるときは止まる。
- LLM を使う処理は、prompt_record_settings の値を output_records.Step の設定に入れる
  （指示文を直したら、処理がやり直されるように）。
"""

import asyncio
import contextlib
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
# 答えのデータベースを、ほかのコマンドが書いている間に待つ秒数と、書き込みを区切る答えの数
# （2つの LLM のコマンドを同時に実行しても、"database is locked" で止まらないように）
DATABASE_TIMEOUT = 600
COMMIT_EVERY = 100
# 形に合う答えが得られなかった問い合わせが、1回に問い合わせた数のこの割合と10件の
# 両方を超えたら止まる
MAXIMUM_INVALID_FRACTION = 0.01


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


def prompt_record_settings(
    settings: LlmSettings, prompts: Sequence[PromptTemplate]
) -> dict[str, Any]:
    """処理の記録に入れる、LLM のモデルと指示文の版。

    サーバの URL や同時に送る数は、結果に効かないので入れない。
    """
    return {
        "llm_model": settings.model,
        "prompt_versions": {prompt.name: prompt.version for prompt in prompts},
    }


@dataclass(frozen=True)
class LlmRequest:
    prompt: PromptTemplate
    values: Mapping[str, str]
    # 答えの形（JSON Schema）
    answer_schema: Mapping[str, Any]
    maximum_tokens: int = 512

    def body(self, model: str, maximum_tokens: int | None = None) -> dict[str, Any]:
        return {
            "model": model,
            "messages": [{"role": "user", "content": self.prompt.render(self.values)}],
            "temperature": 0,
            "seed": 0,
            "max_tokens": maximum_tokens or self.maximum_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": self.answer_schema, "strict": True},
            },
        }


def with_maximum_items(schema: Mapping[str, Any], key: str, number: int) -> dict[str, Any]:
    """答えの形の、一覧の項目 key の長さの上限を number にしたもの（長すぎる答えを形で止める）。"""
    properties = dict(schema["properties"])
    properties[key] = {**properties[key], "maxItems": number}
    return {**schema, "properties": properties}


def require_answers(answers: Sequence[dict[str, Any] | None], what: str, rerun_from: str) -> None:
    """全ての問い合わせに答えがあることを確かめる（なければ止まる）。

    一覧を挙げさせる問い合わせのように、1件の答えがないだけで結果が大きく欠けるときに使う。
    得られなかった答えは保存しないので、実行し直すと、その分だけ問い合わせ直す。
    """
    missing = sum(answer is None for answer in answers)
    if missing:
        raise CommandStopped(
            f"LLM から、{what}の答えが {missing} 件得られませんでした。"
            "実行し直すと、得られなかった分だけ問い合わせ直します。",
            rerun_from=rerun_from,
        )


class InvalidAnswer(Exception):
    """形に合う答えが得られなかった（問い合わせ直しても、同じ結果になる見込みが高い）。"""


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

    def ask_all(self, requests: Sequence[LlmRequest]) -> list[dict[str, Any] | None]:
        """問い合わせごとの答え（JSON を読んだもの。得られなければ None）を、問い合わせの順に返す。

        問い合わせの本文を一度に作らないように、CHUNK_SIZE 件ずつ問い合わせる。
        """
        self._answers_path.parent.mkdir(parents=True, exist_ok=True)
        all_answers: list[dict[str, Any] | None] = []
        with contextlib.closing(
            sqlite3.connect(self._answers_path, timeout=DATABASE_TIMEOUT)
        ) as database:
            # ほかのコマンドが同じファイルに書いている間も読み書きできるように
            database.execute("PRAGMA journal_mode=WAL")
            database.execute(
                "CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, prompt_name TEXT,"
                " prompt_version TEXT, body TEXT, answer TEXT, created_at TEXT)"
            )
            for start in range(0, len(requests), CHUNK_SIZE):
                chunk = requests[start : start + CHUNK_SIZE]
                bodies = [request.body(self._settings.model) for request in chunk]
                keys = [_key_of(body) for body in bodies]
                answers: dict[str, dict[str, Any] | None] = _saved_answers(database, keys)
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
        answers: dict[str, dict[str, Any] | None],
    ) -> None:
        limit = asyncio.Semaphore(self._settings.maximum_concurrent_requests)
        connections = httpx.Limits(
            max_connections=self._settings.maximum_concurrent_requests,
            max_keepalive_connections=self._settings.maximum_concurrent_requests,
        )
        problems: list[str] = []
        stop: CommandStopped | None = None
        async with httpx.AsyncClient(
            base_url=self._settings.server_url,
            transport=self._transport,
            timeout=600,
            limits=connections,
        ) as client:

            async def ask(key: str, index: int) -> tuple[str, int, dict[str, Any] | str]:
                async with limit:
                    try:
                        return key, index, await self._answer(client, requests[index])
                    except InvalidAnswer as invalid:
                        return key, index, str(invalid)

            tasks = [asyncio.create_task(ask(key, index)) for key, index in missing.items()]
            try:
                for number, finished in enumerate(asyncio.as_completed(tasks), start=1):
                    try:
                        key, index, answer = await finished
                    except CommandStopped as stopped:
                        # ほかの問い合わせの答えは受け取ってから止まる
                        stop = stop or stopped
                        continue
                    if isinstance(answer, str):
                        answers[key] = None
                        problems.append(f"{requests[index].prompt.name}: {answer}")
                        continue
                    answers[key] = answer
                    self._save(database, key, requests[index], bodies[index], answer)
                    if number % COMMIT_EVERY == 0:
                        database.commit()
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                database.commit()
        if stop is not None:
            raise stop
        if len(problems) > max(10, MAXIMUM_INVALID_FRACTION * len(missing)):
            raise CommandStopped(
                f"LLM から形に合う答えが得られない問い合わせが {len(problems)} 件ありました"
                f"（例: {problems[0]}）。"
                "モデルの名前（llm.model）・指示文・長さの上限を確かめてください。",
                rerun_from=self._rerun_from,
            )
        if problems:
            print(f"LLM から形に合う答えが得られなかった問い合わせ: {len(problems)} 件")

    async def _answer(self, client: httpx.AsyncClient, request: LlmRequest) -> dict[str, Any]:
        """形に合う答えを返す。長さの上限で切れたときは、上限を2倍にして1回だけ問い合わせ直す。"""
        for maximum_tokens in (request.maximum_tokens, 2 * request.maximum_tokens):
            choice = await self._post(client, request.body(self._settings.model, maximum_tokens))
            if choice.get("finish_reason") != "length":
                break
        if choice.get("finish_reason") != "stop":
            raise InvalidAnswer(f"答えが終わっていません（{choice.get('finish_reason')}）")
        try:
            answer = json.loads(choice["message"]["content"])
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise InvalidAnswer(f"答えを JSON として読めません（{error}）") from error
        if not isinstance(answer, dict):
            raise InvalidAnswer(f"答えが JSON のオブジェクトではありません（{answer!r}）")
        missing_keys = [
            key for key in request.answer_schema.get("required", []) if key not in answer
        ]
        if missing_keys:
            raise InvalidAnswer(f"答えに必要な項目がありません（{missing_keys}）")
        return answer

    async def _post(self, client: httpx.AsyncClient, body: dict[str, Any]) -> dict[str, Any]:
        """問い合わせを送り、答えの選択肢を返す。待てば直るかもしれない失敗は、待って試し直す。"""
        for attempt in range(NUMBER_OF_ATTEMPTS):
            if attempt > 0:
                await asyncio.sleep(2 ** (attempt - 1))
            try:
                response = await client.post("/chat/completions", json=body)
            except httpx.TransportError as error:
                problem = f"LLM のサーバ {self._settings.server_url} に接続できません（{error}）"
                continue
            if response.status_code in (408, 429) or response.status_code >= 500:
                problem = f"LLM のサーバがエラーを返しました（{response.status_code}）"
                continue
            if response.status_code >= 400:
                raise InvalidAnswer(
                    f"サーバが {response.status_code} を返しました: {response.text}"
                )
            return response.json()["choices"][0]
        raise CommandStopped(
            f"{problem}。サーバが動いているかを確かめてから、実行し直してください。",
            rerun_from=self._rerun_from,
        )

    def _save(
        self,
        database: sqlite3.Connection,
        key: str,
        request: LlmRequest,
        body: dict[str, Any],
        answer: dict[str, Any],
    ) -> None:
        database.execute(
            "INSERT OR REPLACE INTO answers VALUES (?, ?, ?, ?, ?, ?)",
            (
                key,
                request.prompt.name,
                request.prompt.version,
                json.dumps(body, ensure_ascii=False),
                json.dumps(answer, ensure_ascii=False),
                datetime.now().astimezone().isoformat(timespec="seconds"),
            ),
        )


def _key_of(body: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


def _saved_answers(database: sqlite3.Connection, keys: Sequence[str]) -> dict[str, Any]:
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
