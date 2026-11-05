"""infini-gram（大規模コーパスで、トークン列の出現回数を数える公開 API）への問い合わせ。

未知語の候補が、OLMo 2 の事前学習データと Pile に一度も現れないことを確かめるのに使う。
索引は Llama-2 のトークナイザでトークン化されているので、語の形をこちらで Llama-2 のトークン番号の
列にして送る。小文字・先頭だけ大文字・すべて大文字のそれぞれについて、語頭の形（空白の後）と、
語の途中の形（改行や記号の直後）を数える。数えた結果は保存し、同じ問い合わせはしない。
"""

import asyncio
import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import httpx
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from lexical_acquisition_mechanisms.command_stopped import CommandStopped

API_URL = "https://api.infini-gram.io/"
LLAMA_TOKENIZER = "hf-internal-testing/llama-tokenizer"
LLAMA_TOKENIZER_REVISION = "d02ad6cb9dd2c2296a6332199fa2fdca5938fef0"
# 語の途中の形を作るときに、前に付けて落とすトークン（"▁" と改行）
NEWLINE_PREFIX = [29871, 13]
COUNTS_PATH = Path("data/infini_gram_counts.sqlite3")
# 1つの問い合わせで「または」で数えられる形の数の上限（API の決まり）
FORMS_PER_QUERY = 4
# 制限に当たったときなどに待つ秒数の最初の値（試すたびに2倍にする）。試す回数は、待つ秒数の合計
# （最後の試行の後は待たない。5 + 10 + … + 320 = 635秒）が、制限の待ち（約600秒）を超えるようにする
NUMBER_OF_ATTEMPTS = 8
FIRST_WAIT_SECONDS = 5


def word_forms(word: str, tokenizer: PreTrainedTokenizerBase) -> list[list[int]]:
    """語の形の Llama-2 のトークン列（重なりを除く）。"""
    forms: list[list[int]] = []
    for variant in dict.fromkeys([word.lower(), word.capitalize(), word.upper()]):
        # 先頭の語は、空白の有無によらず語頭の形（"▁" で始まる）になる
        initial = tokenizer(variant, add_special_tokens=False)["input_ids"]
        # 改行の後では語の途中の形になる。先頭の "▁" と改行（2トークン）を除く
        with_newline = tokenizer("\n" + variant, add_special_tokens=False)["input_ids"]
        if with_newline[: len(NEWLINE_PREFIX)] != NEWLINE_PREFIX:
            raise ValueError(f"改行の後の {variant!r} が、想定したトークン列で始まりません。")
        inside = with_newline[len(NEWLINE_PREFIX) :]
        for form in (initial, inside):
            if form not in forms:
                forms.append(form)
    return forms


class InfiniGramClient:
    def __init__(
        self,
        rerun_from: str,
        counts_path: Path = COUNTS_PATH,
        transport: httpx.AsyncBaseTransport | None = None,
        maximum_concurrent_requests: int = 4,
    ) -> None:
        self._rerun_from = rerun_from
        self._counts_path = counts_path
        self._transport = transport
        self._maximum_concurrent_requests = maximum_concurrent_requests
        self._tokenizer = AutoTokenizer.from_pretrained(
            LLAMA_TOKENIZER, revision=LLAMA_TOKENIZER_REVISION
        )

    def occurrences(self, words: Sequence[str], index: str) -> dict[str, int]:
        """各語の、いずれかの形の出現回数の合計（0 なら、索引のコーパスに一度も現れない）。"""
        queries = {word: self._query_groups(word) for word in words}
        self._counts_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._counts_path) as database:
            database.execute(
                "CREATE TABLE IF NOT EXISTS counts (index_name TEXT, query TEXT, count INTEGER,"
                " PRIMARY KEY (index_name, query))"
            )
            counts = dict(
                database.execute(
                    "SELECT query, count FROM counts WHERE index_name = ?", (index,)
                ).fetchall()
            )
            missing = list(
                dict.fromkeys(
                    json.dumps(group)
                    for groups in queries.values()
                    for group in groups
                    if json.dumps(group) not in counts
                )
            )
            if missing:
                print(f"infini-gram（{index}）に {len(missing)} 件を問い合わせます。")
                try:
                    asyncio.run(self._count_missing(database, index, missing, counts))
                finally:
                    database.commit()
        return {
            word: sum(counts[json.dumps(group)] for group in groups)
            for word, groups in queries.items()
        }

    def _query_groups(self, word: str) -> list[list[list[int]]]:
        """語の形を、1つの問い合わせで数えられる数ずつに分ける。"""
        forms = word_forms(word, self._tokenizer)
        return [
            forms[start : start + FORMS_PER_QUERY]
            for start in range(0, len(forms), FORMS_PER_QUERY)
        ]

    async def _count_missing(
        self, database: sqlite3.Connection, index: str, missing: list[str], counts: dict[str, int]
    ) -> None:
        limit = asyncio.Semaphore(self._maximum_concurrent_requests)
        async with httpx.AsyncClient(transport=self._transport, timeout=120) as client:

            async def count(query: str) -> tuple[str, int]:
                async with limit:
                    return query, await self._post(client, index, json.loads(query))

            tasks = [asyncio.create_task(count(query)) for query in missing]
            try:
                for finished in asyncio.as_completed(tasks):
                    query, number = await finished
                    counts[query] = number
                    database.execute(
                        "INSERT OR REPLACE INTO counts VALUES (?, ?, ?)", (index, query, number)
                    )
            finally:
                # 1つが失敗したら、残りを取り消す（結果を受け取らないままにしない）
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    async def _post(self, client: httpx.AsyncClient, index: str, forms: list[list[int]]) -> int:
        # 形の「または」は、節が1つの CNF（[[形1, 形2, …]]）で表す
        body = {"index": index, "query_type": "count", "query_ids": [forms]}
        for attempt in range(NUMBER_OF_ATTEMPTS):
            try:
                response = await client.post(API_URL, json=body)
            except httpx.TransportError as error:
                problem = f"infini-gram に接続できません（{error}）"
            else:
                if response.status_code not in (403, 429) and response.status_code < 500:
                    result = response.json()
                    if "error" in result:
                        raise RuntimeError(f"infini-gram がエラーを返しました: {result['error']}")
                    return int(result["count"])
                problem = f"infini-gram が {response.status_code} を返しました"
            if attempt < NUMBER_OF_ATTEMPTS - 1:
                await asyncio.sleep(FIRST_WAIT_SECONDS * 2**attempt)
        raise CommandStopped(
            f"{problem}。しばらく待ってから実行し直してください。", rerun_from=self._rerun_from
        )
