"""Pile（重複を除いた版。Hugging Face の EleutherAI/the_pile_deduplicated）の読み込み。

Pile は手元に保存しない。parquet の断片（1,650個、1つ約270MB）を1つずつダウンロードして読み、
読み終えたら消す。データの版は固定し、文書は（断片の番号, 断片の中の行の番号）で指す。
断片は、版を固定した URL から流しながら書く（huggingface_hub の取得では、接続が途中で止まったまま
進まないことがあった）。DOWNLOAD_TIMEOUT 秒のあいだ何も届かなければ、届いた所から取り直す。
"""

import multiprocessing
import shutil
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import httpx
import pyarrow.parquet as parquet
from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url

PILE_REPOSITORY = "EleutherAI/the_pile_deduplicated"
PILE_REVISION = "fcbfcfde4222cbb1acd1d33bad0be250ee14b1bb"
NUMBER_OF_SHARDS = 1650
DOWNLOAD_TIMEOUT = 30
DOWNLOAD_ATTEMPTS = 20
# 取り直す前に待つ秒数（取り直すたびに2倍にし、DOWNLOAD_MAXIMUM_WAIT 秒まで）
DOWNLOAD_RETRY_WAIT = 2
DOWNLOAD_MAXIMUM_WAIT = 60
# 一時的な誤りとして取り直す応答（多すぎる問い合わせと、サーバの一時的な誤り）
RETRIED_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
DOWNLOAD_DIRECTORY = Path("data/pile_download")
# 届く速さがこれより遅い接続は、切って取り直す（同じ所から、1秒に数十 KB しか届かない接続がある）。
# SLOW_DOWNLOAD_SECONDS 秒ごとに、その間に届いたバイト数で判断する
MINIMUM_BYTES_PER_SECOND = 512 * 1024
SLOW_DOWNLOAD_SECONDS = 60


class _SlowDownload(Exception):
    """接続が遅すぎるので、切って取り直す。"""


@dataclass(frozen=True)
class PileDocument:
    shard: int
    row: int
    text: str


def shard_file_names() -> list[str]:
    """断片のファイル名を、断片の番号の順に返す。"""
    files = HfApi().list_repo_files(PILE_REPOSITORY, repo_type="dataset", revision=PILE_REVISION)
    names = sorted(name for name in files if name.startswith("data/train-"))
    if len(names) != NUMBER_OF_SHARDS:
        raise RuntimeError(
            f"Pile の断片が {len(names)} 個あります（{NUMBER_OF_SHARDS} 個のはず）。"
        )
    return names


def download_shard(file_name: str, directory: Path, client: httpx.Client | None = None) -> Path:
    """断片を directory にダウンロードし、そのパスを返す。

    止まったり切れたり、一時的な誤りの応答（RETRIED_STATUS_CODES）が返ったり、届く速さが
    MINIMUM_BYTES_PER_SECOND より遅かったりしたら、少し待ってから、届いた所から続きを取り直す
    （DOWNLOAD_ATTEMPTS 回まで）。大きさが Hugging Face の記録と合わな
    ければ、誤りにする。
    """
    url = hf_hub_url(PILE_REPOSITORY, file_name, repo_type="dataset", revision=PILE_REVISION)
    # 大きさを調べる問い合わせも、一時的な誤り（429・5xx・接続の誤り）なら取り直す
    expected = get_hf_file_metadata(url, retry_on_errors=True).size
    path = directory / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(path.name + ".partial")
    partial_path.unlink(missing_ok=True)
    owned = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=DOWNLOAD_TIMEOUT)
    try:
        for attempt in range(DOWNLOAD_ATTEMPTS):
            received = partial_path.stat().st_size if partial_path.exists() else 0
            if received == expected:
                break
            if attempt:
                time.sleep(min(DOWNLOAD_RETRY_WAIT * 2 ** (attempt - 1), DOWNLOAD_MAXIMUM_WAIT))
            headers = {"Range": f"bytes={received}-"} if received else {}
            try:
                with client.stream("GET", url, headers=headers) as response:
                    response.raise_for_status()
                    # 続きを返さないサーバなら、初めから書き直す
                    mode = "ab" if received and response.status_code == 206 else "wb"
                    with partial_path.open(mode) as file:
                        window_start = time.monotonic()
                        window_bytes = 0
                        for chunk in response.iter_bytes():
                            file.write(chunk)
                            window_bytes += len(chunk)
                            elapsed = time.monotonic() - window_start
                            if elapsed >= SLOW_DOWNLOAD_SECONDS:
                                if window_bytes < MINIMUM_BYTES_PER_SECOND * elapsed:
                                    raise _SlowDownload
                                window_start, window_bytes = time.monotonic(), 0
            except (httpx.TransportError, _SlowDownload):
                continue
            except httpx.HTTPStatusError as error:
                if error.response.status_code not in RETRIED_STATUS_CODES:
                    raise
    finally:
        if owned:
            client.close()
    received = partial_path.stat().st_size if partial_path.exists() else 0
    if received != expected:
        raise RuntimeError(
            f"Pile の断片 {file_name} を落とせませんでした（{received} / {expected} バイト）。"
        )
    partial_path.replace(path)
    return path


def documents_in_shard(
    shard: int,
    file_name: str,
    download_directory: Path = DOWNLOAD_DIRECTORY,
    download: Callable[[str, Path], Path] = download_shard,
) -> Iterator[PileDocument]:
    """断片をダウンロードして、文書を行の順に返す。読み終えたら（途中で止めても）消す。"""
    directory = download_directory / f"shard-{shard:05d}"
    try:
        path = download(file_name, directory)
        row = 0
        for batch in parquet.ParquetFile(path).iter_batches(columns=["text"]):
            for text in batch.column(0).to_pylist():
                yield PileDocument(shard, row, text)
                row += 1
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def scan_shards[Result](
    shards: Sequence[int],
    scan: Callable[[Iterator[PileDocument]], Result],
    number_of_processes: int,
    download_directory: Path = DOWNLOAD_DIRECTORY,
    download: Callable[[str, Path], Path] = download_shard,
) -> Iterator[Result]:
    """断片ごとに scan（文書を受け取って結果を返す関数）を並列に実行し、断片の順に結果を返す。

    別のプロセスは spawn で起動する（スレッドのあるプロセスを fork すると、止まることがあるため）。
    そのため、scan と download は、モジュールの最上位で定義した関数にする。
    """
    file_names = shard_file_names()
    work = partial(
        _scan_one_shard,
        scan=scan,
        file_names=file_names,
        download_directory=download_directory,
        download=download,
    )
    if number_of_processes == 1:
        yield from map(work, shards)
        return
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=number_of_processes, mp_context=context) as executor:
        yield from executor.map(work, shards)


def _scan_one_shard[Result](
    shard: int,
    scan: Callable[[Iterator[PileDocument]], Result],
    file_names: Sequence[str],
    download_directory: Path,
    download: Callable[[str, Path], Path],
) -> Result:
    documents = documents_in_shard(shard, file_names[shard], download_directory, download)
    return scan(documents)
