"""Pile（重複を除いた版。Hugging Face の EleutherAI/the_pile_deduplicated）の読み込み。

Pile は手元に保存しない。parquet の断片（1,650個、1つ約270MB）を1つずつダウンロードして読み、
読み終えたら消す。データの版は固定し、文書は（断片の番号, 断片の中の行の番号）で指す。
"""

import multiprocessing
import shutil
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pyarrow.parquet as parquet
from huggingface_hub import HfApi, hf_hub_download

PILE_REPOSITORY = "EleutherAI/the_pile_deduplicated"
PILE_REVISION = "fcbfcfde4222cbb1acd1d33bad0be250ee14b1bb"
NUMBER_OF_SHARDS = 1650
DOWNLOAD_DIRECTORY = Path("data/pile_download")


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


def download_shard(file_name: str, directory: Path) -> Path:
    """断片を directory にダウンロードし、そのパスを返す。"""
    return Path(
        hf_hub_download(
            PILE_REPOSITORY,
            file_name,
            repo_type="dataset",
            revision=PILE_REVISION,
            local_dir=directory,
        )
    )


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
