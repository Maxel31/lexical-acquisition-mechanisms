"""未知語が既知語になるフェーズで変わった重みを部品ごとに移し、未知語の 3 つの指標を担う部品を探す
（試行の作業。34|）。

学習前の重みを θ1（OLMo 2 の、公開されている stage 1 の最後の重み）、未知語の獲得の最後の重みを θ2、
差を Δ = θ2 − θ1 とする。部品の組 S ごとに、θ1 + Δ_S（足す）と θ2 − Δ_S（戻す）のモデルで、未知語の
3 つの指標（形への慣れ・意味の獲得・語としての統合）を、16| と同じ測り方で層ごとに測る。部品の組は
33| と同じ（measure_second_sense_weight_grafting.component_sets）。
- measure <run>（GPU 1 枚）: outputs/partial-results/unknown-word-weight-grafting/<run>.parquet
  （部品の組・移し方・語・指標・層ごとの値。露出 1000 回以上の未知語 36 個と、意味の相手の名詞）
- summarize（CPU）: 露出 1000 回以上の未知語で、指標ごと（意味と統合は、17| で順番の判定に使った層）
  に、部品の組ごとの値の平均と、θ1 から θ2 までの変化のうち戻した割合（recovered）を、未知語の
  トークン数ごとに summary.txt に書く。
環境変数 FIRST_PHASE_EXPERIMENT に unknown-word-acquisition-replication を渡すと、seed を変えて
学習し直した重み（build_unknown_word_replication.py）を読み、結果を同じディレクトリの下の
unknown-word-acquisition-replication/ に書く。
"""

import os
import sys
from pathlib import Path

import pandas
import pyarrow as arrow
import pyarrow.parquet as parquet
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_second_sense_weight_grafting import component, component_sets  # noqa: E402

from lexical_acquisition_mechanisms import (  # noqa: E402
    neutral_contexts,
    real_word_probes,
    unknown_word_assignments,
    word_finalization,
)
from lexical_acquisition_mechanisms.csv_tables import read_rows  # noqa: E402
from lexical_acquisition_mechanisms.language_model import LanguageModel  # noqa: E402
from lexical_acquisition_mechanisms.neutral_contexts import NeutralContext  # noqa: E402
from lexical_acquisition_mechanisms.settings import load_settings  # noqa: E402
from lexical_acquisition_mechanisms.unknown_word_assignments import (  # noqa: E402
    UnknownWordAssignment,
)
from lexical_acquisition_mechanisms.unknown_word_indicators import (  # noqa: E402
    _measure_run,
    read_real_word_probes,
)
from lexical_acquisition_mechanisms.word_selection import SelectedWord  # noqa: E402
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer  # noqa: E402

MODEL = "OLMo-2-0425-1B"
EXPERIMENT = os.environ.get("FIRST_PHASE_EXPERIMENT", unknown_word_assignments.EXPERIMENT)
DIRECTORY = Path("outputs/partial-results/unknown-word-weight-grafting")
RESULTS = DIRECTORY if EXPERIMENT == unknown_word_assignments.EXPERIMENT else DIRECTORY / EXPERIMENT
# 17| で順番の判定に使った層（形への慣れは層によらない）
LAYERS = {"form": -1, "meaning": 12, "integration": 16}
MINIMUM_EXPOSURE = 1000
BATCH_SIZE = 32


def measure(run: str) -> None:
    assignments = [
        row
        for row in read_rows(unknown_word_assignments.ASSIGNMENTS_PATH, UnknownWordAssignment)
        if row.run == run and row.exposure >= MINIMUM_EXPOSURE
    ]
    # 露出 MINIMUM_EXPOSURE 回以上の未知語と、意味の相手の名詞だけを測る
    measured = {row.unknown_word for row in assignments} | {row.noun for row in assignments}
    words = {
        word.word: word
        for word in read_rows(word_finalization.OUTPUT_PATH, SelectedWord)
        if word.word in measured and word.role in ("unknown", "noun")
    }
    contexts = [row.context for row in read_rows(neutral_contexts.CONTEXTS_PATH, NeutralContext)]
    probes = read_real_word_probes(real_word_probes.probe_paths(MODEL)[0])
    before = load_settings(Path("settings.yaml")).models[MODEL]
    after = Path("outputs/checkpoints") / EXPERIMENT / MODEL / run / "final"
    network = AutoModelForCausalLM.from_pretrained(
        before.repository, revision=before.revision, dtype=torch.float32
    ).to("cuda")
    model = LanguageModel(network, WordTokenizer.from_model(before), BATCH_SIZE)
    parameters = dict(network.named_parameters())
    first = {name: value.detach().clone() for name, value in parameters.items()}
    second = load_file(str(after / "model.safetensors"), device="cuda")
    delta = {name: second[name].float() - first[name] for name in first}
    del second
    owner = {name: component(name) for name in first}
    configurations = [("before", "baseline", set()), ("after", "baseline", None)]
    for name, members in component_sets(network.config.num_hidden_layers).items():
        configurations += [(name, "sufficiency", members), (name, "necessity", members)]
    rows = []
    for name, mode, members in configurations:
        with torch.no_grad():
            for parameter_name, parameter in parameters.items():
                moved = members is None or owner[parameter_name] in members
                if mode == "necessity":
                    moved = not moved
                parameter.copy_(first[parameter_name] + (delta[parameter_name] if moved else 0))
        values = _measure_run(model, run, assignments, words, [], contexts, probes)
        rows += [row | {"configuration": name, "mode": mode} for row in values]
        print(f"{run} {mode} {name}", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    parquet.write_table(arrow.Table.from_pylist(rows), RESULTS / f"{run}.parquet")


def summarize() -> None:
    table = pandas.concat(
        parquet.read_table(path).to_pandas() for path in sorted(RESULTS.glob("run-*.parquet"))
    )
    table = table[(table["kind"] == "unknown") & (table["exposure"] >= MINIMUM_EXPOSURE)]
    table = table[table["layer"] == table["indicator"].map(LAYERS)]
    # 部品の組を、測った順（before・after・部品の組）に並べる
    order = list(dict.fromkeys(zip(table["configuration"], table["mode"], strict=True)))
    by_tokens = table.groupby(["configuration", "mode", "indicator", "token_count"])["value"]
    by_tokens = by_tokens.mean().unstack(["indicator", "token_count"]).reindex(order)
    overall = table.groupby(["configuration", "mode", "indicator"])["value"].mean()
    overall = overall.unstack("indicator").reindex(order)
    text = (
        f"## 露出 {MINIMUM_EXPOSURE} 回以上の未知語での値（form は驚き度、meaning は層 "
        f"{LAYERS['meaning']}、integration は層 {LAYERS['integration']}）\n"
        f"未知語の数（run を通して）: "
        f"{table.drop_duplicates(['run', 'word']).groupby('token_count').size().to_dict()}\n"
        + overall.round(3).to_string()
        + "\n"
    )
    for title, frame in (("全ての未知語", overall), ("未知語のトークン数ごと", by_tokens)):
        start, end = frame.loc[("before", "baseline")], frame.loc[("after", "baseline")]
        text += (
            f"\n## θ1 から θ2 までの変化のうち、戻した割合（recovered。{title}）\n"
            + ((frame - start) / (end - start)).round(2).to_string()
            + "\n"
        )
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    if sys.argv[1] == "measure":
        measure(sys.argv[2])
    else:
        summarize()
