"""ターゲット語と名詞の組ごとに、意味の近さを LLM が評定する（words rate-similarity-with-llm）。

評定は 1（意味が関係ない）〜 7（ほぼ同じ意味）。語の意味は、WordNet の語義の説明で示す。
同じカテゴリの組が高く、別のカテゴリの組が低くなるかは、words measure-hidden-state-similarity が
内部表現の類似度と合わせて確かめる。値は、実験Bの分析の共変量にする。
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from lexical_acquisition_mechanisms import word_selection
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.llm_client import (
    ANSWERS_PATH,
    LlmClient,
    LlmRequest,
    PromptTemplate,
    prompt_record_settings,
)
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.settings import Settings
from lexical_acquisition_mechanisms.word_selection import SelectedWord

COMMAND = "words rate-similarity-with-llm"
OUTPUT_PATH = Path("data/words/similarity_ratings.csv")
PROMPT = "rate-noun-similarity"
SCHEMA = {
    "type": "object",
    "properties": {"similarity": {"type": "integer", "enum": list(range(1, 8))}},
    "required": ["similarity"],
    "additionalProperties": False,
}


@dataclass
class SimilarityRating:
    target: str
    target_category: str
    noun: str
    noun_category: str
    # LLM による意味の近さの評定（1〜7。得られなかったときは 0）
    llm_similarity: int


def rate_similarity_with_llm(settings: Settings, arguments: argparse.Namespace) -> None:
    prompt = PromptTemplate.load(PROMPT)
    step = Step(
        COMMAND,
        inputs={word_selection.OUTPUT_PATH: word_selection.COMMAND},
        outputs=[OUTPUT_PATH],
        settings=prompt_record_settings(settings.llm, [prompt]),
    )
    client = LlmClient(settings.llm, ANSWERS_PATH, rerun_from=COMMAND)
    run_unless_unchanged(
        step,
        lambda: build_similarity_ratings(prompt, client, word_selection.OUTPUT_PATH, OUTPUT_PATH),
        arguments.recompute,
    )


def build_similarity_ratings(
    prompt: PromptTemplate, client: LlmClient, selected_words_path: Path, output_path: Path
) -> None:
    words = read_rows(selected_words_path, SelectedWord)
    targets = [word for word in words if word.role == "target"]
    nouns = [word for word in words if word.role == "noun"]
    pairs = [(target, noun) for target in targets for noun in nouns]
    requests = [
        LlmRequest(
            prompt,
            {
                "target": target.word,
                "target_definition": target.definition,
                "noun": noun.word,
                "noun_definition": noun.definition,
            },
            SCHEMA,
        )
        for target, noun in pairs
    ]
    rows = [
        SimilarityRating(
            target.word,
            target.category,
            noun.word,
            noun.category,
            answer["similarity"] if answer is not None else 0,
        )
        for (target, noun), answer in zip(pairs, client.ask_all(requests), strict=True)
    ]
    write_rows(output_path, SimilarityRating, rows)
