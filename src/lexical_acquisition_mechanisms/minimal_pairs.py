"""最小対の候補を作る（evaluation-data generate-minimal-pairs）。

最小対は、手がかりの1語だけが違い、ターゲット語の語義が変わる2文。ターゲット語と、語義の追加で足す
語（donor。足す数が最大の run のもの）の組ごとに、LLM に候補を作らせる。
- 名詞を足す組（同カテゴリ・別カテゴリ）: 手がかりは、一方のものにだけ合う語
  （例: She cooked the ___ / She parked the ___）。
- 形容詞を足す組: 手がかりは、空欄を形容詞か名詞に決める語（例: It was so ___ / It was the ___）。
次の候補は使わない。
- 空欄（前が空白で、後に英数字が続かない、ちょうど3本の下線 ___）がちょうど1つでない、または
  2文で語の数が違う（___s のように語の形を変える空欄も使わない）
- 違う語が、ちょうど1つで、空欄より前にあるものでない
- 空欄の外に、donor・ターゲット語・ほかの管理対象の語（別の形も）がある（手がかり以外から語義が
  分からないように）
- 空欄にターゲット語を入れたとき、課題の問題文の中（"Sentence: " の後）で、2文のトークン数が違う
  （head の出力を位置ごとに入れ替えるため）
- 同じ組の、ほかの候補と同じ
使える候補が1つもない語義の組があれば止まる（その語義が、後の全ての分析から黙って消えないように）。
語義の組ごとに、最小対の数（evaluation.minimal_pairs_per_sense_pair）×
evaluation.minimal_pair_candidate_factor 個まで残し、data/evaluation/minimal_pair_candidates.csv に
書く（学習前のモデルで確かめるのは、evaluation-data check-with-pretrained-models）。
"""

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

from lexical_acquisition_mechanisms import sense_addition_assignments, word_finalization
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.inflected_forms import inflected_forms
from lexical_acquisition_mechanisms.llm_client import (
    ANSWERS_PATH,
    LlmClient,
    LlmRequest,
    PromptTemplate,
    prompt_record_settings,
    require_answers,
    with_maximum_items,
)
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.pile_word_usages import WORD_PATTERN
from lexical_acquisition_mechanisms.sense_addition_assignments import SenseAdditionAssignment
from lexical_acquisition_mechanisms.settings import Settings
from lexical_acquisition_mechanisms.word_selection import MANAGED_ROLES, SelectedWord
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

COMMAND = "evaluation-data generate-minimal-pairs"
OUTPUT_PATH = Path("data/evaluation/minimal_pair_candidates.csv")
PROMPT = "generate-minimal-pairs"
BLANK = "___"
# 空欄: 前が空白で、ちょうど3本の下線（文の先頭なら、問題文では "Sentence:" の後の空白の後）。
# 後に英数字が続く空欄（___s など）は、語の形を変えるので使わない
BLANK_PATTERN = re.compile(r"(?<=\s)___(?![_\w])")
CUE_INSTRUCTIONS = {
    "noun": (
        "The cue is a word (such as a verb) that fits one of the two kinds of things and not the"
        " other (for example: She cooked the ___ for dinner. / She parked the ___ for dinner.)."
    ),
    "adjective": (
        "In the first sentence the blank is an adjective, and in the second sentence the blank is"
        " a noun. The cue is a word that decides this (for example: It was so ___. / It was"
        " the ___.)."
    ),
}
SCHEMA = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"first": {"type": "string"}, "second": {"type": "string"}},
                "required": ["first", "second"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["pairs"],
    "additionalProperties": False,
}


@dataclass
class MinimalPairCandidate:
    target: str
    target_category: str
    target_token_count: int
    donor: str
    donor_role: str
    # 足す語義の種類（same-category・other-category・adjective）
    kind: str
    # 手がかりが donor の語義を示す文と、ターゲット語の元の語義を示す文（空欄は ___）
    donor_sentence: str
    target_sentence: str
    donor_cue: str
    target_cue: str


def generate_minimal_pairs(settings: Settings, arguments: argparse.Namespace) -> None:
    prompt = PromptTemplate.load(PROMPT)
    step = Step(
        COMMAND,
        inputs={
            word_finalization.OUTPUT_PATH: word_finalization.COMMAND,
            sense_addition_assignments.ASSIGNMENTS_PATH: sense_addition_assignments.COMMAND,
        },
        outputs=[OUTPUT_PATH],
        settings={
            **prompt_record_settings(settings.llm, [prompt]),
            "candidates": candidates_per_sense_pair(settings),
            "tokenizer": next(iter(settings.models.values())).model_dump(),
        },
    )
    client = LlmClient(settings.llm, ANSWERS_PATH, rerun_from=COMMAND)
    run_unless_unchanged(
        step,
        lambda: build_minimal_pair_candidates(
            settings,
            prompt,
            client,
            word_finalization.OUTPUT_PATH,
            sense_addition_assignments.ASSIGNMENTS_PATH,
            OUTPUT_PATH,
        ),
        arguments.recompute,
    )


def candidates_per_sense_pair(settings: Settings) -> int:
    """語義の組ごとに、LLM に作らせる候補の数（最小対の数 × 倍率）。"""
    evaluation = settings.evaluation
    return math.ceil(
        evaluation.minimal_pairs_per_sense_pair * evaluation.minimal_pair_candidate_factor
    )


def build_minimal_pair_candidates(
    settings: Settings,
    prompt: PromptTemplate,
    client: LlmClient,
    words_path: Path,
    assignments_path: Path,
    output_path: Path,
) -> None:
    tokenizer = WordTokenizer.from_model(next(iter(settings.models.values())))
    words = {word.word: word for word in read_rows(words_path, SelectedWord)}
    largest = max(sense_addition_assignments.NUMBERS_OF_SENSES)
    sense_pairs = [
        (row, row.run.rsplit("-", 1)[0])
        for row in read_rows(assignments_path, SenseAdditionAssignment)
        if row.run.endswith(f"-{largest}")
    ]
    number = candidates_per_sense_pair(settings)
    requests = [
        LlmRequest(
            prompt,
            {
                "donor": row.donor,
                "donor_definition": words[row.donor].definition,
                "target": row.target,
                "target_definition": words[row.target].definition,
                "cue_instruction": CUE_INSTRUCTIONS[row.donor_role],
                "number": str(number),
            },
            with_maximum_items(SCHEMA, "pairs", number),
            maximum_tokens=60 * number,
        )
        for row, _ in sense_pairs
    ]
    managed = [word for word in words.values() if word.role in MANAGED_ROLES]
    excluded = frozenset(
        form for word in managed for form in (word.word, *inflected_forms(word.word, word.role))
    )
    candidates = []
    empty = []
    answers = client.ask_all(requests)
    # 答えがない語義の組は、最小対が1つもなくなり、後の分析から黙って消えるので止まる
    require_answers(answers, "最小対", COMMAND)
    for (row, kind), answer in zip(sense_pairs, answers, strict=True):
        kept: list[MinimalPairCandidate] = []
        for pair in (answer or {"pairs": []})["pairs"]:
            cues = _cues(pair["first"], pair["second"], row.target, tokenizer, excluded)
            if cues is None or len(kept) == number:
                continue
            candidate = MinimalPairCandidate(
                row.target,
                row.target_category,
                row.target_token_count,
                row.donor,
                row.donor_role,
                kind,
                pair["first"].strip(),
                pair["second"].strip(),
                *cues,
            )
            # 後の処理は、最小対を donor の文で見分けるので、片方の文だけが同じ候補も使わない
            if all(
                other.donor_sentence != candidate.donor_sentence
                and other.target_sentence != candidate.target_sentence
                for other in kept
            ):
                kept.append(candidate)
        if not kept:
            empty.append(f"{row.target}/{row.donor}")
        candidates += kept
    if empty:
        raise CommandStopped(
            f"使える候補が1つもない語義の組があります（{len(empty)} 組: {empty[:10]}）。"
            "settings.yaml の evaluation.minimal_pair_candidate_factor を大きくするか、"
            f"llm_prompts/{PROMPT}.md を直してください。",
            rerun_from=COMMAND,
        )
    write_rows(output_path, MinimalPairCandidate, candidates)


def _cues(
    first: str,
    second: str,
    target: str,
    tokenizer: WordTokenizer,
    excluded: frozenset[str],
) -> tuple[str, str] | None:
    """2文で違う1語（手がかり）。最小対の条件に合わなければ None。

    excluded は、空欄の外にあってはいけない語（小文字）。
    """
    first_words, second_words = first.split(), second.split()
    for sentence in (first, second):
        if len(BLANK_PATTERN.findall(" " + sentence)) != 1 or sentence.count("_") != len(BLANK):
            return None
        if {word.lower() for word in WORD_PATTERN.findall(sentence)} & excluded:
            return None
    if len(first_words) != len(second_words):
        return None
    differences = [
        index
        for index, (one, other) in enumerate(zip(first_words, second_words, strict=True))
        if one != other
    ]
    blank = next(index for index, word in enumerate(first_words) if BLANK in word)
    if len(differences) != 1 or differences[0] >= blank:
        return None
    # 問題文では、文は "Sentence: " の後に置くので、先頭の空白を付けてトークン化する
    filled = [
        tokenizer.encode([" " + sentence.replace(BLANK, target)]) for sentence in (first, second)
    ]
    if len(filled[0]) != len(filled[1]):
        return None
    return first_words[differences[0]], second_words[differences[0]]
