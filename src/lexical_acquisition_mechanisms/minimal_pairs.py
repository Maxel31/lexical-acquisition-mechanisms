"""最小対の候補を作る（evaluation-data generate-minimal-pairs）。

最小対は、手がかりの1語だけが違い、ターゲット語の語義が変わる2文。ターゲット語と、語義の追加で足す
語（donor。足す数が最大の run のもの）の組ごとに、LLM に候補を作らせる。LLM は、donor の語義を示す
文（空欄 ___ と、その前の手がかり）と、手がかりを置き換える語を書き、ターゲット語の文は、手がかりを
置き換えて作る（2文を別々に書かせると、手がかり以外も違う文になりやすいので）。
- 名詞を足す組（同カテゴリ・別カテゴリ）: 手がかりは、一方のものにだけ合う語
  （例: grape と car なら She peeled the ___ / She parked the ___）。
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
使える候補が最小対の数に届かない語義の組は、頼む数を ASK_AGAIN_STEP ずつ変えて、ASK_AGAIN_TIMES 回
まで問い直し、使える候補を足す（温度0の LLM は、組によって短い答えや、2つの手がかりが同じ語の答えを
返すことがあり、頼む数を変えると別の答えになるので）。
それでも使える候補が1つもない語義の組があれば止まる（その語義が、後の全ての分析から黙って消えないように）。
語義の組ごとに、最小対の数（evaluation.minimal_pairs_per_sense_pair）×
evaluation.minimal_pair_candidate_factor 個まで残し、data/evaluation/minimal_pair_candidates.csv に
書く（学習前のモデルで確かめるのは、evaluation-data check-with-pretrained-models）。
"""

import argparse
import math
import re
from collections.abc import Sequence
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
# 手がかりの語を探すときに、語の前後から除く記号
PUNCTUATION = ".,;:!?\"'()"
# 空欄: 前が空白で、ちょうど3本の下線（文の先頭なら、問題文では "Sentence:" の後の空白の後）。
# 後に英数字が続く空欄（___s など）は、語の形を変えるので使わない
BLANK_PATTERN = re.compile(r"(?<=\s)___(?![_\w])")
# 1組の答えに許す長さ（トークン）
MAXIMUM_TOKENS_PER_PAIR = 60
# 使える候補が最小対の数に届かない組を、問い直す回数と、問い直すたびに頼む数に足す数
ASK_AGAIN_TIMES = 3
ASK_AGAIN_STEP = 5
CUE_INSTRUCTIONS = {
    "noun": (
        "The cue is a word (such as a verb) that fits one of the two kinds of things and not the"
        ' other. Start the sentence with who does something (such as "She", "They" or "The'
        ' farmer"), then the cue verb, then the blank. For example, for "grape" and "car":'
        ' {"sentence": "She peeled the ___.", "first_cue": "peeled", "second_cue": "parked"}.'
        " Choose verbs for what people typically do only with that kind of thing. Do not use verbs"
        ' that fit almost any thing, such as "saw", "spotted", "found", "bought", "placed",'
        ' "carried", "liked" or "wanted".'
    ),
    "adjective": (
        "In the first sentence the blank is an adjective: write an adverb such as"
        ' "so", "very", "too" or "quite" in the sentence, right before the blank ("so ___"); this'
        ' adverb is the first cue word. The second cue word is a word such as "the", "a", "his"'
        ' or "her", which makes the blank a noun. For example, for "quiet" and "car":'
        ' {"sentence": "It was so ___.", "first_cue": "so", "second_cue": "the"} (the second'
        ' sentence is "It was the ___."), and {"sentence": "She looked at his very ___ face."} is'
        ' wrong, because "the ___ face" does not make the blank a noun.'
    ),
}
SCHEMA = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sentence": {"type": "string"},
                    "first_cue": {"type": "string"},
                    "second_cue": {"type": "string"},
                },
                "required": ["sentence", "first_cue", "second_cue"],
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
        settings=record_settings(settings, prompt),
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


def record_settings(settings: Settings, prompt: PromptTemplate) -> dict:
    """最小対の候補を変える設定（実行の記録に書き、変わったら作り直す）。"""
    return {
        **prompt_record_settings(settings.llm, [prompt]),
        "candidates": candidates_per_sense_pair(settings),
        "tokenizer": next(iter(settings.models.values())).model_dump(),
        # 問い合わせの本文・答えの形・長さの上限と、空欄・手がかりを探す決まりも、出力を変える
        "cue_instructions": CUE_INSTRUCTIONS,
        "schema": SCHEMA,
        "maximum_tokens_per_pair": MAXIMUM_TOKENS_PER_PAIR,
        "ask_again": {"times": ASK_AGAIN_TIMES, "step": ASK_AGAIN_STEP},
        "minimal_pairs_per_sense_pair": settings.evaluation.minimal_pairs_per_sense_pair,
        "blank_pattern": BLANK_PATTERN.pattern,
        "cue_punctuation": PUNCTUATION,
    }


def candidates_per_sense_pair(settings: Settings) -> int:
    """語義の組ごとに、LLM に作らせる候補の数（最小対の数 × 倍率）。"""
    evaluation = settings.evaluation
    return math.ceil(
        evaluation.minimal_pairs_per_sense_pair * evaluation.minimal_pair_candidate_factor
    )


@dataclass(frozen=True)
class SensePair:
    """最小対を作る語義の組（ターゲット語の元の語義と、donor の語義）。"""

    target: str
    target_category: str
    target_token_count: int
    donor: str
    donor_role: str
    # 足す語義の種類（same-category・other-category・adjective）
    kind: str


def sense_addition_pairs(assignments_path: Path) -> list[SensePair]:
    """語義の追加の割り当て表の、足す語義の数が最大の run の行（全ての語義の組を含む）。"""
    largest = max(sense_addition_assignments.NUMBERS_OF_SENSES)
    return [
        SensePair(
            row.target,
            row.target_category,
            row.target_token_count,
            row.donor,
            row.donor_role,
            row.run.rsplit("-", 1)[0],
        )
        for row in read_rows(assignments_path, SenseAdditionAssignment)
        if row.run.endswith(f"-{largest}")
    ]


def build_minimal_pair_candidates(
    settings: Settings,
    prompt: PromptTemplate,
    client: LlmClient,
    words_path: Path,
    assignments_path: Path,
    output_path: Path,
) -> None:
    write_minimal_pair_candidates(
        settings, prompt, client, words_path, sense_addition_pairs(assignments_path), output_path
    )


def write_minimal_pair_candidates(
    settings: Settings,
    prompt: PromptTemplate,
    client: LlmClient,
    words_path: Path,
    sense_pairs: Sequence[SensePair],
    output_path: Path,
    command: str = COMMAND,
) -> None:
    """語義の組ごとに、LLM に最小対の候補を作らせ、条件に合うものを output_path に書く。

    command は、止まったときに実行し直すコマンド。
    """
    tokenizer = WordTokenizer.from_model(next(iter(settings.models.values())))
    words = {word.word: word for word in read_rows(words_path, SelectedWord)}
    number = candidates_per_sense_pair(settings)

    def request(row: SensePair, asked: int) -> LlmRequest:
        return LlmRequest(
            prompt,
            {
                "donor": row.donor,
                "donor_definition": words[row.donor].definition,
                "target": row.target,
                "target_definition": words[row.target].definition,
                "cue_instruction": CUE_INSTRUCTIONS[row.donor_role],
                "number": str(asked),
            },
            with_maximum_items(SCHEMA, "pairs", asked),
            maximum_tokens=MAXIMUM_TOKENS_PER_PAIR * asked,
        )

    managed = [word for word in words.values() if word.role in MANAGED_ROLES]
    excluded = frozenset(
        form for word in managed for form in (word.word, *inflected_forms(word.word, word.role))
    )
    answers = client.ask_all([request(row, number) for row in sense_pairs])
    # 答えがない語義の組は、最小対が1つもなくなり、後の分析から黙って消えるので止まる
    require_answers(answers, "最小対", command)
    kept: list[list[MinimalPairCandidate]] = [[] for _ in sense_pairs]
    for found, row, answer in zip(kept, sense_pairs, answers, strict=True):
        _keep_usable(found, answer, row, number, tokenizer, excluded)
    wanted = settings.evaluation.minimal_pairs_per_sense_pair
    for attempt in range(1, ASK_AGAIN_TIMES + 1):
        short = [index for index, found in enumerate(kept) if len(found) < wanted]
        if not short:
            break
        asked = number + attempt * ASK_AGAIN_STEP
        again = client.ask_all([request(sense_pairs[index], asked) for index in short])
        for index, answer in zip(short, again, strict=True):
            _keep_usable(kept[index], answer, sense_pairs[index], number, tokenizer, excluded)
    empty = [
        f"{row.target}/{row.donor}"
        for found, row in zip(kept, sense_pairs, strict=True)
        if not found
    ]
    candidates = [candidate for found in kept for candidate in found]
    if empty:
        raise CommandStopped(
            f"使える候補が1つもない語義の組があります（{len(empty)} 組: {empty[:10]}）。"
            "settings.yaml の evaluation.minimal_pair_candidate_factor を大きくするか、"
            f"llm_prompts/{PROMPT}.md を直してください。",
            rerun_from=command,
        )
    write_rows(output_path, MinimalPairCandidate, candidates)


def _keep_usable(
    kept: list[MinimalPairCandidate],
    answer: dict | None,
    row: SensePair,
    number: int,
    tokenizer: WordTokenizer,
    excluded: frozenset[str],
) -> None:
    """答えの候補のうち、最小対の条件に合い、kept の候補と重ならないものを、number 個まで足す。"""
    for pair in (answer or {"pairs": []})["pairs"]:
        if len(kept) == number:
            return
        sentences = swapped_cue_sentences(pair["sentence"], pair["first_cue"], pair["second_cue"])
        if sentences is None:
            continue
        cues = _cues(*sentences, row.target, tokenizer, excluded)
        if cues is None:
            continue
        candidate = MinimalPairCandidate(
            row.target,
            row.target_category,
            row.target_token_count,
            row.donor,
            row.donor_role,
            row.kind,
            *sentences,
            *cues,
        )
        # 後の処理は、最小対を donor の文で見分けるので、片方の文だけが同じ候補も使わない
        if all(
            other.donor_sentence != candidate.donor_sentence
            and other.target_sentence != candidate.target_sentence
            for other in kept
        ):
            kept.append(candidate)


def swapped_cue_sentences(sentence: str, first_cue: str, second_cue: str) -> tuple[str, str] | None:
    """LLM の文（donor の語義の文）と、手がかりを置き換えた文。

    手がかりが、文の中に（前後の記号を除いて）ちょうど1回、空欄より前にないときは None。
    """
    words = sentence.strip().split()
    first_cue, second_cue = first_cue.strip(), second_cue.strip()
    blank = next((index for index, word in enumerate(words) if BLANK in word), None)
    spots = [index for index, word in enumerate(words) if word.strip(PUNCTUATION) == first_cue]
    if blank is None or not first_cue or len(spots) != 1 or spots[0] >= blank:
        return None
    swapped = list(words)
    swapped[spots[0]] = words[spots[0]].replace(first_cue, second_cue, 1)
    return " ".join(words), " ".join(swapped)


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
