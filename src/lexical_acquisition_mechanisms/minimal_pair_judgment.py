"""最小対の候補を、LLM が2文それぞれで判定する（evaluation-data judge-minimal-pairs）。

generate-minimal-pairs の候補には、手がかりがどちらの語にも合う文（例: She spotted the ___.）
や、意味をなさない文が混ざる（1B の試行では、見本の確認に通ったのは 31%）。学習前のモデルでの確認
（check-with-pretrained-models）は、手がかりが語の出やすさを正しい向きに動かすかだけを見るので、
これらを防げない。そこで候補ごとに、LLM が2文それぞれについて次を判定し、両方とも合格した候補だけを
残す。
- donor の文: 空欄に donor を入れた文（例: We fried the meat.）が自然な英語で、同じ文の空欄に
  ターゲット語を元の語義で入れた文（例: We fried the cola. の cola が飲み物）は、不自然か、ずっと
  考えにくい
- ターゲット語の文: 空欄にターゲット語を入れた文（例: We drank the cola.）が自然な英語で、同じ文の
  空欄に donor を入れた文（例: We drank the meat.）は、不自然か、ずっと考えにくい
形容詞を足す組では、donor の文の空欄は形容詞として、ターゲット語の文の空欄は名詞として判定する。
答えが得られなかった候補は不合格にする。
合格した候補を data/evaluation/judged_minimal_pair_candidates.csv に
（check-with-pretrained-models が読む）、全ての候補の判定と理由を
data/evaluation/minimal_pair_judgments.csv に書く。
"""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from lexical_acquisition_mechanisms import minimal_pairs, word_finalization
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.llm_client import (
    ANSWERS_PATH,
    LlmClient,
    LlmRequest,
    PromptTemplate,
    prompt_record_settings,
)
from lexical_acquisition_mechanisms.minimal_pairs import BLANK, MinimalPairCandidate
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.settings import Settings
from lexical_acquisition_mechanisms.word_selection import SelectedWord

COMMAND = "evaluation-data judge-minimal-pairs"
OUTPUT_DIRECTORY = Path("data/evaluation")
JUDGED_PATH = OUTPUT_DIRECTORY / "judged_minimal_pair_candidates.csv"
JUDGMENTS_PATH = OUTPUT_DIRECTORY / "minimal_pair_judgments.csv"
PROMPT = "judge-minimal-pair"
# 理由と2つの判定の答えに許す長さ（トークン）
MAXIMUM_TOKENS = 400
ROLE_INSTRUCTIONS = {
    "noun": "Both words are nouns.",
    "adjective": (
        "Word A is an adjective and word B is a noun: in sentence 1 the blank must work as an"
        " adjective, and in sentence 2 as a noun."
    ),
}
SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "sentence_1_fits": {"type": "boolean"},
        "sentence_2_fits": {"type": "boolean"},
    },
    "required": ["reason", "sentence_1_fits", "sentence_2_fits"],
    "additionalProperties": False,
}


@dataclass
class MinimalPairJudgment(MinimalPairCandidate):
    # LLM の理由と、donor の文・ターゲット語の文それぞれの判定（答えがなければ理由は空で不合格）
    reason: str
    donor_sentence_fits: bool
    target_sentence_fits: bool
    passed: bool


def judge_minimal_pairs(settings: Settings, arguments: argparse.Namespace) -> None:
    prompt = PromptTemplate.load(PROMPT)
    step = Step(
        COMMAND,
        inputs={
            word_finalization.OUTPUT_PATH: word_finalization.COMMAND,
            minimal_pairs.OUTPUT_PATH: minimal_pairs.COMMAND,
        },
        outputs=[JUDGED_PATH, JUDGMENTS_PATH],
        settings=record_settings(settings, prompt),
    )
    client = LlmClient(settings.llm, ANSWERS_PATH, rerun_from=COMMAND)
    run_unless_unchanged(
        step,
        lambda: write_judged_minimal_pairs(
            prompt,
            client,
            word_finalization.OUTPUT_PATH,
            minimal_pairs.OUTPUT_PATH,
            (JUDGED_PATH, JUDGMENTS_PATH),
        ),
        arguments.recompute,
    )


def record_settings(settings: Settings, prompt: PromptTemplate) -> dict:
    """判定を変える設定（実行の記録に書き、変わったら判定し直す）。"""
    return {
        **prompt_record_settings(settings.llm, [prompt]),
        # 問い合わせの本文・答えの形・長さの上限も、出力を変える
        "role_instructions": ROLE_INSTRUCTIONS,
        "schema": SCHEMA,
        "maximum_tokens": MAXIMUM_TOKENS,
    }


def judgment_request(
    prompt: PromptTemplate, candidate: MinimalPairCandidate, words: dict[str, SelectedWord]
) -> LlmRequest:
    """候補の2文を、空欄に donor とターゲット語をそれぞれ入れて見せる問い合わせ。"""
    donor, target = candidate.donor, candidate.target
    return LlmRequest(
        prompt,
        {
            "donor": donor,
            "donor_definition": words[donor].definition,
            "target": target,
            "target_definition": words[target].definition,
            "role_instruction": ROLE_INSTRUCTIONS[candidate.donor_role],
            "donor_sentence_with_donor": candidate.donor_sentence.replace(BLANK, donor),
            "donor_sentence_with_target": candidate.donor_sentence.replace(BLANK, target),
            "target_sentence_with_target": candidate.target_sentence.replace(BLANK, target),
            "target_sentence_with_donor": candidate.target_sentence.replace(BLANK, donor),
        },
        SCHEMA,
        maximum_tokens=MAXIMUM_TOKENS,
    )


def write_judged_minimal_pairs(
    prompt: PromptTemplate,
    client: LlmClient,
    words_path: Path,
    candidates_path: Path,
    output_paths: tuple[Path, Path],
) -> None:
    judged_path, judgments_path = output_paths
    words = {word.word: word for word in read_rows(words_path, SelectedWord)}
    candidates = read_rows(candidates_path, MinimalPairCandidate)
    answers = client.ask_all([judgment_request(prompt, row, words) for row in candidates])
    judgments = []
    for candidate, answer in zip(candidates, answers, strict=True):
        donor_fits = bool(answer and answer["sentence_1_fits"])
        target_fits = bool(answer and answer["sentence_2_fits"])
        judgments.append(
            MinimalPairJudgment(
                **asdict(candidate),
                reason=answer["reason"] if answer else "",
                donor_sentence_fits=donor_fits,
                target_sentence_fits=target_fits,
                passed=donor_fits and target_fits,
            )
        )
    write_rows(judgments_path, MinimalPairJudgment, judgments)
    passed = [candidate for candidate, row in zip(candidates, judgments, strict=True) if row.passed]
    write_rows(judged_path, MinimalPairCandidate, passed)
    totals = Counter(row.kind for row in judgments)
    kept = Counter(row.kind for row in judgments if row.passed)
    for kind in sorted(totals):
        print(f"{kind}: 合格 {kept[kind]} / {totals[kind]} 組")
    missing = sum(answer is None for answer in answers)
    if missing:
        print(f"答えが得られず不合格にした候補: {missing} 組（実行し直すと問い合わせ直す）")
    groups = {(row.target, row.donor) for row in judgments}
    empty = sorted(groups - {(row.target, row.donor) for row in judgments if row.passed})
    if empty:
        print(
            f"合格した候補が1つもない語義の組があります（{len(empty)} 組: "
            f"{[f'{target}/{donor}' for target, donor in empty[:10]]}）。"
        )
