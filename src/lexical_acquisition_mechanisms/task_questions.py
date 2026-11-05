"""課題の問題を作る（evaluation-data build-task-questions）。

1. LLM が、自然な名詞（文の中での意味）とターゲット語（元の語義）の言い換えの語（同義語。なければ
   少し広い語）を挙げる。WordNet の同義語と上位語も候補にする。先頭スペース付きで1トークン・管理
   対象の語でない・語そのものでない・カテゴリ名でない・語のカテゴリの根（settings.yaml の
   wordnet_roots）とその上位語でない候補から、LLM が最も近い語を選ぶ
   （data/evaluation/paraphrases.csv。合う語がなければ空）。ターゲット語の言い換えは、語義の追加の
   課題1で元の語義の選択肢になるので、どの run の donor の語義も指せる語（WordNet の donor の語義の
   見出しの語と上位語。例: cheetah に lion を足すなら cat）も使わない。donor と共通しない低い
   上位語（例: pew → bench）は使う。
2. 例の候補（task_examples.csv）: 自然な語ごとに最初の文を、課題1（自然な名詞。正解は言い換えの語、
   ほかの選択肢は別のカテゴリの1トークンの自然な名詞3語）、課題2のカテゴリ（自然な名詞）、課題2の
   品詞（品詞のプローブの自然な名詞と形容詞）の問題にする。
3. 自然な語での確認の問題（natural_word_questions.parquet）: 例にしなかった自然な語の文の問題。
   問題ごとに、問う語と違う語の例を evaluation.number_of_task_examples 個選び、前に置く。選択肢の
   並び順と、例の選び方は、seed と問題から決まる。
4. 未知語の獲得の問題（unknown_word_questions.parquet。unknown_word_questions）。
"""

import argparse
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow as arrow
import pyarrow.parquet as parquet

from lexical_acquisition_mechanisms import (
    natural_word_sentences,
    neutral_contexts,
    sense_addition_assignments,
    unknown_word_assignments,
    word_finalization,
)
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.llm_client import (
    ANSWERS_PATH,
    LlmClient,
    LlmRequest,
    PromptTemplate,
    prompt_record_settings,
)
from lexical_acquisition_mechanisms.natural_word_sentences import NaturalWordSentence
from lexical_acquisition_mechanisms.neutral_contexts import NeutralContext
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.sense_addition_assignments import SenseAdditionAssignment
from lexical_acquisition_mechanisms.settings import CategorySettings, Settings
from lexical_acquisition_mechanisms.task_prompts import (
    DISTRACTORS,
    PARTS_OF_SPEECH,
    WORD,
    TaskItem,
    prompt_template,
)
from lexical_acquisition_mechanisms.unknown_word_assignments import UnknownWordAssignment
from lexical_acquisition_mechanisms.unknown_word_questions import unknown_word_questions
from lexical_acquisition_mechanisms.word_selection import MANAGED_ROLES, SelectedWord
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer
from lexical_acquisition_mechanisms.wordnet_senses import sense_words

COMMAND = "evaluation-data build-task-questions"
OUTPUT_DIRECTORY = Path("data/evaluation")
PARAPHRASES_PATH = OUTPUT_DIRECTORY / "paraphrases.csv"
EXAMPLES_PATH = OUTPUT_DIRECTORY / "task_examples.csv"
NATURAL_QUESTIONS_PATH = OUTPUT_DIRECTORY / "natural_word_questions.parquet"
UNKNOWN_WORD_QUESTIONS_PATH = OUTPUT_DIRECTORY / "unknown_word_questions.parquet"
PROMPT = "list-paraphrases"
CHOICE_PROMPT = "choose-paraphrase"
PARAPHRASES_PER_REQUEST = 10
SCHEMA = {
    "type": "object",
    "properties": {"paraphrases": {"type": "array", "items": {"type": "string"}}},
    "required": ["paraphrases"],
    "additionalProperties": False,
}


@dataclass
class Paraphrase:
    word: str
    role: str
    # 言い換えの語（使える語がなければ空）
    paraphrase: str


@dataclass
class TaskExample:
    task: str
    word: str
    # 文（語は {word}）・選択肢（| で区切る）・正解
    sentence: str
    choices: str
    answer: str


def build_task_questions(settings: Settings, arguments: argparse.Namespace) -> None:
    prompts = (PromptTemplate.load(PROMPT), PromptTemplate.load(CHOICE_PROMPT))
    step = Step(
        COMMAND,
        inputs={
            word_finalization.OUTPUT_PATH: word_finalization.COMMAND,
            natural_word_sentences.SENTENCES_PATH: natural_word_sentences.COMMAND,
            unknown_word_assignments.ASSIGNMENTS_PATH: unknown_word_assignments.COMMAND,
            neutral_contexts.CONTEXTS_PATH: neutral_contexts.COMMAND,
            sense_addition_assignments.ASSIGNMENTS_PATH: sense_addition_assignments.COMMAND,
        },
        outputs=[
            PARAPHRASES_PATH,
            EXAMPLES_PATH,
            NATURAL_QUESTIONS_PATH,
            UNKNOWN_WORD_QUESTIONS_PATH,
        ],
        settings={
            **prompt_record_settings(settings.llm, list(prompts)),
            "number_of_task_examples": settings.evaluation.number_of_task_examples,
            "categories": [category.model_dump() for category in settings.words.categories],
            "seed": settings.words.seed,
            "tokenizer": next(iter(settings.models.values())).model_dump(),
        },
    )
    client = LlmClient(settings.llm, ANSWERS_PATH, rerun_from=COMMAND)
    run_unless_unchanged(
        step,
        lambda: write_task_questions(
            settings,
            prompts,
            client,
            (
                word_finalization.OUTPUT_PATH,
                natural_word_sentences.SENTENCES_PATH,
                unknown_word_assignments.ASSIGNMENTS_PATH,
                neutral_contexts.CONTEXTS_PATH,
                sense_addition_assignments.ASSIGNMENTS_PATH,
            ),
            (PARAPHRASES_PATH, EXAMPLES_PATH, NATURAL_QUESTIONS_PATH, UNKNOWN_WORD_QUESTIONS_PATH),
        ),
        arguments.recompute,
    )


def write_task_questions(
    settings: Settings,
    prompts: tuple[PromptTemplate, PromptTemplate],
    client: LlmClient,
    input_paths: tuple[Path, Path, Path, Path, Path],
    output_paths: tuple[Path, Path, Path, Path],
) -> None:
    words_path, sentences_path, assignments_path, contexts_path, donors_path = input_paths
    paraphrases_path, examples_path, questions_path, unknown_questions_path = output_paths
    tokenizer = WordTokenizer.from_model(next(iter(settings.models.values())))
    words = {word.word: word for word in read_rows(words_path, SelectedWord)}
    sentences = read_rows(sentences_path, NaturalWordSentence)
    categories = [category.name for category in settings.words.categories]
    donors: dict[str, set[str]] = {}
    for row in read_rows(donors_path, SenseAdditionAssignment):
        donors.setdefault(row.target, set()).add(row.donor)
    paraphrases = _paraphrases(
        words, sentences, settings.words.categories, donors, prompts, client, tokenizer
    )
    write_rows(paraphrases_path, Paraphrase, paraphrases)
    missing = [row.word for row in paraphrases if row.role == "target" and not row.paraphrase]
    if missing:
        raise CommandStopped(
            f"言い換えの語が見つからないターゲット語があります（{missing}）。"
            "llm_prompts/list-paraphrases.md（候補を挙げる）か llm_prompts/choose-paraphrase.md"
            "（候補から選ぶ。合う語がないと none を選ぶ）を直すか、ターゲット語を選び直して"
            "ください。",
            rerun_from=COMMAND,
        )
    paraphrase_of = {row.word: row.paraphrase for row in paraphrases if row.paraphrase}
    one_token_nouns = {
        word.word: word.category
        for word in words.values()
        if word.role == "natural-noun" and word.token_count == 1
    }
    items: dict[str, list[tuple[TaskItem, str]]] = {}
    for index, sentence in enumerate(sentences):
        word = words[sentence.word]
        template = sentence.sentence[: sentence.word_start] + WORD
        template += sentence.sentence[sentence.word_start + len(word.word) :]
        random_numbers = random.Random(f"{settings.words.seed}:{index}")
        made = _items(word, template, paraphrase_of, one_token_nouns, categories, random_numbers)
        for item in made:
            items.setdefault(item.task, []).append((item, word.word))
    examples = {task: [] for task in items}
    questions = []
    for task, task_items in items.items():
        seen = set()
        for item, word in task_items:
            if word not in seen:
                seen.add(word)
                examples[task].append((item, word))
    write_rows(
        examples_path,
        TaskExample,
        [
            TaskExample(item.task, word, item.sentence, "|".join(item.choices), item.answer)
            for task_examples in examples.values()
            for item, word in task_examples
        ],
    )
    count = settings.evaluation.number_of_task_examples
    for task, task_items in items.items():
        example_set = {id(item) for item, _ in examples[task]}
        for number, (item, word) in enumerate(task_items):
            if id(item) in example_set:
                continue
            others = [example for example in examples[task] if example[1] != word]
            chosen = random.Random(f"{settings.words.seed}:{task}:{number}").sample(others, count)
            questions.append(
                {
                    "task": task,
                    "word": word,
                    "prompt_template": prompt_template(chosen, item),
                    "word_token_ids": tokenizer.word_token_ids(word),
                    "choices": list(item.choices),
                    "answer": item.answer,
                }
            )
    parquet.write_table(arrow.Table.from_pylist(questions), questions_path)
    unknown_questions = unknown_word_questions(
        read_rows(assignments_path, UnknownWordAssignment),
        words,
        read_rows(contexts_path, NeutralContext),
        examples,
        categories,
        count,
        settings.words.seed,
    )
    parquet.write_table(arrow.Table.from_pylist(unknown_questions), unknown_questions_path)


def _items(
    word: SelectedWord,
    template: str,
    paraphrase_of: dict[str, str],
    one_token_nouns: dict[str, str],
    categories: Sequence[str],
    random_numbers: random.Random,
) -> list[TaskItem]:
    """自然な語の1文から作る問題（課題1・カテゴリ・品詞）。"""
    items = []
    if word.role == "natural-noun":
        paraphrase = paraphrase_of.get(word.word, "")
        others = sorted(
            noun
            for noun, category in one_token_nouns.items()
            if category != word.category and noun != paraphrase
        )
        # 言い換えの語がない語と、ほかの選択肢が足りない語は、課題1の問題にしない
        if paraphrase and len(others) >= DISTRACTORS:
            choices = [paraphrase, *random_numbers.sample(others, DISTRACTORS)]
            random_numbers.shuffle(choices)
            items.append(TaskItem("sense", template, tuple(choices), paraphrase))
        shuffled = list(categories)
        random_numbers.shuffle(shuffled)
        items.append(TaskItem("category", template, tuple(shuffled), word.category))
    if word.part_of_speech_probe:
        parts = list(PARTS_OF_SPEECH)
        random_numbers.shuffle(parts)
        answer = "adjective" if word.role == "natural-adjective" else "noun"
        items.append(TaskItem("part-of-speech", template, tuple(parts), answer))
    return items


def _paraphrases(
    words: dict[str, SelectedWord],
    sentences: Sequence[NaturalWordSentence],
    categories: Sequence[CategorySettings],
    donors: dict[str, set[str]],
    prompts: tuple[PromptTemplate, PromptTemplate],
    client: LlmClient,
    tokenizer: WordTokenizer,
) -> list[Paraphrase]:
    """自然な名詞（最初の文で）とターゲット語の、言い換えの語。donors: ターゲット語ごとの donor。"""
    first_sentence = {}
    for sentence in sentences:
        first_sentence.setdefault(sentence.word, sentence.sentence)
    asked = [
        word
        for word in words.values()
        if word.role == "target" or (word.role == "natural-noun" and word.word in first_sentence)
    ]
    listing_prompt, choice_prompt = prompts
    values = [
        {
            "word": word.word,
            "definition": word.definition,
            "sentence": first_sentence.get(word.word, "(none)"),
        }
        for word in asked
    ]
    listed = client.ask_all(
        [
            LlmRequest(listing_prompt, value | {"number": str(PARAPHRASES_PER_REQUEST)}, SCHEMA)
            for value in values
        ]
    )
    managed = {word.word for word in words.values() if word.role in MANAGED_ROLES}
    candidates = []
    for word, answer in zip(asked, listed, strict=True):
        excluded = managed | unusable_words(word, categories, donors.get(word.word, ()), words)
        # LLM が挙げた語に、WordNet の同義語と上位語（使わない語は下で除く）を足す
        listing = (answer or {"paraphrases": []})["paraphrases"]
        proposed = [candidate.strip() for candidate in listing]
        proposed += sorted(sense_words(word.synset)) if word.synset else []
        candidates.append(
            list(
                dict.fromkeys(
                    candidate
                    for candidate in proposed
                    if candidate.isalpha()
                    and candidate.islower()
                    and candidate not in excluded
                    and len(tokenizer.word_token_ids(candidate)) == 1
                )
            )
        )
    # 使える候補から、LLM が語の意味に最も近い語を選ぶ（合う語がなければ none）
    asking = [index for index, usable in enumerate(candidates) if usable]
    chosen = client.ask_all(
        [
            LlmRequest(
                choice_prompt,
                values[index] | {"candidates": ", ".join(candidates[index])},
                _choice_schema(candidates[index]),
            )
            for index in asking
        ]
    )
    choice_of = {
        index: answer["choice"]
        for index, answer in zip(asking, chosen, strict=True)
        if answer and answer["choice"] in candidates[index]
    }
    return [
        Paraphrase(word.word, word.role, choice_of.get(index, ""))
        for index, word in enumerate(asked)
    ]


def _choice_schema(candidates: Sequence[str]) -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
            "choice": {"type": "string", "enum": [*candidates, "none"]},
        },
        "required": ["reason", "choice"],
        "additionalProperties": False,
    }


def unusable_words(
    word: SelectedWord,
    categories: Sequence[CategorySettings],
    donors: Iterable[str],
    words: dict[str, SelectedWord],
) -> set[str]:
    """語の言い換えに使わない語。
    語そのもの・カテゴリ名・語のカテゴリの根とその上位語（同じカテゴリの全ての語に当てはまる）と、
    donor の語義を指せる語（課題1で donor の選択肢とも合う）。
    """
    unusable = {word.word} | {category.name for category in categories}
    for category in categories:
        if category.name == word.category:
            for root in category.wordnet_roots:
                unusable |= sense_words(root)
    for donor in donors:
        if words[donor].synset:
            unusable |= sense_words(words[donor].synset)
    return unusable
