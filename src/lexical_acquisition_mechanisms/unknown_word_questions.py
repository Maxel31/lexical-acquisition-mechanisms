"""未知語の獲得の課題の問題（evaluation-data build-task-questions の一部）。

run ごと・未知語ごと・中立文脈の条件ごとに、課題1（語義の同定。正解は割り当てた名詞、ほかの選択肢は
別のカテゴリの名詞3語）と、課題2のカテゴリの問題を作る。文は「中立文脈 + 語 + .」（文脈なしの条件は
「語 + .」）。問題文の語は {word} のまま書き、評価のときに、未知語か元の名詞のトークン列を入れる
（同じ問題を、元の名詞でも測るため）。ほかの選択肢と選択肢の並び順は run と未知語ごとに、例は
問題ごとに、seed から決める。
"""

import random
from collections.abc import Mapping, Sequence

from lexical_acquisition_mechanisms.neutral_contexts import NeutralContext
from lexical_acquisition_mechanisms.task_prompts import DISTRACTORS, WORD, TaskItem, prompt_template
from lexical_acquisition_mechanisms.unknown_word_assignments import UnknownWordAssignment
from lexical_acquisition_mechanisms.word_selection import SelectedWord


def unknown_word_questions(
    assignments: Sequence[UnknownWordAssignment],
    words: Mapping[str, SelectedWord],
    contexts: Sequence[NeutralContext],
    examples: Mapping[str, list[tuple[TaskItem, str]]],
    categories: Sequence[str],
    number_of_examples: int,
    seed: int,
) -> list[dict[str, object]]:
    nouns = [word for word in words.values() if word.role == "noun"]
    questions = []
    for assignment in assignments:
        random_numbers = random.Random(f"{seed}:{assignment.run}:{assignment.unknown_word}")
        others = sorted(noun.word for noun in nouns if noun.category != assignment.noun_category)
        senses = [assignment.noun, *random_numbers.sample(others, DISTRACTORS)]
        random_numbers.shuffle(senses)
        shuffled_categories = list(categories)
        random_numbers.shuffle(shuffled_categories)
        for context in contexts:
            sentence = f"{context.context} {WORD}." if context.context else f"{WORD}."
            items = [
                TaskItem("sense", sentence, tuple(senses), assignment.noun),
                TaskItem(
                    "category", sentence, tuple(shuffled_categories), assignment.noun_category
                ),
            ]
            for item in items:
                key = ":".join(
                    map(
                        str,
                        (seed, assignment.run, assignment.unknown_word, context.number, item.task),
                    )
                )
                chosen = random.Random(key).sample(examples[item.task], number_of_examples)
                questions.append(
                    {
                        "run": assignment.run,
                        "unknown_word": assignment.unknown_word,
                        "noun": assignment.noun,
                        "condition": context.number,
                        "task": item.task,
                        "prompt_template": prompt_template(chosen, item),
                        "choices": list(item.choices),
                        "answer": item.answer,
                    }
                )
    return questions
