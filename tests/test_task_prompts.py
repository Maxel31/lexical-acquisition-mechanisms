"""課題の問題文のテスト。"""

import pytest

from lexical_acquisition_mechanisms.settings import ModelSource
from lexical_acquisition_mechanisms.task_prompts import (
    TaskItem,
    choice_token_ids,
    encode_prompt,
    encode_prompt_with_word_spans,
    item_text,
    prompt_template,
)
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer


@pytest.fixture
def tokenizer(tiny_model_source: ModelSource) -> WordTokenizer:
    return WordTokenizer.from_model(tiny_model_source)


def test_prompt_puts_the_word_tokens_in_both_lines(tokenizer: WordTokenizer) -> None:
    example = TaskItem("category", "She ate a ripe {word}.", ("food", "animal"), "food")
    question = TaskItem("category", "Do you know about the {word}.", ("animal", "food"), "animal")

    template = prompt_template([(example, "peach")], question)
    token_ids = encode_prompt(template, tokenizer.word_token_ids("zubbet"), tokenizer)

    assert item_text(example, "peach") == (
        "Sentence: She ate a ripe peach.\n"
        "Question: In this sentence, the word peach is a kind of: food / animal\n"
        "Answer: food"
    )
    assert template.endswith("the word {word} is a kind of: animal / food\nAnswer:")
    assert token_ids == tokenizer.encode([template.replace("{word}", "zubbet")])
    with pytest.raises(ValueError, match="1トークンではありません"):
        choice_token_ids(["food", "zubbet"], tokenizer)


def test_word_spans_cover_the_inserted_word(tokenizer: WordTokenizer) -> None:
    question = TaskItem("sense", "She ate the {word}.", ("food", "tool"), "food")
    word_ids = tokenizer.word_token_ids("zubbet")

    token_ids, spans = encode_prompt_with_word_spans(
        prompt_template([], question), word_ids, tokenizer
    )

    # 文の中と、問いの行の2か所
    assert len(spans) == 2
    assert all([token_ids[index] for index in span] == word_ids for span in spans)
    assert tokenizer.decode(token_ids[: spans[0].start]) == "Sentence: She ate the"
