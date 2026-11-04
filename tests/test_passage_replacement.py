"""用例の語の置き換えのテスト。"""

import pytest

from lexical_acquisition_mechanisms.passage_replacement import (
    PassageReplacer,
    Replacement,
    ReplacementError,
)
from lexical_acquisition_mechanisms.settings import ModelSource
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer


@pytest.fixture(scope="module")
def tokenizer(tiny_model_source: ModelSource) -> WordTokenizer:
    return WordTokenizer.from_model(tiny_model_source)


def word_replacement(tokenizer: WordTokenizer, source: str, target: str) -> Replacement:
    return Replacement(
        tuple(tokenizer.word_token_ids(source)),
        tuple(tokenizer.word_token_ids(target)),
        target,
        adjust_articles=True,
    )


def positions_of(tokenizer: WordTokenizer, token_ids: list[int], word: str) -> list[int]:
    word_id = tokenizer.word_token_ids(word)[0]
    return [index for index, token in enumerate(token_ids) if token == word_id]


def test_articles_follow_the_new_word(tokenizer: WordTokenizer) -> None:
    token_ids = tokenizer.encode(["We ate an apple and the apple's core."])
    replacer = PassageReplacer(tokenizer, tokens_per_passage=len(token_ids))

    replaced = replacer.replace(
        token_ids,
        positions_of(tokenizer, token_ids, "apple"),
        word_replacement(tokenizer, "apple", "car"),
    )

    assert tokenizer.decode(replaced.token_ids) == "We ate a car and the car's core."
    assert replaced.records == [("an apple", "a car"), ("apple", "car")]
    assert [tokenizer.decode(replaced.token_ids[p : p + 1]) for p in replaced.positions] == [
        " car"
    ] * 2


def test_capitalized_article_before_a_vowel(tokenizer: WordTokenizer) -> None:
    token_ids = tokenizer.encode(["A pasta was served. Then we left the room."])
    replacer = PassageReplacer(tokenizer, tokens_per_passage=len(token_ids))

    replaced = replacer.replace(
        token_ids,
        positions_of(tokenizer, token_ids, "pasta"),
        word_replacement(tokenizer, "pasta", "obrasiva"),
    )

    # 長くなった分（obrasiva は3トークン）は、置き換えた語から遠い端から除く
    assert len(replaced.token_ids) == len(token_ids)
    assert tokenizer.decode(replaced.token_ids).startswith("An obrasiva was served.")


def test_new_token_keeps_the_article(tokenizer: WordTokenizer) -> None:
    token_ids = tokenizer.encode(["I saw an apple today."])
    (new_token,) = tokenizer.unused_token_ids(1)
    replacement = Replacement(
        tuple(tokenizer.word_token_ids("apple")), (new_token,), "<new_01>", adjust_articles=False
    )
    replacer = PassageReplacer(tokenizer, tokens_per_passage=len(token_ids))

    replaced = replacer.replace(token_ids, positions_of(tokenizer, token_ids, "apple"), replacement)

    assert tokenizer.decode(replaced.token_ids) == f"I saw an <新しいトークン {new_token}> today."


def test_missing_occurrences_are_errors(tokenizer: WordTokenizer) -> None:
    token_ids = tokenizer.encode(["We ate an apple."])
    replacer = PassageReplacer(tokenizer, tokens_per_passage=len(token_ids))
    replacement = word_replacement(tokenizer, "apple", "car")

    with pytest.raises(ReplacementError, match="置き換える出現がありません"):
        replacer.replace(token_ids, [], replacement)
    with pytest.raises(ReplacementError, match="位置 0 に元の語がありません"):
        replacer.replace(token_ids, [0], replacement)


def test_parts_of_words_are_not_treated_as_articles(tokenizer: WordTokenizer) -> None:
    # " pec|an| pie" の "an" は、pecan の一部で冠詞ではない
    token_ids = tokenizer.encode(["We ate pecan pie today."])
    replacer = PassageReplacer(tokenizer, tokens_per_passage=len(token_ids))

    replaced = replacer.replace(
        token_ids,
        positions_of(tokenizer, token_ids, "pie"),
        word_replacement(tokenizer, "pie", "car"),
    )

    assert tokenizer.decode(replaced.token_ids) == "We ate pecan car today."
    assert replaced.records == [("pie", "car")]
