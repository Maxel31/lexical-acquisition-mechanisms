"""語のトークン化のテスト。OLMo 2 のトークナイザを使う（初回だけ Hugging Face から取得する）。"""

from pathlib import Path

import pytest

from lexical_acquisition_mechanisms.settings import load_settings
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def word_tokenizer() -> WordTokenizer:
    repository = load_settings(REPOSITORY_ROOT / "settings.yaml").models["OLMo-2-1124-7B"]
    try:
        return WordTokenizer.from_model(repository)
    except OSError as error:
        pytest.skip(f"OLMo 2 のトークナイザを読み込めません: {error}")


def test_leading_space_changes_the_tokens(word_tokenizer: WordTokenizer) -> None:
    assert len(word_tokenizer.word_token_ids("pasta")) == 1
    assert len(word_tokenizer.word_token_ids("pasta", leading_space=False)) == 2
    assert word_tokenizer.decode(word_tokenizer.word_token_ids("resonunt")) == " resonunt"
    assert len(word_tokenizer.word_token_ids("resonunt")) == 2


@pytest.mark.parametrize("word", ["", " pasta", "pasta "])
def test_word_with_surrounding_spaces_is_rejected(word_tokenizer: WordTokenizer, word: str) -> None:
    with pytest.raises(ValueError, match="空白"):
        word_tokenizer.word_token_ids(word)


def test_parts_split_before_a_word_match_the_whole_sentence(
    word_tokenizer: WordTokenizer,
) -> None:
    whole = word_tokenizer.encode(["Do you know about the pasta"])

    assert word_tokenizer.encode(["Do you know about the", " pasta"]) == whole
    assert (
        word_tokenizer.encode(["Do you know about the", word_tokenizer.word_token_ids("pasta")])
        == whole
    )
    # 空白を前の部分に残して区切ると、空白だけのトークンが入り、全体とは一致しない
    assert word_tokenizer.encode(["Do you know about the ", "pasta"]) != whole


def test_word_can_be_split_differently_from_the_standard(word_tokenizer: WordTokenizer) -> None:
    two_pieces = word_tokenizer.split_word_token_ids([" pas", "ta"])
    three_pieces = word_tokenizer.split_word_token_ids([" p", "as", "ta"])

    assert len(two_pieces) == 2
    assert len(three_pieces) == 3
    assert word_tokenizer.decode(two_pieces) == " pasta"
    assert word_tokenizer.decode(three_pieces) == " pasta"
    assert word_tokenizer.encode(["Do you know about the", two_pieces])[-2:] == two_pieces


def test_piece_that_is_not_one_token_is_rejected(word_tokenizer: WordTokenizer) -> None:
    with pytest.raises(ValueError, match="1トークンになりません"):
        word_tokenizer.split_word_token_ids([" resonunt"])


def test_special_token_text_is_tokenized_literally(word_tokenizer: WordTokenizer) -> None:
    end_of_text = word_tokenizer.end_of_text_id

    assert end_of_text not in word_tokenizer.encode(["a <|endoftext|> b"])
    # 個人情報の置き換えに使われた追加トークンは、事前学習のときと同じく1トークンになる
    assert word_tokenizer.encode(["|||EMAIL_ADDRESS|||"]) == [100_262]
    assert word_tokenizer.encode([[end_of_text], "Do"])[0] == end_of_text
    assert word_tokenizer.decode([end_of_text]) == "<|endoftext|>"


def test_unused_token_ids_are_the_rows_after_the_vocabulary(word_tokenizer: WordTokenizer) -> None:
    # トークナイザの語彙は 100,278 個、モデルの設定の vocab_size（埋め込みの行数）は 100,352
    assert word_tokenizer.number_of_embedding_rows == 100_352
    assert word_tokenizer.unused_token_ids(24) == list(range(100_278, 100_302))
    assert word_tokenizer.unused_token_ids(74)[-1] == 100_351

    for count in (75, -1):
        with pytest.raises(ValueError, match="使われていないトークン番号は 74 個です"):
            word_tokenizer.unused_token_ids(count)


def test_new_tokens_are_visible_when_decoded(word_tokenizer: WordTokenizer) -> None:
    new_token = word_tokenizer.unused_token_ids(1)[0]
    token_ids = word_tokenizer.encode(["Do you know about the", [new_token], "?"])

    assert word_tokenizer.decode(token_ids) == "Do you know about the <新しいトークン 100278>?"


@pytest.mark.parametrize("token_id", [-1, 100_352])
def test_token_ids_outside_the_embedding_are_rejected(
    word_tokenizer: WordTokenizer, token_id: int
) -> None:
    with pytest.raises(ValueError, match="埋め込みの行"):
        word_tokenizer.encode(["Do", [token_id]])
    with pytest.raises(ValueError, match="埋め込みの行"):
        word_tokenizer.decode([token_id])


def test_spans_give_the_position_of_each_part(word_tokenizer: WordTokenizer) -> None:
    word_ids = word_tokenizer.word_token_ids("resonunt")
    token_ids, spans = word_tokenizer.encode_with_spans(
        [[word_tokenizer.end_of_text_id], "Do you know about the", word_ids, "?"]
    )

    assert token_ids[spans[2].start : spans[2].stop] == word_ids
    # 語の最後のトークン（resonunt なら unt）の位置
    assert token_ids[spans[2][-1]] == word_ids[-1]
    assert spans[3].stop == len(token_ids)
