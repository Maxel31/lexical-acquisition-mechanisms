"""課題の問題文（設計書「下流課題の共通形式」）。

問題の前に、例をいくつか見せる。答えは生成させず、"Answer:" の次に来る選択肢の語（先頭スペース付きで
1トークン）の対数確率を比べて決める。問題文の中の語は {word} と書き、トークン番号の列で差し込む
（1トークンの未知語は、新しいトークン番号で入れるため）。

課題1（語義の同定）:
    Sentence: I put the knife in the drawer.
    Question: In this sentence, the word knife means the same as: spoon / blade / bottle / lamp
    Answer: blade
課題2（属性の判定。カテゴリと品詞）:
    Question: In this sentence, the word peach is a kind of: food / animal / ...
    Question: In this sentence, the part of speech of the word quiet is: noun / adjective
"""

from collections.abc import Sequence
from dataclasses import dataclass

from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

WORD = "{word}"
QUESTION_LINES = {
    "sense": "In this sentence, the word {word} means the same as: {choices}",
    "category": "In this sentence, the word {word} is a kind of: {choices}",
    "part-of-speech": "In this sentence, the part of speech of the word {word} is: {choices}",
}
PARTS_OF_SPEECH = ("noun", "adjective")


@dataclass(frozen=True)
class TaskItem:
    """1つの問題（例として見せるときは answer も書く）。"""

    # 課題の種類（sense・category・part-of-speech）
    task: str
    # 文（語は {word}）と、問う語の表記（例では語そのもの）
    sentence: str
    choices: tuple[str, ...]
    answer: str


def item_text(item: TaskItem, word: str = WORD, with_answer: bool = True) -> str:
    """問題の3行。word を文と問いの語に入れる（{word} のままなら、後でトークン番号を入れる）。"""
    question = QUESTION_LINES[item.task].format(word=word, choices=" / ".join(item.choices))
    answer = f" {item.answer}" if with_answer else ""
    return f"Sentence: {item.sentence.replace(WORD, word)}\nQuestion: {question}\nAnswer:{answer}"


def prompt_template(examples: Sequence[tuple[TaskItem, str]], question: TaskItem) -> str:
    """例（問題と、その問題の語）を並べ、最後に答えのない問題を置いた問題文（語は {word}）。"""
    pieces = [item_text(example, word) for example, word in examples]
    pieces.append(item_text(question, with_answer=False))
    return "\n\n".join(pieces)


def encode_prompt(template: str, word_ids: Sequence[int], tokenizer: WordTokenizer) -> list[int]:
    """問題文の {word} に、語のトークン列（先頭スペース付きの形）を入れたトークン列。

    {word} の前の空白は、語のトークン列の先頭スペースとして入れる。
    """
    pieces = template.split(WORD)
    parts: list[str | Sequence[int]] = []
    for index, piece in enumerate(pieces):
        is_last = index == len(pieces) - 1
        text = piece if is_last else piece.removesuffix(" ")
        if text:
            parts.append(text)
        if not is_last:
            parts.append(list(word_ids))
    return tokenizer.encode(parts)


def choice_token_ids(choices: Sequence[str], tokenizer: WordTokenizer) -> list[int]:
    """選択肢の語の、先頭スペース付きのトークン（1トークンでなければエラー）。"""
    token_ids = []
    for choice in choices:
        ids = tokenizer.word_token_ids(choice)
        if len(ids) != 1:
            raise ValueError(f"選択肢 {choice!r} が、先頭スペース付きで1トークンではありません。")
        token_ids.append(ids[0])
    return token_ids
