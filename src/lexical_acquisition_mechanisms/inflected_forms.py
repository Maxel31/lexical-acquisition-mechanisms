"""語の、標準の形以外の形（名詞の複数形、形容詞の比較級・最上級）。

用例には、語の標準の形（先頭スペース付き・小文字・単数）だけを含め、それ以外の形を含む用例は
使わない（設計書 0-4）。形は、規則で作るもの（s・es、子音 + y → ies、形容詞の er・est）と、
WordNet の例外の一覧にあるもの（例: mice・shelves・better・bigger）を使う。
"""

from functools import cache

from lexical_acquisition_mechanisms.wordnet_senses import load_wordnet

# 役割ごとの品詞（WordNet の記号。n: 名詞、a: 形容詞）
PARTS_OF_SPEECH = {
    "noun": "n",
    "target": "n",
    "natural-noun": "n",
    "adjective": "a",
    "natural-adjective": "a",
}
VOWELS = "aeiou"


def inflected_forms(word: str, role: str) -> frozenset[str]:
    """語の、標準の形以外の形（小文字。s・es は品詞によらず含める）。"""
    part_of_speech = PARTS_OF_SPEECH[role]
    forms = {word + "s", word + "es", *_exception_forms(part_of_speech).get(word, ())}
    stem_before_y = (
        word[:-1] if len(word) > 1 and word[-1] == "y" and word[-2] not in VOWELS else None
    )
    if part_of_speech == "n":
        if stem_before_y:
            forms.add(stem_before_y + "ies")
    else:
        forms |= {word + "r", word + "st"} if word.endswith("e") else {word + "er", word + "est"}
        if stem_before_y:
            forms |= {stem_before_y + "ier", stem_before_y + "iest"}
    return frozenset(forms - {word})


@cache
def _exception_forms(part_of_speech: str) -> dict[str, set[str]]:
    """WordNet の例外の一覧の、元の形から、例外の形への対応。"""
    forms: dict[str, set[str]] = {}
    for form, bases in load_wordnet()._exception_map[part_of_speech].items():
        for base in bases:
            forms.setdefault(base, set()).add(form)
    return forms
