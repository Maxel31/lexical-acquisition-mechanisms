"""語の、標準の形以外の形のテスト。"""

from lexical_acquisition_mechanisms.inflected_forms import inflected_forms


def test_nouns_have_plural_forms() -> None:
    assert {"cherries", "cherrys"} <= inflected_forms("cherry", "noun")
    # WordNet の例外の一覧にある不規則な複数形
    assert "shelves" in inflected_forms("shelf", "target")
    assert "mice" in inflected_forms("mouse", "natural-noun")
    # 名詞には、比較級の形を作らない
    assert "lighter" not in inflected_forms("light", "noun")


def test_adjectives_have_comparative_and_superlative_forms() -> None:
    assert {"quieter", "quietest"} <= inflected_forms("quiet", "adjective")
    assert {"later", "latest"} <= inflected_forms("late", "adjective")
    assert {"happier", "happiest"} <= inflected_forms("happy", "natural-adjective")
    assert {"better", "best", "bigger"} <= (
        inflected_forms("good", "adjective") | inflected_forms("big", "adjective")
    )
