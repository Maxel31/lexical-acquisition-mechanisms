"""テストで共有する部品。

テストは GPU を使わない（torch が GPU を調べる前に、GPU を見えなくする）。
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer, Olmo2Config, Olmo2ForCausalLM  # noqa: E402

from lexical_acquisition_mechanisms import wordnet_senses  # noqa: E402
from lexical_acquisition_mechanisms.settings import ModelSource, load_settings  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def tiny_model_source(tmp_path_factory: pytest.TempPathFactory) -> ModelSource:
    """無作為な重みの極小の OLMo 2 と、OLMo 2 のトークナイザを保存したディレクトリ。

    埋め込みの行数（100,352）とトークナイザは本物と同じにする（未使用のトークン番号を扱うため）。
    """
    source = load_settings(REPOSITORY_ROOT / "settings.yaml").models["OLMo-2-1124-7B"]
    try:
        tokenizer = AutoTokenizer.from_pretrained(source.repository, revision=source.revision)
    except OSError as error:
        pytest.skip(f"OLMo 2 のトークナイザを読み込めません: {error}")
    configuration = Olmo2Config(
        vocab_size=100352,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )
    torch.manual_seed(0)
    model = Olmo2ForCausalLM(configuration)
    directory = tmp_path_factory.mktemp("tiny-olmo2")
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return ModelSource(repository=str(directory))


@pytest.fixture(scope="session")
def wordnet_directory(tmp_path_factory: pytest.TempPathFactory):
    """WordNet を、テスト用の一時的なディレクトリにダウンロードして使う。"""
    original = wordnet_senses.NLTK_DATA_DIRECTORY
    wordnet_senses.NLTK_DATA_DIRECTORY = tmp_path_factory.mktemp("data") / "nltk_data"
    wordnet_senses.load_wordnet.cache_clear()
    try:
        wordnet_senses.load_wordnet()
    except (RuntimeError, OSError) as error:
        pytest.skip(f"WordNet を用意できません: {error}")
    yield wordnet_senses.NLTK_DATA_DIRECTORY
    wordnet_senses.NLTK_DATA_DIRECTORY = original
    wordnet_senses.load_wordnet.cache_clear()
