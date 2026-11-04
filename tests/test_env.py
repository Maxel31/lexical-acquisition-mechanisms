"""環境の確認: 主要な依存パッケージが読み込めること。"""


def test_import() -> None:
    import torch
    import transformers

    import lexical_acquisition_mechanisms

    assert lexical_acquisition_mechanisms.__doc__
    assert torch.__version__
    assert transformers.__version__
