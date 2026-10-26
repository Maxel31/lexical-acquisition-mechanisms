"""環境の確認: 主要な依存パッケージが読み込めること。"""


def test_import() -> None:
    import torch
    import transformers

    import lam

    assert lam.__doc__
    assert torch.__version__
    assert transformers.__version__
