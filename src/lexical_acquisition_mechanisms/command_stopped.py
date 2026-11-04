"""処理が止まったことを、コマンドの入口に伝える例外。

コマンドの入口と各段階の処理の両方が使うので、ほかのモジュールに依存しないここに置く。
"""


class CommandStopped(Exception):
    """処理を続けられず、人が直してから実行し直す必要がある。

    message には、何が起きていて何を直せばよいかを書く。rerun_from には、直した後に
    実行し直すコマンド（例: `words select`）を書く。
    """

    def __init__(self, message: str, rerun_from: str) -> None:
        super().__init__(message)
        self.message = message
        self.rerun_from = rerun_from
