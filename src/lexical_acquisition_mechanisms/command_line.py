"""コマンド lexical-acquisition-mechanisms の入口。

`lexical-acquisition-mechanisms <段階> <処理> [引数]` の形で、README の各コマンドを実行する。
各段階の処理は、実装するときに all_actions() に足す。段階のモジュールからは、このモジュールを
import しない（import が循環するため）。止まるときは command_stopped.CommandStopped を使う。
"""

import argparse
import shlex
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from lexical_acquisition_mechanisms import (
    adjective_candidates,
    base_data,
    noun_candidates,
    unknown_word_candidates,
    word_selection,
)
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.settings import Settings, SettingsError, load_settings

COMMAND_NAME = "lexical-acquisition-mechanisms"

# README の段階と、--help に出す説明
STAGE_DESCRIPTIONS = {
    "base-data": "基データ（OLMo 2 の事前学習データの先頭）を用意する",
    "words": "実験で操作する語を選ぶ",
    "passages": "Pile から用例を切り出し、語義を判定する",
    "assignments": "run ごとに、どの用例をどの語に置き換えて差し込むかを決める",
    "training-data": "run ごとの学習データを作る",
    "evaluation-data": "評価データを作る",
    "training": "継続事前学習を行う",
    "unknown-word-acquisition": "未知語の獲得を評価・分析する",
    "sense-addition": "語義の追加を評価・分析する",
    "dependence-on-category-judging-heads": "カテゴリ判定 head への未知語の依存を分析する",
    "results": "全結果を集計する",
}


@dataclass(frozen=True)
class Action:
    """1つの処理（例: `words select`）。"""

    stage: str
    name: str
    description: str
    run: Callable[[Settings, argparse.Namespace], None]
    add_arguments: Callable[[argparse.ArgumentParser], None] | None = None


def all_actions() -> list[Action]:
    """README の全コマンド。"""
    return [
        Action(
            "base-data",
            "download",
            "基データ（OLMo 2 の事前学習データの、学習順で先頭の約3億トークン）と、"
            "その直後の検証データをダウンロードする",
            base_data.download,
        ),
        Action(
            "words",
            "generate-noun-candidates",
            "名詞の候補を LLM が挙げ、WordNet でのカテゴリと、Pile での使われ方を調べる",
            noun_candidates.generate_noun_candidates,
        ),
        Action(
            "words",
            "generate-adjective-candidates",
            "形容詞の候補を LLM が挙げ、Pile で形容詞として使われる割合と、"
            "カテゴリとの結びつきを調べる",
            adjective_candidates.generate_adjective_candidates,
        ),
        Action(
            "words",
            "generate-unknown-word-candidates",
            "未知語の候補（実在語の母音を替えた文字列と、ARC 非語データベースの非語）のうち、"
            "OLMo 2 の事前学習データと Pile に一度も現れないものを集める",
            unknown_word_candidates.generate_unknown_word_candidates,
        ),
        Action(
            "words",
            "select",
            "候補から、名詞72語・ターゲット語18語・形容詞72語と、自然な語を選ぶ",
            word_selection.select_words,
        ),
    ]


def build_parser(actions: Sequence[Action]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=COMMAND_NAME,
        description="言語モデルにおける語彙獲得と語義追加の機構解析の実験を行う。",
    )
    actions_by_stage: dict[str, list[Action]] = {}
    for action in actions:
        actions_by_stage.setdefault(action.stage, []).append(action)

    stage_parsers = parser.add_subparsers(dest="stage", metavar="<段階>", required=True)
    for stage, stage_actions in actions_by_stage.items():
        if stage not in STAGE_DESCRIPTIONS:
            raise ValueError(f"段階 {stage} は README にありません。")
        stage_parser = stage_parsers.add_parser(
            stage, help=STAGE_DESCRIPTIONS[stage], description=STAGE_DESCRIPTIONS[stage]
        )
        action_parsers = stage_parser.add_subparsers(dest="action", metavar="<処理>", required=True)
        for action in stage_actions:
            action_parser = action_parsers.add_parser(
                action.name, help=action.description, description=action.description
            )
            action_parser.add_argument(
                "--recompute", action="store_true", help="入力と設定が前回と同じでも、やり直す"
            )
            if action.add_arguments is not None:
                action.add_arguments(action_parser)
            action_parser.set_defaults(selected_action=action)
    return parser


def main(
    command_line_arguments: Sequence[str] | None = None, actions: Sequence[Action] | None = None
) -> int:
    """コマンドを実行し、終了コード（0 は成功）を返す。"""
    if command_line_arguments is None:
        command_line_arguments = sys.argv[1:]
    parser = build_parser(all_actions() if actions is None else actions)
    arguments = parser.parse_args(command_line_arguments)
    action: Action = arguments.selected_action
    try:
        action.run(load_settings(), arguments)
    except SettingsError as error:
        _report_stop(str(error), rerun_from=shlex.join(command_line_arguments))
        return 1
    except CommandStopped as stopped:
        _report_stop(stopped.message, stopped.rerun_from)
        return 1
    return 0


def _report_stop(message: str, rerun_from: str) -> None:
    print(f"止まりました: {message}", file=sys.stderr)
    print(
        f"直したら、次のコマンドから実行し直してください: uv run {COMMAND_NAME} {rerun_from}",
        file=sys.stderr,
    )
