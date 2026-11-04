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
    acquisition_order,
    adjective_candidates,
    assignments,
    attribute_direction_patching,
    attribute_probes,
    base_data,
    category_dependence,
    continued_pretraining,
    dependence_regression,
    evaluation_finalization,
    head_and_mlp_effects,
    hidden_state_similarity,
    minimal_pairs,
    natural_word_sentences,
    neutral_contexts,
    noun_candidates,
    passage_collection,
    passage_finalization,
    passage_review,
    passage_split,
    pretrained_model_checks,
    probe_and_normalization_words,
    real_word_probes,
    run_settings,
    sense_accuracy,
    sense_addition_report,
    significant_heads_and_mlps,
    similarity_ratings,
    task_questions,
    training_data,
    unknown_word_candidates,
    unknown_word_indicators,
    unknown_word_report,
    word_finalization,
    word_selection,
    word_sense_judgment,
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
            "候補から、名詞72語・ターゲット語18語・形容詞72語・未知語72個と、自然な語を選ぶ",
            word_selection.select_words,
        ),
        Action(
            "words",
            "rate-similarity-with-llm",
            "ターゲット語と名詞の組ごとに、意味の近さを LLM が評定する",
            similarity_ratings.rate_similarity_with_llm,
        ),
        Action(
            "evaluation-data",
            "select-neutral-contexts",
            "中立文脈（語の意味を示さない文の書き出し）を、公開されている OLMo 2 で、"
            "どのカテゴリの語も予測しやすくしていないことを確かめて選ぶ",
            neutral_contexts.select_neutral_contexts,
        ),
        Action(
            "words",
            "measure-hidden-state-similarity",
            "ターゲット語と名詞の組ごとに、公開されている OLMo 2 で内部表現の類似度を測り、"
            "確認用の data/words/review.csv を書き出す",
            hidden_state_similarity.measure_hidden_state_similarity,
        ),
        Action(
            "words",
            "finalize",
            "確認用の CSV で approved が 1 の語で、以降の手順で使う語の一覧を書き出す"
            "（0 の語は外し、words select で代わりの語を選ぶ）",
            word_finalization.finalize_words,
            word_finalization.add_arguments,
        ),
        Action(
            "passages",
            "collect",
            "Pile から、選んだ語を含む用例の候補を切り出す",
            passage_collection.collect_passages,
        ),
        Action(
            "passages",
            "remove-duplicates-and-split",
            "ほぼ同じ用例の候補を除き、元の文書の単位で学習用と評価用に分ける",
            passage_split.remove_duplicates_and_split,
        ),
        Action(
            "passages",
            "judge-word-sense",
            "候補の中で、語が意図した語義で使われているかを、LLM が出現ごとに判定する",
            word_sense_judgment.judge_word_sense,
        ),
        Action(
            "passages",
            "sample-for-review",
            "名詞・形容詞・ターゲット語ごとに、合格した用例を20件抜き出し、"
            "確認用の data/passages/review.csv を書き出す",
            passage_review.sample_for_review,
        ),
        Action(
            "passages",
            "finalize",
            "語ごとに LLM の判定の精度（合格とした候補のうち、人も 1 とした割合）を出し、"
            "合格した候補を、学習用・評価用の用例として書き出す",
            passage_finalization.finalize_passages,
            passage_finalization.add_arguments,
        ),
        Action(
            "assignments",
            "build",
            "run ごとに、どの語の用例を、どの語に置き換えて差し込むか"
            "（未知語の獲得では件数も）を決める",
            assignments.build_assignments,
            assignments.add_arguments,
        ),
        Action(
            "training-data",
            "build",
            "run ごとの学習データを作る: 割り当てのとおりに用例の語を置き換え、基データに差し込む",
            training_data.build_training_data,
            training_data.add_arguments,
        ),
        Action(
            "training",
            "write-run-settings",
            "run ごとの学習の設定ファイルを outputs/runs/<実験>/<モデル>/ に書き出す",
            run_settings.write_run_settings,
            run_settings.add_arguments,
        ),
        Action(
            "training",
            "run",
            "run の設定ファイルのとおりに継続事前学習を行う（torchrun から、GPU ごとに起動する）",
            continued_pretraining.run_training,
            continued_pretraining.add_arguments,
        ),
        Action(
            "evaluation-data",
            "extract-natural-word-sentences",
            "自然な語の用例から、課題の例と、プローブ（内部表現から語の性質を当てる分類器）の"
            "学習に使う文を取り出す",
            natural_word_sentences.extract_natural_word_sentences,
        ),
        Action(
            "evaluation-data",
            "build-probe-and-normalization-words",
            "プローブの学習に使う実在語と非語（実在しない文字列）と、未知語の指標を正規化する"
            "（既知の語の値を 1 にそろえる）ときの基準にする既知の語を作る",
            probe_and_normalization_words.build_probe_and_normalization_words,
        ),
        Action(
            "evaluation-data",
            "generate-minimal-pairs",
            "最小対（手がかりの1語だけが違い、ターゲット語の語義が変わる2文）を作る",
            minimal_pairs.generate_minimal_pairs,
        ),
        Action(
            "evaluation-data",
            "build-task-questions",
            "課題の問題を作る（言い換えの語は LLM が挙げる）",
            task_questions.build_task_questions,
        ),
        Action(
            "evaluation-data",
            "check-with-pretrained-models",
            "公開されている OLMo 2 の3モデルで、最小対の手がかりが語義を決めているかを確かめ、"
            "確認に通った最小対から一部を、確認用の data/evaluation/minimal_pairs_review.csv に"
            "書き出す",
            pretrained_model_checks.check_with_pretrained_models,
        ),
        Action(
            "evaluation-data",
            "finalize",
            "approved が 0 の最小対を除き、評価に使う最小対と課題の問題を書き出す",
            evaluation_finalization.finalize_evaluation_data,
            evaluation_finalization.add_arguments,
        ),
        Action(
            "unknown-word-acquisition",
            "train-real-word-probes",
            "公開されているモデルで、内部表現から実在語かどうかを当てるプローブを学習し、"
            "正解率を出す",
            real_word_probes.train_real_word_probes,
            real_word_probes.add_arguments,
        ),
        Action(
            "unknown-word-acquisition",
            "measure-indicators",
            "各 run のモデルで、未知語の3つの指標（形への慣れ・意味・語としての統合）を"
            "層ごとに測る",
            unknown_word_indicators.measure_indicators,
            unknown_word_indicators.add_arguments,
        ),
        Action(
            "unknown-word-acquisition",
            "analyze-acquisition-order",
            "指標を、差し込み0回の未知語が 0、既知の語が 1 になるように正規化し、"
            "3つの指標が獲得される順番を検定する",
            acquisition_order.analyze_acquisition_order,
            acquisition_order.add_arguments,
        ),
        Action(
            "unknown-word-acquisition",
            "make-report",
            "図と表を outputs/figures/unknown-word-acquisition/ に作る",
            unknown_word_report.make_report,
        ),
        Action(
            "sense-addition",
            "measure-sense-accuracy",
            "最小対の文で、ターゲット語の語義を使い分けられているかを測る",
            sense_accuracy.measure_sense_accuracy,
            sense_accuracy.add_arguments,
        ),
        Action(
            "sense-addition",
            "measure-head-and-mlp-effects",
            "各 head と MLP の出力を、最小対のもう一方の文での出力に入れ替え、語義の選択への効果を"
            "測る",
            head_and_mlp_effects.measure_head_and_mlp_effects,
            head_and_mlp_effects.add_arguments,
        ),
        Action(
            "sense-addition",
            "select-significant-heads-and-mlps",
            "効果の信頼区間を出し、有意な head と MLP を選ぶ",
            significant_heads_and_mlps.select_significant_heads_and_mlps,
            significant_heads_and_mlps.add_arguments,
        ),
        Action(
            "sense-addition",
            "train-attribute-probes",
            "選んだ head の出力から、名詞のカテゴリと品詞を当てるプローブを学習する",
            attribute_probes.train_attribute_probes,
            attribute_probes.add_arguments,
        ),
        Action(
            "sense-addition",
            "test-attribute-directions",
            "head の出力のうち属性を表す方向の成分を入れ替えて、属性を判定する head を確かめる",
            attribute_direction_patching.test_attribute_directions,
            attribute_direction_patching.add_arguments,
        ),
        Action(
            "sense-addition",
            "make-report",
            "図と表を outputs/figures/sense-addition/ に作る",
            sense_addition_report.make_report,
        ),
        Action(
            "dependence-on-category-judging-heads",
            "measure-dependence",
            "カテゴリを判定する head を無効化し、未知語と元の名詞でカテゴリの判定がどれだけ落ちるか"
            "を測る",
            category_dependence.measure_dependence,
            category_dependence.add_arguments,
        ),
        Action(
            "dependence-on-category-judging-heads",
            "fit-regression",
            "依存度を、意味と語としての統合の指標などで説明する回帰を当てはめる",
            dependence_regression.fit_regression,
            dependence_regression.add_arguments,
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
