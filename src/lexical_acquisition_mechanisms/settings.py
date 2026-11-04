"""設定ファイル settings.yaml の読み込みと検証。

実験の設定は、すべてリポジトリの最上位のディレクトリにある settings.yaml に書く。
項目は、その項目を使う処理を実装するときに足す。
"""

import itertools
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
)

SETTINGS_FILE_NAME = "settings.yaml"

# 出力のパス（例: outputs/runs/<実験>/<モデル>/）に使うので、パスに使える文字に限る
ModelName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
MODEL_NAME_RULE = "出力のパスに使うので、英数字と . _ - だけにし、英数字で始めてください"


class SettingsError(Exception):
    """設定ファイルが読めない、または内容が正しくない。"""


class SettingsSection(BaseModel):
    """設定の節に共通の決まり。

    知らない項目（書き間違い）と、読み込んだ後に項目へ代入することを許さない
    （辞書の中身の書き換えまでは防がない）。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelSource(SettingsSection):
    """モデルの取得元。"""

    # Hugging Face のリポジトリ（またはローカルのディレクトリ）
    repository: str
    # リポジトリの版（ブランチ・タグ・コミット）。書かなければ、既定のブランチ（main）
    revision: str | None = None


class LlmSettings(SettingsSection):
    """語の候補の生成や用例の判定に使う LLM（vLLM のサーバ）。"""

    # OpenAI 互換の API の URL
    server_url: str
    # サーバで動かすモデルの名前
    model: str
    # 同時に送る問い合わせの数の上限
    maximum_concurrent_requests: int = Field(gt=0)


class BaseDataSettings(SettingsSection):
    """基データ（OLMo 2 の事前学習データの、学習順で先頭の部分）と検証データ。"""

    # OLMo 2 のステージ1の学習設定（学習データの一覧と seed）の URL。版を固定する
    pretraining_configuration_url: str
    # 学習順の先頭から取る系列（4,096トークン）の数
    number_of_base_sequences: int = Field(gt=0)
    # その直後から取る、検証データの系列の数
    number_of_validation_sequences: int = Field(gt=0)


class CategorySettings(SettingsSection):
    """名詞のカテゴリ（食べ物・動物などの意味の分類）。"""

    # カテゴリの名前。課題2（属性の判定）の選択肢の語でもある
    name: str = Field(pattern=r"^[a-z]+$")
    # LLM に、このカテゴリの名詞を挙げさせるときの説明
    description: str
    # このカテゴリの語義の上位語（WordNet の synset の名前。例: food.n.01）
    wordnet_roots: list[str] = Field(min_length=1)


class WordsSettings(SettingsSection):
    """実験で操作する語と、自然な語の選び方。"""

    categories: list[CategorySettings] = Field(min_length=2)
    # LLM に語を挙げさせるときの、挙げる語の種類
    # （名詞はカテゴリごと、形容詞は全体で、種類ごとに1回ずつ挙げさせる）
    listing_kinds: list[str] = Field(min_length=1)
    # 1回に挙げさせる名詞と形容詞の数
    nouns_per_listing: int = Field(gt=0)
    adjectives_per_listing: int = Field(gt=0)
    # 頻度と使われ方を見積もるときに読む、Pile の先頭の断片の数（全部で1,650個）
    pile_sample_shards: int = Field(gt=0, le=1650)
    # 使われ方を LLM が判定する、語ごとの見本の数
    usage_examples_per_word: int = Field(gt=0)
    # 見本を選ぶときなどに使う seed
    seed: int

    @field_validator("categories")
    @classmethod
    def _names_are_unique(cls, categories: list[CategorySettings]) -> list[CategorySettings]:
        names = [category.name for category in categories]
        if len(set(names)) != len(names):
            raise ValueError(f"カテゴリの名前が重なっています: {names}")
        return categories


class UnknownWordSettings(SettingsSection):
    """未知語（モデルが見たことのない文字列）の候補の作り方。"""

    # 一度も現れないことを確かめる、infini-gram の索引
    corpus_indexes: list[str] = Field(min_length=1)
    # ARC 非語データベースの、同音の実在語がない非語の TSV の URL
    arc_nonwords_url: str
    # トークン数（2・3）と綴りの近さ（near・far）の組ごとに集める、一度も現れない候補の数
    candidates_per_group: int = Field(gt=0)


class WordSelectionSettings(SettingsSection):
    """候補から語を選ぶときの条件（words select）。"""

    # Pile の見本で、意図した語義（形容詞は、形容詞として）で使われていた割合の下限
    minimum_usage_share: float = Field(ge=0, le=1)
    # その割合を当てにできるように、LLM が判定できた見本の数の下限
    minimum_judged_usages: int = Field(gt=0)
    # Pile で取れる用例の見積もりに、意図した使われ方の割合を掛けた数が、必要な数の何倍以上か
    passage_margin: float = Field(ge=1)
    # 形容詞72語の、どのカテゴリとの結びつきの評定（1〜7）も、これ以下にする
    maximum_category_association: int = Field(ge=1, le=7)


class PassageSettings(SettingsSection):
    """用例（管理対象の語を含む、Pile から切り出した文章）の作り方。"""

    # 用例の長さ（トークン数）
    tokens_per_passage: int = Field(gt=0)
    # 1つの文書から取る用例の数の上限
    maximum_passages_per_document: int = Field(gt=0)
    # 語ごとの評価用の用例の数
    evaluation_passages_per_word: int = Field(gt=0)


class UnknownWordAcquisitionSettings(SettingsSection):
    """未知語の獲得。"""

    # 露出回数（1 run の学習データに差し込む、1つの未知語の用例の数）の水準。
    # 8次の直交ラテン方格で割り当てるので、8つにする
    exposure_levels: list[int]

    @field_validator("exposure_levels")
    @classmethod
    def _eight_increasing_levels(cls, levels: list[int]) -> list[int]:
        increasing = all(lower < upper for lower, upper in itertools.pairwise(levels))
        if len(levels) != 8 or not increasing or levels[0] < 0:
            raise ValueError(
                f"露出回数の水準は、0 以上の数を8つ、小さい順に並べてください: {levels}"
            )
        return levels


class SenseAdditionSettings(SettingsSection):
    """語義の追加。"""

    # 露出数 K（1 run の学習データに差し込む、1つの donor の用例の数）
    exposures_per_donor: int = Field(gt=0)


class EvaluationSettings(SettingsSection):
    """評価データ。"""

    # 自然な語（実験で操作しない実在語）ごとの、課題の例やプローブの学習に使う文の数
    sentences_per_natural_word: int = Field(gt=0)
    # 中立文脈（語の意味を示さない文の書き出し）の候補。前から順に、条件に合うものを使う
    neutral_context_candidates: list[str] = Field(min_length=1)
    # 使う中立文脈の数（文脈なしを除く）
    number_of_neutral_contexts: int = Field(gt=0)
    # 中立文脈の条件: 自然な名詞の対数確率のカテゴリごとの平均（全カテゴリの平均との差）が、
    # 全候補での平均から、どのカテゴリでもこれ（自然対数）より離れない
    maximum_neutral_context_deviation: float = Field(gt=0)


class Settings(SettingsSection):
    """settings.yaml の全体。"""

    # --model で指定する名前と、モデルの取得元
    models: dict[ModelName, ModelSource] = Field(min_length=1)
    llm: LlmSettings
    base_data: BaseDataSettings
    words: WordsSettings
    unknown_words: UnknownWordSettings
    word_selection: WordSelectionSettings
    passages: PassageSettings
    unknown_word_acquisition: UnknownWordAcquisitionSettings
    sense_addition: SenseAdditionSettings
    evaluation: EvaluationSettings


def load_settings(path: Path = Path(SETTINGS_FILE_NAME)) -> Settings:
    """設定ファイルを読み込み、検証する。"""
    try:
        with path.open(encoding="utf-8") as file:
            content = yaml.load(file, Loader=_LoaderRejectingDuplicateKeys)
    except FileNotFoundError as error:
        raise SettingsError(
            f"設定ファイル {path} がありません。"
            "リポジトリの最上位のディレクトリで実行してください。"
        ) from error
    except yaml.YAMLError as error:
        raise SettingsError(f"設定ファイル {path} を YAML として読めません:\n{error}") from error
    if not isinstance(content, dict):
        raise SettingsError(f"設定ファイル {path} の最上位は、項目名と値の組にしてください。")
    try:
        return Settings.model_validate(content)
    except ValidationError as error:
        raise SettingsError(_describe_problems(path, error)) from error


class _LoaderRejectingDuplicateKeys(yaml.SafeLoader):
    """同じ項目名が2回あるとエラーにする（PyYAML の既定では、後の値が黙って使われる）。"""


def _construct_mapping_rejecting_duplicate_keys(
    loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict:
    keys = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in keys:
            raise yaml.constructor.ConstructorError(
                None, None, f"項目 {key!r} が2回書かれています", key_node.start_mark
            )
        keys.add(key)
    return loader.construct_mapping(node, deep=deep)


_LoaderRejectingDuplicateKeys.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping_rejecting_duplicate_keys
)


# pydantic の英語の説明を、何を直せばよいかが分かる日本語にする（ない種類は英語のまま出す）。
# {gt} などには、pydantic が示す境界の値が入る
_PROBLEM_DESCRIPTIONS = {
    "missing": "必要な項目がありません",
    "extra_forbidden": "知らない項目です（書き間違いでないか確かめてください）",
    "dict_type": "「名前: 値」の組を並べてください",
    "model_type": "「項目名: 値」の組を並べてください",
    "string_type": "文字列にしてください",
    "too_short": "1つ以上書いてください",
    "int_parsing": "整数にしてください",
    "greater_than": "{gt} より大きい数にしてください",
    "greater_than_equal": "{ge} 以上の数にしてください",
    "less_than_equal": "{le} 以下の数にしてください",
    "string_pattern_mismatch": f"使えない文字があります（{MODEL_NAME_RULE}）",
}


def _describe_problems(path: Path, error: ValidationError) -> str:
    lines = [f"設定ファイル {path} の内容が正しくありません:"]
    for problem in error.errors():
        description = _PROBLEM_DESCRIPTIONS.get(problem["type"])
        if description is None:
            description = problem["msg"]
        else:
            description = description.format(**problem.get("ctx", {}))
        lines.append(f"- {_describe_location(problem['loc'])}: {description}")
    return "\n".join(lines)


def _describe_location(parts: tuple[int | str, ...]) -> str:
    """項目の場所を、models.tiny のように書く。名前（辞書のキー）の誤りは、その名前を示す。"""
    if parts and parts[-1] == "[key]":
        return f"{'.'.join(str(part) for part in parts[:-2])} の名前 {parts[-2]!r}"
    return ".".join(str(part) for part in parts)
