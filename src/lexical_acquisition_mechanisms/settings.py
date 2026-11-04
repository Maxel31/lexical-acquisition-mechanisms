"""設定ファイル settings.yaml の読み込みと検証。

実験の設定は、すべてリポジトリの最上位のディレクトリにある settings.yaml に書く。
項目は、その項目を使う処理を実装するときに足す。
"""

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
    # LLM に、カテゴリごとに名詞を挙げさせるときの、挙げる語の種類（種類ごとに1回ずつ挙げさせる）
    noun_listing_kinds: list[str] = Field(min_length=1)
    # 1回に挙げさせる名詞の数
    nouns_per_listing: int = Field(gt=0)
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


class Settings(SettingsSection):
    """settings.yaml の全体。"""

    # --model で指定する名前と、モデルの取得元
    models: dict[ModelName, ModelSource] = Field(min_length=1)
    llm: LlmSettings
    base_data: BaseDataSettings
    words: WordsSettings


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
