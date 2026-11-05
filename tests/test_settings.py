"""設定ファイルの読み込みと検証のテスト。"""

import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from lexical_acquisition_mechanisms.settings import Settings, SettingsError, load_settings

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def write_settings(directory: Path, text: str) -> Path:
    path = directory / "settings.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_repository_settings_are_valid() -> None:
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")

    assert {name: source.repository for name, source in settings.models.items()} == {
        "OLMo-2-0425-1B": "allenai/OLMo-2-0425-1B",
        "OLMo-2-1124-7B": "allenai/OLMo-2-1124-7B",
        "OLMo-2-1124-13B": "allenai/OLMo-2-1124-13B",
    }
    # 版は、ブランチの名前ではなくコミットで固定する
    assert all(
        re.fullmatch(r"[0-9a-f]{40}", source.revision) for source in settings.models.values()
    )


def test_revision_can_be_omitted() -> None:
    content = load_settings(REPOSITORY_ROOT / "settings.yaml").model_dump()
    content["models"] = {"tiny": {"repository": "./tiny"}}
    content["training"]["learning_rate_by_model"]["tiny"] = 1e-4
    content["training"]["sequences_per_device_by_model"]["tiny"] = 1

    assert Settings.model_validate(content).models["tiny"].revision is None


def test_category_names_must_be_unique() -> None:
    content = load_settings(REPOSITORY_ROOT / "settings.yaml").model_dump()
    content["words"]["categories"][1]["name"] = "food"

    with pytest.raises(ValueError, match="カテゴリの名前が重なっています"):
        Settings.model_validate(content)


def test_bounds_are_explained(tmp_path: Path) -> None:
    text = (REPOSITORY_ROOT / "settings.yaml").read_text(encoding="utf-8")
    path = write_settings(
        tmp_path,
        text.replace("maximum_concurrent_requests: 1024", "maximum_concurrent_requests: 0"),
    )

    with pytest.raises(SettingsError, match="llm.maximum_concurrent_requests: 0 より大きい数"):
        load_settings(path)


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="がありません。リポジトリの最上位"):
        load_settings(tmp_path / "settings.yaml")


@pytest.mark.parametrize(
    ("text", "expected_message"),
    [
        (
            "models:\n  tiny:\n    repository: ./tiny\nmodel_names: []\n",
            "- model_names: 知らない項目です",
        ),
        (
            "models:\n  tiny:\n    repository: ./tiny\n    branch: main\n",
            "- models.tiny.branch: 知らない項目です",
        ),
        (
            "models:\n  tiny:\n    revision: main\n",
            "- models.tiny.repository: 必要な項目がありません",
        ),
        ("{}\n", "- models: 必要な項目がありません"),
        ("models: {}\n", "- models: 1つ以上書いてください"),
        ("models: [allenai/OLMo-2-1124-7B]\n", "- models: 「名前: 値」の組を並べてください"),
        ("models:\n  tiny: ./tiny\n", "- models.tiny: 「項目名: 値」の組を並べてください"),
        ("models:\n  tiny:\n    repository: 3\n", "- models.tiny.repository: 文字列にしてください"),
        (
            "models: {'tiny/model': {repository: ./tiny}}\n",
            "- models の名前 'tiny/model': 使えない文字があります",
        ),
    ],
)
def test_invalid_content_is_explained(tmp_path: Path, text: str, expected_message: str) -> None:
    path = write_settings(tmp_path, text)

    with pytest.raises(SettingsError) as error:
        load_settings(path)

    assert expected_message in str(error.value)


@pytest.mark.parametrize(
    ("text", "expected_message"),
    [
        ("models: [unclosed\n", "YAML として読めません"),
        ("models: {a: {repository: ./a}}\nmodels: {b: {repository: ./b}}\n", "項目 'models' が2回"),
        ("models:\n  a: {repository: ./a}\n  a: {repository: ./b}\n", "項目 'a' が2回"),
        ("- models\n", "最上位は、項目名と値の組にしてください"),
        ("", "最上位は、項目名と値の組にしてください"),
    ],
)
def test_malformed_file_is_explained(tmp_path: Path, text: str, expected_message: str) -> None:
    path = write_settings(tmp_path, text)

    with pytest.raises(SettingsError) as error:
        load_settings(path)

    assert expected_message in str(error.value)


def test_yaml_error_points_at_the_file(tmp_path: Path) -> None:
    path = write_settings(tmp_path, "models: [unclosed\n")

    with pytest.raises(SettingsError) as error:
        load_settings(path)

    assert f'in "{path}", line' in str(error.value)


def test_fields_cannot_be_reassigned_after_loading() -> None:
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")

    with pytest.raises(ValueError):
        settings.models = {}


def test_exposure_levels_must_be_eight_increasing_numbers() -> None:
    content = load_settings(REPOSITORY_ROOT / "settings.yaml").model_dump()
    content["unknown_word_acquisition"]["exposure_levels"] = [0, 10, 1]

    with pytest.raises(ValueError, match="0 以上の数を8つ、小さい順に"):
        Settings.model_validate(content)


def test_training_values_are_needed_for_every_model() -> None:
    content = yaml.safe_load((REPOSITORY_ROOT / "settings.yaml").read_text(encoding="utf-8"))
    del content["training"]["sequences_per_device_by_model"]["OLMo-2-1124-13B"]

    with pytest.raises(ValidationError, match="sequences_per_device_by_model"):
        Settings.model_validate(content)


def test_checkpoint_fractions_are_inside_training() -> None:
    content = yaml.safe_load((REPOSITORY_ROOT / "settings.yaml").read_text(encoding="utf-8"))
    content["training"]["checkpoint_fractions"] = [25, 50]

    with pytest.raises(ValidationError, match="0 より大きく 1 より小さい"):
        Settings.model_validate(content)


def test_passages_per_added_sense_is_at_most_the_largest_exposure() -> None:
    content = yaml.safe_load((REPOSITORY_ROOT / "settings.yaml").read_text(encoding="utf-8"))
    content["sense_addition"]["passages_per_added_sense"] = 20000

    with pytest.raises(ValidationError, match="露出回数の最大（10000）以下"):
        Settings.model_validate(content)
