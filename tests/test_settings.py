"""設定ファイルの読み込みと検証のテスト。"""

from pathlib import Path

import pytest

from lexical_acquisition_mechanisms.settings import SettingsError, load_settings

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def write_settings(directory: Path, text: str) -> Path:
    path = directory / "settings.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_repository_settings_are_valid() -> None:
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")

    assert settings.models == {
        "OLMo-2-0425-1B": "allenai/OLMo-2-0425-1B",
        "OLMo-2-1124-7B": "allenai/OLMo-2-1124-7B",
        "OLMo-2-1124-13B": "allenai/OLMo-2-1124-13B",
    }


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="がありません。リポジトリの最上位"):
        load_settings(tmp_path / "settings.yaml")


@pytest.mark.parametrize(
    ("text", "expected_message"),
    [
        ("models:\n  tiny: ./tiny\nmodel_names: []\n", "- model_names: 知らない項目です"),
        ("{}\n", "- models: 必要な項目がありません"),
        ("models: {}\n", "- models: 1つ以上書いてください"),
        ("models: [allenai/OLMo-2-1124-7B]\n", "- models: 「名前: 値」の組を並べてください"),
        ("models:\n  tiny: 3\n", "- models.tiny: 文字列にしてください"),
        (
            "models: {'tiny/model': ./tiny}\n",
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
        ("models:\n  a: ./a\nmodels:\n  b: ./b\n", "項目 'models' が2回書かれています"),
        ("models:\n  a: ./a\n  a: ./b\n", "項目 'a' が2回書かれています"),
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
