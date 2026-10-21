"""Loading tldrtube_config.toml."""

from pathlib import Path

import pytest

from tldrtube.config import DEFAULT_TEMPERATURE, ConfigError, load_config
from tldrtube.models import DEFAULT_MODEL


def test_missing_file_uses_defaults(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    settings = load_config()
    assert settings.model == DEFAULT_MODEL
    assert settings.temperature == DEFAULT_TEMPERATURE == 0.8


def test_omitted_keys_use_defaults(tmp_path: Path) -> None:
    model_only = tmp_path / "model.toml"
    model_only.write_text('model = "custom-model"\n', encoding="utf-8")
    settings = load_config(model_only)
    assert settings.model == "custom-model"
    assert settings.temperature == DEFAULT_TEMPERATURE

    temperature_only = tmp_path / "temperature.toml"
    temperature_only.write_text("temperature = 0.2\n", encoding="utf-8")
    settings = load_config(temperature_only)
    assert settings.model == DEFAULT_MODEL
    assert settings.temperature == 0.2


def test_file_sets_model_and_temperature(tmp_path: Path) -> None:
    path = tmp_path / "tldrtube_config.toml"
    path.write_text('model = "  other-model  "\ntemperature = 1\n', encoding="utf-8")
    settings = load_config(path)
    assert settings.model == "other-model"
    assert settings.temperature == 1.0


def test_invalid_config_is_rejected(tmp_path: Path) -> None:
    cases = {
        "model = [\n": "tldrtube_config.toml",
        "model = 3\n": "model must be a non-empty string",
        'model = ""\n': "model must be a non-empty string",
        'temperature = "hot"\n': "temperature must be a number",
        "temperature = true\n": "temperature must be a number",
        "temperature = -0.1\n": "temperature must be 0 or greater",
        "extra = 1\n": "unknown setting: extra",
    }
    for text, message in cases.items():
        path = tmp_path / "tldrtube_config.toml"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ConfigError, match=message):
            load_config(path)
