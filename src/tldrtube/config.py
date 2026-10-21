"""Load the Ollama model name and temperature from tldrtube_config.toml."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from tldrtube.models import DEFAULT_MODEL

# Ollama's chat API uses 0.8 when the caller does not set a temperature.
DEFAULT_TEMPERATURE = 0.8
CONFIG_FILENAME = "tldrtube_config.toml"
_ALLOWED_KEYS = frozenset({"model", "temperature"})


class ConfigError(Exception):
    """tldrtube_config.toml is present but cannot be used."""


@dataclass(frozen=True)
class Settings:
    """Model name and sampling temperature for the Ollama client."""

    model: str
    temperature: float


def load_config(path: Path | None = None) -> Settings:
    """Read tldrtube_config.toml from the working directory.

    A missing file uses the built-in defaults. A present file may omit either
    key. Unknown keys, bad types, and invalid TOML are errors.
    """

    config_path = Path.cwd() / CONFIG_FILENAME if path is None else path
    if not config_path.is_file():
        return Settings(model=DEFAULT_MODEL, temperature=DEFAULT_TEMPERATURE)
    try:
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{config_path.name}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path.name} must contain settings.")
    unknown = sorted(set(raw) - _ALLOWED_KEYS)
    if unknown:
        names = ", ".join(unknown)
        label = "setting" if len(unknown) == 1 else "settings"
        raise ConfigError(f"{config_path.name}: unknown {label}: {names}.")
    return Settings(
        model=_model(raw.get("model", DEFAULT_MODEL), config_path.name),
        temperature=_temperature(raw.get("temperature", DEFAULT_TEMPERATURE), config_path.name),
    )


def _model(value: object, filename: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ConfigError(f"{filename}: model must be a non-empty string.")


def _temperature(value: object, filename: str) -> float:
    # bool is a subclass of int, so a TOML true would otherwise pass the numeric check.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{filename}: temperature must be a number.")
    if value < 0:
        raise ConfigError(f"{filename}: temperature must be 0 or greater.")
    return float(value)
