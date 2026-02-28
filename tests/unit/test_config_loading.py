"""Tests for settings loading and startup validation."""

from pathlib import Path
import sys

import pytest

# Ensure tests are runnable via both `python -m pytest` and `.venv/bin/pytest`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main
from src.core.settings import SettingsError, load_settings


VALID_SETTINGS_YAML = """\
llm:
  provider: openai
embedding:
  provider: openai
vector_store:
  provider: chroma
retrieval:
  top_k: 5
rerank:
  enabled: false
evaluation:
  enabled: false
observability:
  log_level: INFO
"""


def test_load_settings_success(tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(VALID_SETTINGS_YAML, encoding="utf-8")

    settings = load_settings(settings_file)

    assert settings.llm.provider == "openai"
    assert settings.embedding.provider == "openai"
    assert settings.vector_store.provider == "chroma"
    assert settings.retrieval.top_k == 5
    assert settings.observability.log_level == "INFO"


def test_load_settings_missing_field_has_readable_error(tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(
        VALID_SETTINGS_YAML.replace(
            "embedding:\n  provider: openai",
            "embedding:\n  provider: ",
        ),
        encoding="utf-8",
    )

    with pytest.raises(SettingsError, match="embedding.provider"):
        load_settings(settings_file)


def test_main_returns_zero_when_settings_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "settings.yaml").write_text(VALID_SETTINGS_YAML, encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    assert main.main() == 0


def test_main_fail_fast_on_invalid_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "settings.yaml").write_text(
        VALID_SETTINGS_YAML.replace("provider: openai", "provider: ", 1),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    exit_code = main.main()
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "llm.provider" in captured.err
