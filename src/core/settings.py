"""Settings loading and validation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import yaml


class SettingsError(ValueError):
    """Raised when settings cannot be loaded or validated."""


@dataclass(frozen=True)
class LLMSettings:
    provider: str


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str


@dataclass(frozen=True)
class VectorStoreSettings:
    provider: str


@dataclass(frozen=True)
class RetrievalSettings:
    top_k: int


@dataclass(frozen=True)
class RerankSettings:
    enabled: bool


@dataclass(frozen=True)
class EvaluationSettings:
    enabled: bool


@dataclass(frozen=True)
class ObservabilitySettings:
    log_level: str


@dataclass(frozen=True)
class Settings:
    llm: LLMSettings
    embedding: EmbeddingSettings
    vector_store: VectorStoreSettings
    retrieval: RetrievalSettings
    rerank: RerankSettings
    evaluation: EvaluationSettings
    observability: ObservabilitySettings


def _require_mapping(data: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise SettingsError(f"Missing or invalid field: {key}")
    return value


def _require_str(data: Dict[str, Any], key: str, field_path: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SettingsError(f"Missing or invalid field: {field_path}")
    return value.strip()


def validate_settings(settings: Settings) -> None:
    """Validate required fields and basic value constraints."""
    if not settings.llm.provider:
        raise SettingsError("Missing or invalid field: llm.provider")
    if not settings.embedding.provider:
        raise SettingsError("Missing or invalid field: embedding.provider")
    if not settings.vector_store.provider:
        raise SettingsError("Missing or invalid field: vector_store.provider")
    if settings.retrieval.top_k <= 0:
        raise SettingsError("Missing or invalid field: retrieval.top_k")
    if not settings.observability.log_level:
        raise SettingsError("Missing or invalid field: observability.log_level")


def load_settings(path: Union[str, Path]) -> Settings:
    """Load YAML settings and validate required fields."""
    settings_path = Path(path)
    if not settings_path.exists():
        raise SettingsError(f"Settings file not found: {settings_path}")

    try:
        raw = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SettingsError(f"Failed to parse YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise SettingsError("Settings file must contain a mapping at root level")

    llm_raw = _require_mapping(raw, "llm")
    embedding_raw = _require_mapping(raw, "embedding")
    vector_store_raw = _require_mapping(raw, "vector_store")
    retrieval_raw = _require_mapping(raw, "retrieval")
    rerank_raw = _require_mapping(raw, "rerank")
    evaluation_raw = _require_mapping(raw, "evaluation")
    observability_raw = _require_mapping(raw, "observability")

    settings = Settings(
        llm=LLMSettings(provider=_require_str(llm_raw, "provider", "llm.provider")),
        embedding=EmbeddingSettings(
            provider=_require_str(embedding_raw, "provider", "embedding.provider")
        ),
        vector_store=VectorStoreSettings(
            provider=_require_str(vector_store_raw, "provider", "vector_store.provider")
        ),
        retrieval=RetrievalSettings(top_k=int(retrieval_raw.get("top_k", 5))),
        rerank=RerankSettings(enabled=bool(rerank_raw.get("enabled", False))),
        evaluation=EvaluationSettings(enabled=bool(evaluation_raw.get("enabled", False))),
        observability=ObservabilitySettings(
            log_level=_require_str(observability_raw, "log_level", "observability.log_level")
        ),
    )

    validate_settings(settings)
    return settings
