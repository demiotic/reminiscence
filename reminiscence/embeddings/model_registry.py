# reminiscence/embeddings/model_registry.py
"""Model registry - only defaults."""

from pathlib import Path
from typing import Optional
import yaml

from ..utils.logging import get_logger

logger = get_logger(__name__)

_defaults: Optional[dict] = None


def get_default_model(backend: str) -> str:
    """Get default model for backend."""
    global _defaults

    if _defaults is None:
        config_path = Path(__file__).parent / "models.yaml"

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                _defaults = data.get("default", {})
            logger.debug("defaults_loaded", defaults=_defaults)
        except Exception as e:
            logger.error("defaults_load_failed", error=str(e), exc_info=True)
            # Hardcoded fallback
            _defaults = {
                "fastembed": "intfloat/multilingual-e5-small",
                "sentence_transformers": "sentence-transformers/all-MiniLM-L6-v2",
            }

    default = _defaults.get(backend, "intfloat/multilingual-e5-small")
    return default
