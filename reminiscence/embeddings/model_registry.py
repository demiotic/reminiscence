"""Shared model registry for all embedding backends."""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml

from ..utils.logging import get_logger

logger = get_logger(__name__)

_registry: Optional[Dict[str, Any]] = None


def get_model_registry() -> Dict[str, Any]:
    """Load model registry from YAML file (cached)."""
    global _registry

    if _registry is None:
        config_path = Path(__file__).parent / "models.yaml"

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                _registry = yaml.safe_load(f)
            logger.debug("model_registry_loaded", path=str(config_path))
        except Exception as e:
            logger.error("model_registry_load_failed", error=str(e), exc_info=True)
            _registry = {
                "sentence_transformers": {},
                "fastembed": {},
                "default": {
                    "fastembed": "multilingual-e5-small",
                    "sentence_transformers": "all-MiniLM-L6-v2",
                },
            }

    return _registry


def get_default_model(backend: str) -> str:
    """Get default model name for a backend."""
    registry = get_model_registry()
    defaults = registry.get("default", {})
    return defaults.get(backend, "multilingual-e5-small")


def get_model_info(backend: str, model_name: str) -> Optional[Dict[str, Any]]:
    """
    Get model info from registry.

    Args:
        backend: "sentence_transformers" or "fastembed"
        model_name: Short name or full name

    Returns:
        Model info dict or None if not found
    """
    registry = get_model_registry()

    # Check in backend's models by short name
    backend_models = registry.get(backend, {})
    if model_name in backend_models:
        return backend_models[model_name]

    # Try full_name match
    for name, info in backend_models.items():
        if info.get("full_name") == model_name:
            return info

    # Try cross-backend (sentence-transformers models work in fastembed)
    if backend == "fastembed":
        st_models = registry.get("sentence_transformers", {})
        if model_name in st_models:
            return st_models[model_name]

        for name, info in st_models.items():
            if info.get("full_name") == model_name:
                return info

    logger.warning("model_not_in_registry", backend=backend, model=model_name)
    return None
