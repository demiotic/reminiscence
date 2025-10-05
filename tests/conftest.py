"""Fixtures compartidos para tests de Memora."""

import pytest
import tempfile
import shutil
from pathlib import Path

from memora import Memora, CacheConfig


@pytest.fixture
def temp_cache_dir():
    """Crea directorio temporal para caché en disco."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def memory_config():
    """Configuración para caché en memoria (rápido para tests)."""
    return CacheConfig(
        db_uri="memory://",
        similarity_threshold=0.75,
        enable_metrics=True,
        log_level="WARNING",  # Silenciar logs en tests
        ttl_seconds=None,
    )


@pytest.fixture
def disk_config(temp_cache_dir):
    """Configuración para caché en disco."""
    return CacheConfig(
        db_uri=str(Path(temp_cache_dir) / "test_cache.db"),
        similarity_threshold=0.75,
        enable_metrics=True,
        log_level="WARNING",
        ttl_seconds=None,
    )


@pytest.fixture
def memora_memory(memory_config):
    """Instancia de Memora en memoria."""
    return Memora(memory_config)


@pytest.fixture
def memora_disk(disk_config):
    """Instancia de Memora en disco."""
    return Memora(disk_config)


@pytest.fixture
def sample_queries():
    """Queries de prueba con diferentes niveles de similitud."""
    return {
        "identical": ["¿Qué es Python?", "¿Qué es Python?"],
        "similar": [
            "¿Qué es Python?",
            "Explica qué es Python",
            "¿Puedes describir Python?",
        ],
        "different": ["¿Qué es Python?", "¿Cómo instalo Node.js?", "Receta de paella"],
    }


@pytest.fixture
def sample_contexts():
    """Contextos de prueba."""
    return {
        "llm_gpt4": {"agent": "llm", "model": "gpt-4"},
        "llm_claude": {"agent": "llm", "model": "claude"},
        "sql_prod": {"agent": "sql", "db": "prod"},
        "sql_dev": {"agent": "sql", "db": "dev"},
    }
