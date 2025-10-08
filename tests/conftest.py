"""Shared fixtures for Reminiscence tests."""

import pytest
import tempfile
import shutil
import structlog
import logging
from pathlib import Path

from reminiscence import Reminiscence, ReminiscenceConfig
from reminiscence.cache import CacheOperations
from reminiscence.embeddings import create_embedder
from reminiscence.storage import create_storage_backend
from reminiscence.eviction import create_eviction_policy
from reminiscence.metrics import CacheMetrics


# ============================================================================
# SINGLETON CLEANUP FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def clear_singletons():
    """
    Clear singleton instances between tests for isolation.

    This ensures each test starts with a clean slate for:
    - Storage backends (LanceDBBackend)
    - OpenTelemetry exporters

    autouse=True means this runs automatically for every test.
    """
    yield

    # Cleanup after each test
    from reminiscence.storage.lancedb import LanceDBBackend
    from reminiscence.metrics.exporters import OpenTelemetryExporter

    LanceDBBackend._clear_instances()
    OpenTelemetryExporter._clear_instances()


# ============================================================================
# STRUCTLOG RESET FIXTURES
# ============================================================================


@pytest.fixture
def reset_config():
    """Reset configuration singleton to force reload."""
    from reminiscence.config import ReminiscenceConfig

    if hasattr(ReminiscenceConfig, "_instance"):
        delattr(ReminiscenceConfig, "_instance")
    yield


@pytest.fixture
def reset_structlog():
    """
    Reset structlog configuration before and after each test.

    This ensures complete isolation between tests that use different
    logging configurations (JSON vs text).
    """
    # Reset before test
    structlog.reset_defaults()

    # Clear any cached loggers
    structlog._config._CONFIG = structlog._config._Configuration()

    # Reset standard logging
    logging.root.handlers = []
    logging.root.setLevel(logging.WARNING)

    yield

    # Reset after test
    structlog.reset_defaults()
    structlog._config._CONFIG = structlog._config._Configuration()
    logging.root.handlers = []


@pytest.fixture
def json_logging_env(reset_structlog, reset_config, monkeypatch):
    """
    Configure environment for JSON logging tests.

    Sets REMINISCENCE_JSON_LOGS=true and REMINISCENCE_LOG_LEVEL=INFO
    before any code runs, ensuring proper structlog configuration.
    """
    monkeypatch.setenv("REMINISCENCE_JSON_LOGS", "true")
    monkeypatch.setenv("REMINISCENCE_LOG_LEVEL", "INFO")

    # Force configuration reload
    from reminiscence.utils.logging import configure_logging

    configure_logging(log_level="INFO", json_logs=True)

    yield


@pytest.fixture
def text_logging_env(reset_structlog, reset_config, monkeypatch):
    """
    Configure environment for text logging tests.

    Sets REMINISCENCE_JSON_LOGS=false and REMINISCENCE_LOG_LEVEL=INFO
    before any code runs, ensuring proper structlog configuration.
    """
    monkeypatch.setenv("REMINISCENCE_JSON_LOGS", "false")
    monkeypatch.setenv("REMINISCENCE_LOG_LEVEL", "INFO")

    # Force configuration reload
    from reminiscence.utils.logging import configure_logging

    configure_logging(log_level="INFO", json_logs=False)

    yield


# ============================================================================
# DIRECTORY FIXTURES
# ============================================================================


@pytest.fixture
def temp_cache_dir():
    """Create temporary directory for disk cache."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


# ============================================================================
# CONFIG FIXTURES
# ============================================================================


@pytest.fixture
def memory_config():
    """Configuration for in-memory cache (fast for tests)."""
    return ReminiscenceConfig(
        db_uri="memory://",
        similarity_threshold=0.75,
        enable_metrics=True,
        log_level="WARNING",
        ttl_seconds=None,
    )


@pytest.fixture
def disk_config(temp_cache_dir):
    """Configuration for disk-based cache."""
    return ReminiscenceConfig(
        db_uri=str(Path(temp_cache_dir) / "test_cache.db"),
        similarity_threshold=0.75,
        enable_metrics=True,
        log_level="WARNING",
        ttl_seconds=None,
    )


# ============================================================================
# REMINISCENCE FIXTURES - High-level API
# ============================================================================


@pytest.fixture(scope="session")
def reminiscence_memory_session():
    """
    Shared in-memory Reminiscence instance (loaded once per session).

    Uses scope="session" to load the embeddings model only once
    and reuse it across all tests, improving performance.
    """
    config = ReminiscenceConfig(
        db_uri="memory://",
        similarity_threshold=0.75,
        enable_metrics=True,
        log_level="WARNING",
        ttl_seconds=None,
    )
    return Reminiscence(config)


@pytest.fixture
def reminiscence_memory(reminiscence_memory_session):
    """
    Clean in-memory Reminiscence instance for each test.

    Reuses the global session instance but clears cache and resets
    metrics before each test for speed.
    """
    # Clear cache and reset metrics before each test
    reminiscence_memory_session.clear()
    yield reminiscence_memory_session


@pytest.fixture
def reminiscence_disk(disk_config):
    """Disk-based Reminiscence instance (new for each test)."""
    return Reminiscence(disk_config)


# ============================================================================
# CACHE OPERATIONS FIXTURES - Low-level API
# ============================================================================


@pytest.fixture(scope="module")
def cache_ops_session():
    """
    Shared CacheOperations instance for entire test module.

    More lightweight than Reminiscence for testing core cache logic.
    The model is loaded once per module for performance.

    Use this when you need direct access to CacheOperations internals
    or want to test with custom configurations.
    """
    config = ReminiscenceConfig(
        db_uri="memory://",
        similarity_threshold=0.75,
        enable_metrics=True,
        log_level="WARNING",
    )

    embedder = create_embedder(config)
    storage = create_storage_backend(config, embedder.embedding_dim)
    eviction = create_eviction_policy(config.eviction_policy)
    metrics = CacheMetrics()

    return CacheOperations(
        storage=storage,
        embedder=embedder,
        eviction=eviction,
        config=config,
        metrics=metrics,
    )


@pytest.fixture
def cache_ops(cache_ops_session):
    """
    Clean CacheOperations instance for each test.

    Reuses the same session instance but clears storage and metrics
    before each test for isolation.

    This is faster than creating a new instance because the embedding
    model is already loaded.
    """
    # Clear storage and reset metrics
    cache_ops_session.storage.clear()
    cache_ops_session.metrics.reset()
    yield cache_ops_session


# ============================================================================
# TEST DATA FIXTURES
# ============================================================================


@pytest.fixture
def sample_queries():
    """Test queries with different similarity levels."""
    return {
        "identical": ["What is Python?", "What is Python?"],
        "similar": [
            "What is Python?",
            "Explain me Python",
            "Can you describe Python?",
        ],
        "different": [
            "What is Python?",
            "How can I install Cargo?",
            "Give me a cheese cake recipe",
        ],
    }


@pytest.fixture
def sample_contexts():
    """Test contexts for different agents."""
    return {
        "llm_gpt4": {"agent": "llm", "model": "gpt-4"},
        "llm_claude": {"agent": "llm", "model": "claude"},
        "sql_prod": {"agent": "sql", "db": "prod"},
        "sql_dev": {"agent": "sql", "db": "dev"},
    }
