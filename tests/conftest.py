"""Shared fixtures for Reminiscence tests."""

import pytest
import tempfile
import shutil
import structlog
import logging
import subprocess
import requests
import time
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
    structlog.reset_defaults()

    structlog._config._CONFIG = structlog._config._Configuration()

    logging.root.handlers = []
    logging.root.setLevel(logging.WARNING)

    yield

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
        otel_enabled=False,  # Explicitly disable OTEL
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
        otel_enabled=False,  # Explicitly disable OTEL
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
        otel_enabled=False,
    )
    return Reminiscence(config)


@pytest.fixture(scope="module")
def reminiscence_memory_module():
    """
    Shared Reminiscence instance for entire module (loads model once).

    Uses scope="module" to load embeddings model only once per test module,
    dramatically improving test performance (load time ~2s saved per test).
    """
    config = ReminiscenceConfig(
        db_uri="memory://",
        similarity_threshold=0.75,
        enable_metrics=True,
        log_level="WARNING",
        otel_enabled=False,
    )
    cache = Reminiscence(config)

    return cache


@pytest.fixture
def reminiscence(reminiscence_memory_module):
    """
    Clean cache for each test, but reuses the same model.

    This fixture:
    1. Reuses the module-scoped instance (no model reload)
    2. Clears cache before each test (isolation)
    3. Resets metrics before each test (clean counters)

    Result: Fast tests (~50ms) with proper isolation.
    """
    reminiscence_memory_module.clear()

    if reminiscence_memory_module.metrics:
        reminiscence_memory_module.metrics.reset()

    yield reminiscence_memory_module


@pytest.fixture
def reminiscence_disk(disk_config):
    """Disk-based Reminiscence instance (new for each test)."""
    return Reminiscence(disk_config)


# ============================================================================
# CACHE OPERATIONS FIXTURES - Low-level API
# ============================================================================


@pytest.fixture(scope="module")
def cache_ops_module():
    """Shared CacheOperations for module (loads model once)."""
    config = ReminiscenceConfig(
        db_uri="memory://",
        similarity_threshold=0.75,
        enable_metrics=True,
        log_level="WARNING",
        max_entries=10,
        eviction_policy="fifo",
        otel_enabled=False,
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
def cache_ops(cache_ops_module):
    """Clean CacheOps for each test, reuses model."""
    cache_ops_module.storage.clear()
    cache_ops_module.metrics.reset()

    cache_ops_module.eviction = create_eviction_policy(
        cache_ops_module.config.eviction_policy
    )

    yield cache_ops_module


# ============================================================================
# EVICTION POLICY FIXTURES
# ============================================================================


@pytest.fixture(scope="module")
def fifo_ops_module():
    """FIFO cache operations (model loaded once per module)."""
    config = ReminiscenceConfig(
        db_uri="memory://",
        max_entries=3,
        eviction_policy="fifo",
        enable_metrics=True,
        similarity_threshold=0.95,
        log_level="WARNING",
        otel_enabled=False,
    )

    embedder = create_embedder(config)
    storage = create_storage_backend(config, embedder.embedding_dim)
    eviction = create_eviction_policy("fifo")
    metrics = CacheMetrics()

    return CacheOperations(storage, embedder, eviction, config, metrics)


@pytest.fixture
def fifo_ops(fifo_ops_module):
    """Clean FIFO ops for each test, reuses model."""
    fifo_ops_module.storage.clear()
    fifo_ops_module.metrics.reset()
    fifo_ops_module.eviction = create_eviction_policy("fifo")
    yield fifo_ops_module


@pytest.fixture(scope="module")
def lru_ops_module():
    """LRU cache operations (model loaded once per module)."""
    config = ReminiscenceConfig(
        db_uri="memory://",
        max_entries=3,
        eviction_policy="lru",
        enable_metrics=True,
        similarity_threshold=0.95,
        log_level="WARNING",
        otel_enabled=False,
    )

    embedder = create_embedder(config)
    storage = create_storage_backend(config, embedder.embedding_dim)
    eviction = create_eviction_policy("lru")
    metrics = CacheMetrics()

    return CacheOperations(storage, embedder, eviction, config, metrics)


@pytest.fixture
def lru_ops(lru_ops_module):
    """Clean LRU ops for each test, reuses model."""
    lru_ops_module.storage.clear()
    lru_ops_module.metrics.reset()
    lru_ops_module.eviction = create_eviction_policy("lru")
    yield lru_ops_module


@pytest.fixture(scope="module")
def lfu_ops_module():
    """LFU cache operations (model loaded once per module)."""
    config = ReminiscenceConfig(
        db_uri="memory://",
        max_entries=3,
        eviction_policy="lfu",
        enable_metrics=True,
        similarity_threshold=0.95,
        log_level="WARNING",
        otel_enabled=False,
    )

    embedder = create_embedder(config)
    storage = create_storage_backend(config, embedder.embedding_dim)
    eviction = create_eviction_policy("lfu")
    metrics = CacheMetrics()

    return CacheOperations(storage, embedder, eviction, config, metrics)


@pytest.fixture
def lfu_ops(lfu_ops_module):
    """Clean LFU ops for each test, reuses model."""
    lfu_ops_module.storage.clear()
    lfu_ops_module.metrics.reset()
    lfu_ops_module.eviction = create_eviction_policy("lfu")
    yield lfu_ops_module


@pytest.fixture
def ops_factory():
    """
    Factory fixture for creating CacheOperations with custom config.

    Use this for tests that need non-standard max_entries or other config.

    Usage:
        def test_something(ops_factory):
            ops = ops_factory("fifo", max_entries=5)
            ops.store(...)
    """

    def _create(eviction_policy: str = "fifo", max_entries: int = 3, **kwargs):
        # Ensure OTEL is disabled unless explicitly enabled
        kwargs.setdefault("otel_enabled", False)

        config = ReminiscenceConfig(
            db_uri="memory://",
            max_entries=max_entries,
            eviction_policy=eviction_policy,
            enable_metrics=True,
            similarity_threshold=0.95,
            log_level="WARNING",
            **kwargs,
        )

        embedder = create_embedder(config)
        storage = create_storage_backend(config, embedder.embedding_dim)
        eviction = create_eviction_policy(eviction_policy)
        metrics = CacheMetrics()

        return CacheOperations(storage, embedder, eviction, config, metrics)

    return _create


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
