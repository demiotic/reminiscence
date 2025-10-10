"""Shared fixtures for Reminiscence tests."""

import pytest
import tempfile
import shutil
import structlog
import logging

from reminiscence import Reminiscence, ReminiscenceConfig
from reminiscence.cache import CacheOperations
from reminiscence.embeddings.fastembed import FastEmbedEmbedder
from reminiscence.storage import LanceDBBackend
from reminiscence.eviction import create_eviction_policy
from reminiscence.metrics import CacheMetrics


# ============================================================================
# SINGLETON CLEANUP FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def clear_singletons():
    """Clear singleton instances between tests for isolation."""
    yield

    from reminiscence.storage.lancedb import LanceDBBackend
    from reminiscence.metrics.exporters import OpenTelemetryExporter

    LanceDBBackend._clear_instances()
    OpenTelemetryExporter._clear_instances()


# ============================================================================
# SHARED EMBEDDER (session-scoped) - HUGE SPEEDUP
# ============================================================================


@pytest.fixture(scope="session")
def shared_embedder():
    """Load embedder once per test session (~2s saved per test)."""
    config = ReminiscenceConfig(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return FastEmbedEmbedder(config)


@pytest.fixture(scope="session")
def embedding_dim(shared_embedder):
    """Get embedding dimension once."""
    return len(shared_embedder.embed("test"))


# ============================================================================
# REMINISCENCE FACTORY (uses shared embedder)
# ============================================================================


@pytest.fixture
def reminiscence_factory(shared_embedder):
    """Factory to create Reminiscence with shared embedder."""
    created_instances = []

    def _create(**config_kwargs):
        config_kwargs.setdefault("db_uri", "memory://")
        config_kwargs.setdefault("log_level", "WARNING")
        config_kwargs.setdefault("otel_enabled", False)

        config = ReminiscenceConfig(**config_kwargs)
        cache = Reminiscence(config, embedder=shared_embedder)

        created_instances.append(cache)
        return cache

    yield _create

    for cache in created_instances:
        if hasattr(cache, "backend"):
            cache.backend.clear()
    LanceDBBackend._clear_instances()


@pytest.fixture
def reminiscence(reminiscence_factory):
    """Default Reminiscence instance."""
    return reminiscence_factory()


# ============================================================================
# CACHE OPERATIONS FIXTURES (uses shared embedder)
# ============================================================================


@pytest.fixture
def ops_factory(shared_embedder, embedding_dim):
    """Factory to create CacheOperations with shared embedder."""
    created_backends = []

    def _create(
        eviction_policy="fifo", max_entries=None, ttl_seconds=None, **config_kwargs
    ):
        config_kwargs.setdefault("db_uri", "memory://")
        config_kwargs.setdefault("log_level", "WARNING")
        config_kwargs.setdefault("otel_enabled", False)

        config = ReminiscenceConfig(
            eviction_policy=eviction_policy,
            max_entries=max_entries,
            ttl_seconds=ttl_seconds,
            **config_kwargs,
        )

        storage = LanceDBBackend(config, embedding_dim)
        created_backends.append(storage)

        eviction = create_eviction_policy(eviction_policy)
        metrics = CacheMetrics()

        return CacheOperations(storage, shared_embedder, eviction, config, metrics)

    yield _create

    for backend in created_backends:
        backend.clear()
    LanceDBBackend._clear_instances()


@pytest.fixture
def cache_ops(ops_factory):
    """Default CacheOperations (no eviction limit)."""
    return ops_factory("fifo", max_entries=None)


@pytest.fixture
def fifo_ops(ops_factory):
    """FIFO CacheOperations with max_entries=3."""
    return ops_factory("fifo", max_entries=3, similarity_threshold=0.95)


@pytest.fixture
def lru_ops(ops_factory):
    """LRU CacheOperations with max_entries=3."""
    return ops_factory("lru", max_entries=3, similarity_threshold=0.95)


@pytest.fixture
def lfu_ops(ops_factory):
    """LFU CacheOperations with max_entries=3."""
    return ops_factory("lfu", max_entries=3, similarity_threshold=0.95)


# ============================================================================
# STRUCTLOG RESET FIXTURES
# ============================================================================


@pytest.fixture
def reset_config():
    """Reset configuration singleton."""
    from reminiscence.config import ReminiscenceConfig

    if hasattr(ReminiscenceConfig, "_instance"):
        delattr(ReminiscenceConfig, "_instance")
    yield


@pytest.fixture
def reset_structlog():
    """Reset structlog configuration."""
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
    """Configure JSON logging."""
    monkeypatch.setenv("REMINISCENCE_JSON_LOGS", "true")
    monkeypatch.setenv("REMINISCENCE_LOG_LEVEL", "INFO")

    from reminiscence.utils.logging import configure_logging

    configure_logging(log_level="INFO", json_logs=True)

    yield


@pytest.fixture
def text_logging_env(reset_structlog, reset_config, monkeypatch):
    """Configure text logging."""
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
# ENCRYPTION FIXTURES
# ============================================================================


@pytest.fixture(scope="session")
def age_keypair():
    """Generate real age keypair once per session."""
    try:
        from pyrage import x25519

        identity = x25519.Identity.generate()
        return (str(identity), str(identity.to_public()))
    except Exception:
        return (
            "AGE-SECRET-KEY-1GQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ",
            "age1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
        )


@pytest.fixture
def age_private_key(age_keypair):
    """Private key string."""
    return age_keypair[0]


@pytest.fixture
def age_public_key(age_keypair):
    """Public key string."""
    return age_keypair[1]


@pytest.fixture
def age_encryption(age_private_key):
    """AgeEncryption instance."""
    from reminiscence.encryption import AgeEncryption

    return AgeEncryption(key=age_private_key)


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
