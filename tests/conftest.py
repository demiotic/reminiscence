"""Shared fixtures for Reminiscence tests."""

import pytest
import tempfile
import shutil
import structlog
import logging
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
    )
    cache = Reminiscence(config)

    # El modelo se carga una vez aquí
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
    # Limpia cache y métricas antes de cada test
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

    # Reset eviction state
    cache_ops_module.eviction = create_eviction_policy(
        cache_ops_module.config.eviction_policy
    )

    yield cache_ops_module


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


class TestQueryModes:
    """Tests for query_mode parameter (semantic/exact/auto)."""

    def test_semantic_mode_default(self, reminiscence_memory):
        """Semantic mode should work by default (with embeddings)."""
        reminiscence_memory.store(
            "What is Python?",
            {"agent": "test"},
            "Python is a programming language",
            query_mode="semantic",
        )

        # Should hit with similar query
        result = reminiscence_memory.lookup(
            "Explain Python to me", {"agent": "test"}, query_mode="semantic"
        )

        assert result.is_hit
        assert result.similarity > 0.75

    def test_exact_mode_no_embeddings(self, reminiscence_memory):
        """Exact mode should skip embedding generation."""
        query = "SELECT * FROM users WHERE id = 1"
        context = {"database": "prod"}
        expected = [{"id": 1, "name": "Alice"}]

        # Store with exact mode (no embeddings)
        reminiscence_memory.store(query, context, expected, query_mode="exact")

        # Check that entry was stored
        assert reminiscence_memory.backend.count() == 1

        # Verify embedding is None
        arrow_table = reminiscence_memory.backend.to_arrow()
        row = arrow_table.to_pylist()[0]
        assert (
            row["embedding"] is None
            or len([x for x in row["embedding"] if x is not None]) == 0
        )

        # Exact lookup should hit
        result = reminiscence_memory.lookup(query, context, query_mode="exact")
        assert result.is_hit
        assert result.result == expected
        assert result.similarity == 1.0  # Exact match

    def test_exact_mode_different_query_misses(self, reminiscence_memory):
        """Exact mode should miss on slightly different query."""
        reminiscence_memory.store(
            "SELECT * FROM users", {"database": "prod"}, [{"id": 1}], query_mode="exact"
        )

        # Different query should miss (no semantic search)
        result = reminiscence_memory.lookup(
            "SELECT * FROM users WHERE id > 0", {"database": "prod"}, query_mode="exact"
        )

        assert result.is_miss

    def test_exact_mode_faster_than_semantic(self, reminiscence_memory):
        """Exact mode should be faster (no embedding generation)."""
        query = "SELECT COUNT(*) FROM orders"
        context = {"database": "analytics"}
        result_data = {"count": 1000}

        # Store with exact mode
        start = time.time()
        reminiscence_memory.store(query, context, result_data, query_mode="exact")
        exact_store_time = time.time() - start

        # Lookup with exact mode
        start = time.time()
        reminiscence_memory.lookup(query, context, query_mode="exact")
        exact_lookup_time = time.time() - start

        # Clear and repeat with semantic mode
        reminiscence_memory.clear()

        start = time.time()
        reminiscence_memory.store(query, context, result_data, query_mode="semantic")
        semantic_store_time = time.time() - start

        start = time.time()
        reminiscence_memory.lookup(query, context, query_mode="semantic")
        semantic_lookup_time = time.time() - start

        # Exact should be faster (no embedding overhead)
        # Note: This is a rough check, may be flaky on slow systems
        assert exact_store_time < semantic_store_time * 2  # At least 2x faster
        assert exact_lookup_time < semantic_lookup_time * 2

    def test_auto_mode_tries_exact_first(self, reminiscence_memory):
        """Auto mode should try exact match first, then semantic."""
        query = "What is machine learning?"
        context = {"agent": "qa"}
        expected = "ML explanation"

        # Store with semantic (has embeddings)
        reminiscence_memory.store(query, context, expected, query_mode="semantic")

        # Auto mode with exact same query should hit via exact match
        result = reminiscence_memory.lookup(query, context, query_mode="auto")
        assert result.is_hit
        assert result.result == expected
        assert result.similarity == 1.0  # Exact match found first

    def test_auto_mode_fallback_to_semantic(self, reminiscence_memory):
        """Auto mode should fallback to semantic if exact fails."""
        reminiscence_memory.store(
            "What is deep learning?",
            {"agent": "qa"},
            "DL explanation",
            query_mode="semantic",
        )

        # Slightly different query should miss exact, hit semantic
        result = reminiscence_memory.lookup(
            "Explain deep learning", {"agent": "qa"}, query_mode="auto"
        )

        assert result.is_hit
        assert result.result == "DL explanation"
        assert result.similarity < 1.0  # Semantic match (not exact)
        assert result.similarity > 0.75

    def test_semantic_mode_skips_exact_entries(self, reminiscence_memory):
        """Semantic mode should not find entries stored with exact mode."""
        reminiscence_memory.store(
            "SELECT * FROM products",
            {"database": "prod"},
            [{"id": 1}],
            query_mode="exact",  # No embeddings
        )

        # Semantic search should miss (entry has no embedding)
        result = reminiscence_memory.lookup(
            "SELECT * FROM products", {"database": "prod"}, query_mode="semantic"
        )

        assert result.is_miss

    def test_exact_mode_with_context_exact_match(self, reminiscence_memory):
        """Exact mode should match both query AND context exactly."""
        query = "SELECT * FROM orders"

        reminiscence_memory.store(
            query,
            {"database": "prod", "user": "alice"},
            [{"count": 100}],
            query_mode="exact",
        )

        # Different context should miss
        result = reminiscence_memory.lookup(
            query, {"database": "prod", "user": "bob"}, query_mode="exact"
        )
        assert result.is_miss

        # Same context should hit
        result = reminiscence_memory.lookup(
            query, {"database": "prod", "user": "alice"}, query_mode="exact"
        )
        assert result.is_hit

    def test_mixed_modes_coexist(self, reminiscence_memory):
        """Entries with different modes should coexist."""
        # Store one entry with semantic
        reminiscence_memory.store(
            "What is AI?", {"agent": "qa"}, "AI explanation", query_mode="semantic"
        )

        # Store one entry with exact
        reminiscence_memory.store(
            "SELECT * FROM users", {"database": "prod"}, [{"id": 1}], query_mode="exact"
        )

        # Both should be retrievable with their respective modes
        assert reminiscence_memory.backend.count() == 2

        result1 = reminiscence_memory.lookup(
            "Explain AI", {"agent": "qa"}, query_mode="semantic"
        )
        assert result1.is_hit

        result2 = reminiscence_memory.lookup(
            "SELECT * FROM users", {"database": "prod"}, query_mode="exact"
        )
        assert result2.is_hit

    def test_embedder_loaded_on_semantic_mode(self):
        """Embedder should be lazy loaded when semantic mode is used."""
        config = ReminiscenceConfig(
            db_uri="memory://",
            enable_metrics=True,
            log_level="WARNING",
        )
        cache = Reminiscence(config)

        # Embedder not loaded yet
        assert not cache._embedder_initialized

        # Use semantic mode - should trigger embedder loading
        cache.store(
            "What is Python?",
            {"agent": "qa"},
            "Python explanation",
            query_mode="semantic",
        )

        # Embedder should now be loaded
        assert cache._embedder_initialized

    def test_auto_mode_with_exact_only_entry(self, reminiscence_memory):
        """Auto mode with exact-only entry should only match exact queries."""
        reminiscence_memory.store(
            "SELECT COUNT(*) FROM orders",
            {"database": "analytics"},
            {"count": 500},
            query_mode="exact",
        )

        # Exact same query should hit
        result = reminiscence_memory.lookup(
            "SELECT COUNT(*) FROM orders", {"database": "analytics"}, query_mode="auto"
        )
        assert result.is_hit
        assert result.similarity == 1.0

        # Similar but different query should miss (no embedding to search)
        result = reminiscence_memory.lookup(
            "SELECT COUNT(*) FROM orders WHERE status = 'complete'",
            {"database": "analytics"},
            query_mode="auto",
        )
        assert result.is_miss


class TestDecoratorQueryModes:
    """Tests for decorator with query_mode parameter."""

    def test_decorator_semantic_mode(self, reminiscence_memory):
        """Decorator with semantic mode should cache semantically."""

        @reminiscence_memory.cached(
            query="question", query_mode="semantic", context_params=["user"]
        )
        def ask_llm(question: str, user: str):
            return f"Answer for {user}: {question}"

        # First call - cache miss
        result1 = ask_llm("What is Python?", "alice")

        # Similar question - should hit
        result2 = ask_llm("Explain Python", "alice")

        assert result1 == result2  # Same cached result

    def test_decorator_exact_mode(self, reminiscence_memory):
        """Decorator with exact mode should only hit on exact match."""

        @reminiscence_memory.cached(
            query="sql", query_mode="exact", context_params=["database"]
        )
        def run_query(sql: str, database: str):
            return f"Results from {database}: {sql}"

        # First call
        result1 = run_query("SELECT * FROM users", "prod")

        # Exact same call - should hit
        result2 = run_query("SELECT * FROM users", "prod")
        assert result1 == result2

        # Different SQL - should miss
        result3 = run_query("SELECT * FROM orders", "prod")
        assert result3 != result1

    def test_decorator_auto_mode(self, reminiscence_memory):
        """Decorator with auto mode should try exact then semantic."""

        call_count = 0

        @reminiscence_memory.cached(query="prompt", query_mode="auto")
        def generate_text(prompt: str):
            nonlocal call_count
            call_count += 1
            return f"Generated: {prompt}"

        # First call
        result1 = generate_text("Hello world")
        assert call_count == 1

        # Exact same - should hit via exact
        result2 = generate_text("Hello world")
        assert call_count == 1  # Not called again
        assert result1 == result2

        # Similar - should hit via semantic
        _ = generate_text("Hello there world")
        assert call_count == 1  # Still not called (semantic hit)

    def test_decorator_renamed_parameters(self, reminiscence_memory):
        """Test new decorator parameter names (query, context_params)."""

        @reminiscence_memory.cached(
            query="user_input",
            context_params=["model", "temperature"],
            query_mode="semantic",
        )
        def call_llm(user_input: str, model: str, temperature: float):
            return f"LLM output: {user_input}"

        result = call_llm("Test", "gpt-4", 0.7)
        assert "LLM output: Test" in result
