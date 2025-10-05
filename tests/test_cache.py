"""Tests para SemanticCache."""

import pytest
import time
from memora import SemanticCache, CacheConfig


def test_cache_initialization(cache):
    """Test inicialización básica."""
    assert cache is not None
    assert cache.embedding_dim > 0
    assert cache.table.count_rows() == 0


def test_cache_miss_empty(cache, sample_context):
    """Test MISS en caché vacío."""
    result = cache.get("Test query", sample_context)
    assert result is None
    assert cache.metrics.misses == 1
    assert cache.metrics.hits == 0


def test_cache_set_and_get(cache, sample_context):
    """Test guardar y recuperar de caché."""
    query = "Explícame Python"
    response = "Python es un lenguaje..."

    cache.set(query, sample_context, response)

    # Query exacta debería dar HIT
    result = cache.get(query, sample_context)
    assert result == response
    assert cache.metrics.hits == 1


def test_cache_similar_queries(cache, sample_context):
    """Test que queries similares den HIT."""
    cache.set("Explícame Python", sample_context, "Python es...")

    # Query similar debería dar HIT (con threshold 0.75)
    result = cache.get("Qué es Python", sample_context)
    assert result == "Python es..."
    assert cache.metrics.hits == 1


def test_cache_different_context(cache):
    """Test que contextos diferentes den MISS."""
    ctx1 = {"tools": ["search"]}
    ctx2 = {"tools": ["code_execute"]}

    cache.set("Explícame Python", ctx1, "Response 1")

    # Mismo query, diferente contexto → MISS
    result = cache.get("Explícame Python", ctx2)
    assert result is None
    assert cache.metrics.misses == 1


def test_get_or_compute(cache, sample_context, fake_llm):
    """Test get_or_compute con MISS y HIT."""
    query = "Test query"

    # Primera vez: MISS, llama a compute_fn
    result1 = cache.get_or_compute(query, sample_context, fake_llm)
    assert result1 == f"Response for: {query}"
    assert cache.metrics.misses == 1

    # Segunda vez: HIT, no llama a compute_fn
    result2 = cache.get_or_compute(query, sample_context, fake_llm)
    assert result2 == result1
    assert cache.metrics.hits == 1


def test_ttl_expiration():
    """Test que entries expiren según TTL."""
    config = CacheConfig(
        db_uri="memory://",
        ttl_seconds=1,  # 1 segundo TTL
        enable_metrics=True,
        log_level="WARNING",  # Silenciar logs en tests
    )
    cache = SemanticCache(config)
    ctx = {"tools": ["search"]}

    cache.set("Query", ctx, "Response")

    # Inmediato: HIT
    assert cache.get("Query", ctx) == "Response"

    # Después de TTL: MISS
    time.sleep(1.1)
    assert cache.get("Query", ctx) is None


def test_cleanup_old_entries():
    """Test limpieza de entries antiguas."""
    config = CacheConfig(
        db_uri="memory://",
        ttl_seconds=3600,
        cleanup_threshold=0.0,  # Forzar cleanup inmediato (sin threshold)
        log_level="WARNING",
    )
    cache = SemanticCache(config)
    ctx = {"tools": ["search"]}

    cache.set("Query 1", ctx, "Response 1")
    cache.set("Query 2", ctx, "Response 2")

    assert cache.table.count_rows() == 2

    # Limpiar entries con max_age_seconds=0 (todas expiran)
    deleted = cache.cleanup_old_entries(max_age_seconds=0)

    assert deleted == 2  # Se eliminaron 2 entries
    assert cache.table.count_rows() == 0


def test_cleanup_with_threshold():
    """Test que cleanup respeta el threshold."""
    config = CacheConfig(
        db_uri="memory://",
        ttl_seconds=100,
        cleanup_threshold=0.5,  # Solo limpiar si >50% expiradas
        log_level="WARNING",
    )
    cache = SemanticCache(config)
    ctx = {"tools": ["search"]}

    # Añadir 10 entries
    for i in range(10):
        cache.set(f"Query {i}", ctx, f"Response {i}")

    assert cache.table.count_rows() == 10

    # Intentar cleanup con max_age que expira solo 3 entries (30%)
    # No debería limpiar porque 30% < threshold 50%
    deleted = cache.cleanup_old_entries(max_age_seconds=70)

    assert deleted == 0  # No limpió nada
    assert cache.table.count_rows() == 10  # Todas siguen ahí


def test_metrics_tracking(sample_context, fake_llm):
    """Test que métricas se trackeen correctamente."""
    # Crear cache fresco para este test (no usar fixture)
    config = CacheConfig(db_uri="memory://", log_level="WARNING")
    cache = SemanticCache(config)

    cache.get_or_compute("Explain quantum physics", sample_context, fake_llm)
    cache.get_or_compute("Explain quantum physics", sample_context, fake_llm)
    cache.get_or_compute("Recipe for cheese cake", sample_context, fake_llm)

    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 2
    assert stats["hit_rate"] == "33.33%"


def test_cross_lingual_caching(cache):
    """Test que cache funcione cross-lingual."""
    ctx = {"tools": ["search"]}

    cache.set("Explícame Python", ctx, "Python es...")

    # Query en inglés debería dar HIT (modelo multilingüe con threshold 0.75)
    result = cache.get("What is Python", ctx)
    assert result == "Python es..."


def test_config_factory_methods():
    """Test factory methods de CacheConfig."""
    prod_config = CacheConfig.for_production("./test.db")
    assert prod_config.db_uri == "./test.db"
    assert prod_config.ttl_seconds == 86400
    assert prod_config.log_level == "INFO"

    dev_config = CacheConfig.for_development()
    assert dev_config.db_uri == "memory://"
    assert dev_config.log_level == "DEBUG"
