"""Tests para memora.core.Memora."""

import pytest
import time
from memora import Memora, CacheConfig


class TestMemoraBasics:
    """Tests básicos de inicialización y operaciones."""

    def test_init_memory(self, memora_memory):
        """Test inicialización con memory backend."""
        assert memora_memory is not None
        assert memora_memory.table.count_rows() == 0

    def test_init_disk(self, memora_disk):
        """Test inicialización con disk backend."""
        assert memora_disk is not None
        assert memora_disk.config.db_uri != "memory://"

    def test_get_stats_empty(self, memora_memory):
        """Test estadísticas con caché vacío."""
        stats = memora_memory.get_stats()
        assert stats["total_entries"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0


class TestLookupAndStore:
    """Tests de búsqueda y almacenamiento."""

    def test_lookup_miss_empty_cache(self, memora_memory):
        """Lookup en caché vacío debe retornar miss."""
        result = memora_memory.lookup("test query", {"agent": "test"})

        assert result.is_miss
        assert result.result is None
        assert result.similarity is None

    def test_store_and_lookup_exact(self, memora_memory):
        """Store seguido de lookup exacto debe hacer HIT."""
        query = "¿Qué es Python?"
        context = {"agent": "llm"}
        expected = "Python es un lenguaje de programación"

        # Store
        memora_memory.store(query, context, expected)
        assert memora_memory.table.count_rows() == 1

        # Lookup exacto
        result = memora_memory.lookup(query, context)

        assert result.is_hit
        assert result.result == expected
        assert result.similarity >= 0.99  # Casi 1.0 por ser idéntico
        assert result.matched_query == query

    def test_lookup_similar_query(self, memora_memory, sample_queries):
        """Lookup con query similar debe hacer HIT si supera threshold."""
        original_query = sample_queries["similar"][0]
        similar_query = sample_queries["similar"][1]
        context = {"agent": "llm"}

        # Store original
        memora_memory.store(original_query, context, "respuesta")

        # Lookup similar
        result = memora_memory.lookup(similar_query, context)

        # Debería hacer HIT (queries son semánticamente similares)
        assert result.is_hit
        assert result.similarity > 0.75

    def test_lookup_different_query_miss(self, memora_memory, sample_queries):
        """Lookup con query muy diferente debe hacer MISS."""
        query1 = sample_queries["different"][0]
        query2 = sample_queries["different"][1]
        context = {"agent": "llm"}

        # Store primera query
        memora_memory.store(query1, context, "respuesta 1")

        # Lookup query diferente
        result = memora_memory.lookup(query2, context)

        # Debería hacer MISS (queries son diferentes)
        assert result.is_miss or result.similarity < 0.75

    def test_lookup_different_context_miss(self, memora_memory, sample_contexts):
        """Lookup con contexto diferente debe hacer MISS."""
        query = "¿Qué es Python?"

        # Store con contexto 1
        memora_memory.store(query, sample_contexts["llm_gpt4"], "respuesta gpt4")

        # Lookup con contexto 2
        result = memora_memory.lookup(query, sample_contexts["llm_claude"])

        # MISS porque el contexto es diferente
        assert result.is_miss

    def test_store_multiple_contexts(self, memora_memory, sample_contexts):
        """Múltiples stores con diferentes contextos deben coexistir."""
        query = "¿Qué es Python?"

        memora_memory.store(query, sample_contexts["llm_gpt4"], "respuesta gpt4")
        memora_memory.store(query, sample_contexts["llm_claude"], "respuesta claude")

        assert memora_memory.table.count_rows() == 2

        # Cada contexto debe retornar su respuesta
        result_gpt4 = memora_memory.lookup(query, sample_contexts["llm_gpt4"])
        result_claude = memora_memory.lookup(query, sample_contexts["llm_claude"])

        assert result_gpt4.result == "respuesta gpt4"
        assert result_claude.result == "respuesta claude"


class TestTTL:
    """Tests de Time-To-Live."""

    def test_ttl_not_expired(self):
        """Entry dentro del TTL debe hacer HIT."""
        config = CacheConfig(
            db_uri="memory://", ttl_seconds=10, enable_metrics=True, log_level="WARNING"
        )
        memora = Memora(config)

        query = "test"
        context = {"agent": "test"}

        memora.store(query, context, "resultado")

        # Lookup inmediato (no expirado)
        result = memora.lookup(query, context)
        assert result.is_hit

    def test_ttl_expired(self):
        """Entry fuera del TTL debe hacer MISS."""
        config = CacheConfig(
            db_uri="memory://",
            ttl_seconds=1,  # 1 segundo
            enable_metrics=True,
            log_level="WARNING",
        )
        memora = Memora(config)

        query = "test"
        context = {"agent": "test"}

        memora.store(query, context, "resultado")

        # Esperar que expire
        time.sleep(1.5)

        # Lookup después de expiración
        result = memora.lookup(query, context)
        assert result.is_miss


class TestCheckAvailability:
    """Tests de check_availability."""

    def test_availability_hit(self, memora_memory):
        """Check availability debe retornar available=True si existe."""
        query = "test"
        context = {"agent": "test"}

        memora_memory.store(query, context, "resultado")

        avail = memora_memory.check_availability(query, context)

        assert avail.available
        assert avail.age_seconds is not None
        assert avail.similarity >= 0.99

    def test_availability_miss(self, memora_memory):
        """Check availability debe retornar available=False si no existe."""
        avail = memora_memory.check_availability("no existe", {"agent": "test"})

        assert not avail.available
        assert avail.age_seconds is None


class TestInvalidation:
    """Tests de invalidación."""

    def test_invalidate_by_context(self, memora_memory):
        """Invalidar por contexto debe eliminar entries específicas."""
        context1 = {"agent": "agent1"}
        context2 = {"agent": "agent2"}

        memora_memory.store("query1", context1, "result1")
        memora_memory.store("query2", context2, "result2")

        assert memora_memory.table.count_rows() == 2

        # Invalidar contexto 1
        deleted = memora_memory.invalidate(context=context1)

        assert deleted == 1
        assert memora_memory.table.count_rows() == 1

        # Verificar que solo quedó context2
        result = memora_memory.lookup("query2", context2)
        assert result.is_hit

    def test_invalidate_by_age(self, memora_memory):
        """Invalidar por edad debe eliminar solo entries antiguas."""
        context = {"agent": "test"}

        # Store primera entry
        memora_memory.store("query1", context, "result1")
        print(f"Query1 timestamp: {int(time.time())}")

        # Esperar un poco
        time.sleep(0.2)

        # Store segunda entry
        memora_memory.store("query2", context, "result2")
        print(f"Query2 timestamp: {int(time.time())}")

        assert memora_memory.table.count_rows() == 2

        # Ver todos los timestamps
        table = memora_memory.table.to_arrow()
        print(f"Timestamps en DB: {table['timestamp'].to_pylist()}")

        # Calcular cutoff
        now = time.time()
        older_than = 0.1
        cutoff = int(now - older_than)
        print(f"Now: {now}, Cutoff: {cutoff}, Older_than: {older_than}")

        # Invalidar entries más antiguas que 0.1 segundos
        deleted = memora_memory.invalidate(older_than_seconds=0.1)

        print(f"Deleted: {deleted}")

        assert deleted == 1  # Solo la primera
        assert memora_memory.table.count_rows() == 1


class TestMetrics:
    """Tests de métricas."""

    def test_metrics_hits_and_misses(self, memora_memory):
        """Métricas deben trackear hits y misses correctamente."""
        query = "test"
        context = {"agent": "test"}

        # Miss
        memora_memory.lookup(query, context)

        # Store
        memora_memory.store(query, context, "resultado")

        # Hit
        memora_memory.lookup(query, context)

        stats = memora_memory.get_stats()

        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert "50.00%" in stats["hit_rate"]

    def test_metrics_disabled(self):
        """Con metrics deshabilitadas, stats no debe incluir hits/misses."""
        config = CacheConfig(
            db_uri="memory://", enable_metrics=False, log_level="WARNING"
        )
        memora = Memora(config)

        stats = memora.get_stats()

        assert "hits" not in stats
        assert "misses" not in stats


class TestEdgeCases:
    """Tests de casos extremos."""

    def test_empty_query(self, memora_memory):
        """Query vacío debe funcionar."""
        memora_memory.store("", {"agent": "test"}, "resultado")
        result = memora_memory.lookup("", {"agent": "test"})

        assert result.is_hit

    def test_unicode_query(self, memora_memory):
        """Query con unicode debe funcionar."""
        query = "¿Qué es Python? 🐍 中文 العربية"
        context = {"agent": "test"}

        memora_memory.store(query, context, "resultado")
        result = memora_memory.lookup(query, context)

        assert result.is_hit

    def test_large_result(self, memora_memory):
        """Resultados grandes deben almacenarse correctamente."""
        large_result = "x" * 10000  # 10KB

        memora_memory.store("query", {"agent": "test"}, large_result)
        result = memora_memory.lookup("query", {"agent": "test"})

        assert result.is_hit
        assert result.result == large_result

    def test_complex_context(self, memora_memory):
        """Contexto complejo con nested dicts debe funcionar."""
        context = {
            "agent": "complex",
            "config": {
                "model": "gpt-4",
                "params": {"temperature": 0.7, "max_tokens": 100},
            },
            "tools": ["search", "calculator"],
        }

        memora_memory.store("query", context, "resultado")
        result = memora_memory.lookup("query", context)

        assert result.is_hit
