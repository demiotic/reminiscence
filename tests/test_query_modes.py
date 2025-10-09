"""Tests for query_mode functionality (exact, semantic, auto)."""

from reminiscence import Reminiscence, ReminiscenceConfig


class TestQueryModeExact:
    """Test exact query mode."""

    def test_exact_mode_no_embedding_generated(self):
        """Exact mode should not generate embeddings."""
        config = ReminiscenceConfig(db_uri="memory://", log_level="WARNING")
        cache = Reminiscence(config)

        cache.store(
            "SELECT * FROM users", {"db": "prod"}, "result_data", query_mode="exact"
        )

        # Should be stored in exact table
        assert cache.backend.exact_table.count_rows() == 1
        assert cache.backend.semantic_table.count_rows() == 0

    def test_exact_mode_matches_only_exact_queries(self):
        """Exact mode should only match identical queries."""
        config = ReminiscenceConfig(db_uri="memory://", log_level="WARNING")
        cache = Reminiscence(config)

        cache.store(
            "SELECT * FROM users", {"db": "prod"}, "result1", query_mode="exact"
        )

        # Exact match
        result = cache.lookup("SELECT * FROM users", {"db": "prod"}, query_mode="exact")
        assert result.is_hit
        assert result.similarity == 1.0

        # Different query (even semantically similar)
        result = cache.lookup(
            "SELECT * FROM users WHERE id > 0", {"db": "prod"}, query_mode="exact"
        )
        assert result.is_miss

    def test_exact_mode_respects_context(self):
        """Exact mode should respect context exactly."""
        config = ReminiscenceConfig(db_uri="memory://", log_level="WARNING")
        cache = Reminiscence(config)

        cache.store("query", {"db": "prod"}, "result_prod", query_mode="exact")
        cache.store("query", {"db": "dev"}, "result_dev", query_mode="exact")

        result_prod = cache.lookup("query", {"db": "prod"}, query_mode="exact")
        result_dev = cache.lookup("query", {"db": "dev"}, query_mode="exact")

        assert result_prod.is_hit
        assert result_dev.is_hit
        assert result_prod.result == "result_prod"
        assert result_dev.result == "result_dev"


class TestQueryModeSemantic:
    """Test semantic query mode."""

    def test_semantic_mode_generates_embeddings(self):
        """Semantic mode should generate and use embeddings."""
        config = ReminiscenceConfig(db_uri="memory://", log_level="WARNING")
        cache = Reminiscence(config)

        cache.store(
            "What is Python?",
            {"agent": "qa"},
            "Python is a language",
            query_mode="semantic",
        )

        # Should be stored in semantic table
        assert cache.backend.semantic_table.count_rows() == 1
        assert cache.backend.exact_table.count_rows() == 0

    def test_semantic_mode_matches_similar_queries(self):
        """Semantic mode should match semantically similar queries."""
        config = ReminiscenceConfig(
            db_uri="memory://", similarity_threshold=0.75, log_level="WARNING"
        )
        cache = Reminiscence(config)

        cache.store(
            "What is machine learning?",
            {"agent": "qa"},
            "ML is a subset of AI",
            query_mode="semantic",
        )

        result = cache.lookup(
            "Explain machine learning", {"agent": "qa"}, query_mode="semantic"
        )

        assert result.is_hit
        assert result.similarity > 0.75


class TestQueryModeAuto:
    """Test auto query mode."""

    def test_auto_mode_short_queries_use_exact(self):
        """Auto mode should use exact matching for short queries."""
        config = ReminiscenceConfig(db_uri="memory://", log_level="WARNING")
        cache = Reminiscence(config)

        short_query = "SELECT * FROM users"  # < 200 chars
        cache.store(short_query, {"db": "prod"}, "result", query_mode="auto")

        # Should be in exact table
        assert cache.backend.exact_table.count_rows() == 1
        assert cache.backend.semantic_table.count_rows() == 0

    def test_auto_mode_long_queries_use_semantic(self):
        """Auto mode should use semantic matching for long queries."""
        config = ReminiscenceConfig(db_uri="memory://", log_level="WARNING")
        cache = Reminiscence(config)

        long_query = "a" * 250  # > 200 chars
        cache.store(long_query, {"agent": "test"}, "result", query_mode="auto")

        # Should be in semantic table
        assert cache.backend.semantic_table.count_rows() == 1
        assert cache.backend.exact_table.count_rows() == 0

    def test_auto_mode_lookup_fallback(self):
        """Auto mode lookup should try exact first, then semantic."""
        config = ReminiscenceConfig(
            db_uri="memory://", similarity_threshold=0.75, log_level="WARNING"
        )
        cache = Reminiscence(config)

        # Store with semantic
        cache.store(
            "What is Python programming?",
            {"agent": "qa"},
            "Python is a language",
            query_mode="semantic",
        )

        # Auto lookup tries exact first (miss), then semantic (hit)
        result = cache.lookup(
            "Explain Python programming", {"agent": "qa"}, query_mode="auto"
        )

        assert result.is_hit


class TestQueryModeMixed:
    """Test mixed usage of query modes."""

    def test_different_modes_use_different_tables(self):
        """Different query modes should use separate tables."""
        config = ReminiscenceConfig(db_uri="memory://", log_level="WARNING")
        cache = Reminiscence(config)

        cache.store("query1", {"agent": "test"}, "result1", query_mode="exact")
        cache.store("query2", {"agent": "test"}, "result2", query_mode="semantic")

        assert cache.backend.exact_table.count_rows() == 1
        assert cache.backend.semantic_table.count_rows() == 1
        assert cache.backend.count() == 2

    def test_lookup_mode_must_match_storage_mode(self):
        """Lookup with wrong mode should miss."""
        config = ReminiscenceConfig(db_uri="memory://", log_level="WARNING")
        cache = Reminiscence(config)

        cache.store("query", {"agent": "test"}, "result", query_mode="exact")

        # Lookup with semantic mode won't find it
        result = cache.lookup("query", {"agent": "test"}, query_mode="semantic")
        assert result.is_miss

        # Lookup with exact mode will find it
        result = cache.lookup("query", {"agent": "test"}, query_mode="exact")
        assert result.is_hit


class TestQueryModeDecorator:
    """Test query_mode with decorator."""

    def test_decorator_with_exact_mode(self):
        """Decorator should support exact mode."""
        config = ReminiscenceConfig(db_uri="memory://", log_level="WARNING")
        cache = Reminiscence(config)

        @cache.cached(query="sql", query_mode="exact", context_params=["db"])
        def execute_sql(sql: str, db: str):
            return f"executed: {sql}"

        result1 = execute_sql("SELECT * FROM users", "prod")
        result2 = execute_sql("SELECT * FROM users", "prod")

        assert result1 == result2
        assert cache.backend.exact_table.count_rows() == 1

    def test_decorator_with_semantic_mode(self):
        """Decorator should support semantic mode."""
        config = ReminiscenceConfig(db_uri="memory://", log_level="WARNING")
        cache = Reminiscence(config)

        call_count = 0

        @cache.cached(query="question", query_mode="semantic")
        def ask_llm(question: str):
            nonlocal call_count
            call_count += 1
            return f"answer to {question}"

        result1 = ask_llm("What is Python?")
        result2 = ask_llm("What is Python?")

        assert call_count == 1
        assert result1 == result2
        assert cache.backend.semantic_table.count_rows() == 1
