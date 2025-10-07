"""Tests for fingerprint.py - context hashing."""

from reminiscence.utils.fingerprint import create_fingerprint


class TestFingerprintBasics:
    """Basic fingerprint functionality tests."""

    def test_deterministic(self):
        """Fingerprint should be deterministic."""
        ctx = {"agent": "qa", "model": "gpt-4"}
        fp1 = create_fingerprint(ctx)
        fp2 = create_fingerprint(ctx)
        assert fp1 == fp2

    def test_sha256_length(self):
        """Fingerprint should be 64 hex characters (SHA256)."""
        ctx = {"agent": "test"}
        fp = create_fingerprint(ctx)
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)

    def test_empty_context(self):
        """Empty contexts should have consistent fingerprint."""
        fp1 = create_fingerprint({})
        fp2 = create_fingerprint({})
        assert fp1 == fp2
        assert len(fp1) == 64

    def test_none_values(self):
        """None values should be handled."""
        ctx1 = {"param": None}
        ctx2 = {"param": None}
        assert create_fingerprint(ctx1) == create_fingerprint(ctx2)

    def test_bool_values(self):
        """Bool values should be distinguished."""
        ctx1 = {"enabled": True}
        ctx2 = {"enabled": False}
        assert create_fingerprint(ctx1) != create_fingerprint(ctx2)


class TestFingerprintKeyOrdering:
    """Tests for key order independence."""

    def test_dict_key_order_irrelevant(self):
        """Dict key order should not matter."""
        ctx1 = {"a": 1, "b": 2, "c": 3}
        ctx2 = {"c": 3, "a": 1, "b": 2}
        assert create_fingerprint(ctx1) == create_fingerprint(ctx2)

    def test_nested_dict_key_order(self):
        """Nested dict key order should not matter."""
        ctx1 = {"agent": "sql", "config": {"db": "prod", "timeout": 30}}
        ctx2 = {"config": {"timeout": 30, "db": "prod"}, "agent": "sql"}
        assert create_fingerprint(ctx1) == create_fingerprint(ctx2)


class TestFingerprintStructures:
    """Tests for complex data structures."""

    def test_nested_dict(self):
        """Nested dicts should work."""
        ctx = {"agent": "sql", "config": {"db": "prod", "timeout": 30}}
        fp = create_fingerprint(ctx)
        assert len(fp) == 64

    def test_list_order_matters(self):
        """List order should matter (exact matching)."""
        ctx1 = {"tools": ["search", "calculator"]}
        ctx2 = {"tools": ["calculator", "search"]}
        # Different order = different fingerprint
        assert create_fingerprint(ctx1) != create_fingerprint(ctx2)

    def test_list_same_order(self):
        """Lists with same order should match."""
        ctx1 = {"tools": ["search", "calculator"]}
        ctx2 = {"tools": ["search", "calculator"]}
        assert create_fingerprint(ctx1) == create_fingerprint(ctx2)

    def test_deeply_nested(self):
        """Deeply nested structures."""
        ctx = {"level1": {"level2": {"level3": {"level4": {"value": 42}}}}}
        fp = create_fingerprint(ctx)
        assert len(fp) == 64


class TestFingerprintRealWorld:
    """Tests with real-world use cases."""

    def test_agent_context(self):
        """Typical agent context."""
        ctx = {
            "agent_id": "sql_analyzer",
            "version": "v1.2.3",
            "config": {"database": "production", "timeout": 30, "max_rows": 1000},
            "tools": ["query", "format"],
        }
        fp = create_fingerprint(ctx)
        assert len(fp) == 64
        # Same context = same fingerprint
        assert create_fingerprint(ctx) == fp

    def test_config_change_changes_fingerprint(self):
        """Minimal config change should change fingerprint."""
        ctx1 = {"agent": "analyzer", "model": "gpt-4", "temperature": 0.0}
        ctx2 = {"agent": "analyzer", "model": "gpt-4", "temperature": 0.1}
        assert create_fingerprint(ctx1) != create_fingerprint(ctx2)

    def test_multiagent_context(self):
        """Multi-agent system context."""
        ctx = {
            "pipeline": "analysis",
            "step": 3,
            "upstream": ["step1", "step2"],
            "config": {"parallel": True, "retry_count": 3},
        }
        fp = create_fingerprint(ctx)
        assert isinstance(fp, str)
        assert len(fp) == 64

    def test_tools_with_params(self):
        """Tools with parameters (LLM use case)."""
        ctx1 = {
            "agent": "qa",
            "tools": [{"name": "search", "params": {"limit": 10}}, {"name": "calc"}],
        }
        ctx2 = {
            "agent": "qa",
            "tools": [{"name": "search", "params": {"limit": 10}}, {"name": "calc"}],
        }
        assert create_fingerprint(ctx1) == create_fingerprint(ctx2)

    def test_tools_different_order(self):
        """Tools in different order should differ."""
        ctx1 = {"tools": [{"name": "search"}, {"name": "calc"}]}
        ctx2 = {"tools": [{"name": "calc"}, {"name": "search"}]}
        assert create_fingerprint(ctx1) != create_fingerprint(ctx2)


class TestFingerprintEdgeCases:
    """Tests for edge cases."""

    def test_very_long_values(self):
        """Very long values should hash correctly."""
        ctx = {
            "description": "a" * 10000,
            "value": 42,
        }
        fp = create_fingerprint(ctx)
        assert len(fp) == 64  # Always 64 chars

    def test_special_characters(self):
        """Special characters in strings."""
        ctx = {
            "query": "SELECT * FROM users WHERE name = 'O\"Reilly'",
            "param": "hello\nworld\t!",
        }
        fp1 = create_fingerprint(ctx)
        fp2 = create_fingerprint(ctx)
        assert fp1 == fp2

    def test_unicode_values(self):
        """Unicode values should work."""
        ctx1 = {"text": "Hello 世界"}
        ctx2 = {"text": "Hello 世界"}
        assert create_fingerprint(ctx1) == create_fingerprint(ctx2)

    def test_numeric_types(self):
        """Different numeric representations."""
        # JSON serialization treats these as different
        ctx1 = {"value": 42}
        ctx2 = {"value": 42.0}
        # These will be DIFFERENT because JSON preserves type
        # 42 -> "42", 42.0 -> "42.0"
        assert create_fingerprint(ctx1) != create_fingerprint(ctx2)


class TestFingerprintDeterminism:
    """Determinism tests."""

    def test_multiple_calls_same_result(self):
        """Multiple calls should give same result."""
        ctx = {"agent": "test", "params": {"a": 1, "b": 2}, "tools": ["t1", "t2"]}
        fingerprints = [create_fingerprint(ctx) for _ in range(100)]
        # All should be identical
        assert len(set(fingerprints)) == 1

    def test_different_contexts_different_fingerprints(self):
        """Different contexts should have different fingerprints."""
        contexts = [
            {"agent": "a"},
            {"agent": "b"},
            {"agent": "a", "version": "v1"},
            {"agent": "a", "version": "v2"},
            {},
        ]
        fingerprints = [create_fingerprint(ctx) for ctx in contexts]
        # All should be unique
        assert len(set(fingerprints)) == len(contexts)

    def test_collision_resistance(self):
        """Similar contexts should have different hashes."""
        contexts = [
            {"model": "gpt-4"},
            {"model": "gpt-3"},
            {"model": "claude"},
            {"model": "gpt-4", "temperature": 0.7},
            {"model": "gpt-4", "temperature": 0.8},
        ]
        fingerprints = [create_fingerprint(ctx) for ctx in contexts]
        # All unique
        assert len(set(fingerprints)) == len(contexts)
