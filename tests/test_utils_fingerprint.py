"""Tests for fingerprint.py - context normalization."""

from memora.utils.fingerprint import create_fingerprint, fingerprint_matches


class TestFingerprintNormalization:
    """Type normalization tests."""

    def test_float_vs_int_normalization(self):
        """Integer floats should equal ints."""
        ctx1 = {"timeout": 30}
        ctx2 = {"timeout": 30.0}

        assert create_fingerprint(ctx1) == create_fingerprint(ctx2)
        assert fingerprint_matches(ctx1, ctx2)

    def test_float_precision(self):
        """Floats with minimal numeric drift should be equal."""
        # Realistic case: difference due to floating point arithmetic
        ctx1 = {"threshold": 0.1 + 0.2}
        ctx2 = {"threshold": 0.3}

        assert create_fingerprint(ctx1) == create_fingerprint(ctx2)

        # Rounding to 10 decimals makes them equal
        assert create_fingerprint(ctx1) == create_fingerprint(ctx2)

    def test_empty_context(self):
        """Empty contexts should have consistent fingerprint."""
        ctx1 = {}
        ctx2 = {}

        fp1 = create_fingerprint(ctx1)
        fp2 = create_fingerprint(ctx2)

        assert fp1 == fp2
        assert len(fp1) == 64  # SHA256 = 64 hex chars

    def test_none_values(self):
        """None should be handled correctly."""
        ctx1 = {"param": None}
        ctx2 = {"param": None}

        assert create_fingerprint(ctx1) == create_fingerprint(ctx2)

    def test_bool_values(self):
        """Bools should be distinguished."""
        ctx1 = {"enabled": True}
        ctx2 = {"enabled": False}

        assert create_fingerprint(ctx1) != create_fingerprint(ctx2)


class TestFingerprintStructures:
    """Tests for complex structures."""

    def test_nested_dict(self):
        """Nested dicts should work."""
        ctx1 = {"agent": "sql", "config": {"db": "prod", "timeout": 30}}
        ctx2 = {
            "agent": "sql",
            "config": {
                "db": "prod",
                "timeout": 30.0,  # Float vs int
            },
        }

        assert create_fingerprint(ctx1) == create_fingerprint(ctx2)

    def test_list_order_matters(self):
        """List order should matter."""
        ctx1 = {"tools": ["search", "calculator"]}
        ctx2 = {"tools": ["calculator", "search"]}

        # Different order = different fingerprint
        assert create_fingerprint(ctx1) != create_fingerprint(ctx2)

    def test_list_values(self):
        """Lists with same values in same order."""
        ctx1 = {"tools": ["search", "calculator"]}
        ctx2 = {"tools": ["search", "calculator"]}

        assert create_fingerprint(ctx1) == create_fingerprint(ctx2)

    def test_dict_key_order_irrelevant(self):
        """Dict key order should not matter."""
        ctx1 = {"a": 1, "b": 2, "c": 3}
        ctx2 = {"c": 3, "a": 1, "b": 2}

        assert create_fingerprint(ctx1) == create_fingerprint(ctx2)


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

        # Same context should give same fingerprint
        assert create_fingerprint(ctx) == fp

    def test_slight_config_change(self):
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


class TestFingerprintEdgeCases:
    """Tests for edge cases."""

    def test_very_long_values(self):
        """Very long values."""
        ctx = {
            "description": "a" * 10000,  # 10k char string
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
        """Unicode values."""
        ctx1 = {"text": "Hello 世界"}
        ctx2 = {"text": "Hello 世界"}

        assert create_fingerprint(ctx1) == create_fingerprint(ctx2)

    def test_deeply_nested(self):
        """Deeply nested structures."""
        ctx = {"level1": {"level2": {"level3": {"level4": {"value": 42}}}}}

        fp = create_fingerprint(ctx)
        assert len(fp) == 64

    def test_mixed_number_types(self):
        """Mix of ints and floats."""
        ctx1 = {"int_val": 10, "float_val": 3.14, "int_as_float": 20.0}
        ctx2 = {
            "int_val": 10,
            "float_val": 3.14,
            "int_as_float": 20,  # Int, not float
        }

        assert create_fingerprint(ctx1) == create_fingerprint(ctx2)


class TestFingerprintDeterminism:
    """Determinism tests."""

    def test_multiple_calls_same_result(self):
        """Multiple calls should give same result."""
        ctx = {"agent": "test", "params": {"a": 1, "b": 2}, "tools": ["t1", "t2"]}

        fingerprints = [create_fingerprint(ctx) for _ in range(100)]

        # All should be equal
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
