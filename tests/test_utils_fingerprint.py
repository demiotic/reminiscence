"""Tests para fingerprint.py - normalización de contextos."""

import pytest
from memora.utils.fingerprint import create_fingerprint, fingerprint_matches


class TestFingerprintNormalization:
    """Tests de normalización de tipos."""

    def test_float_vs_int_normalization(self):
        """Floats enteros deben ser iguales a ints."""
        ctx1 = {"timeout": 30}
        ctx2 = {"timeout": 30.0}

        assert create_fingerprint(ctx1) == create_fingerprint(ctx2)
        assert fingerprint_matches(ctx1, ctx2)

    def test_float_precision(self):
        """Floats con drift numérico mínimo deben ser iguales."""
        # Caso realista: diferencia por aritmética de punto flotante
        ctx1 = {"threshold": 0.1 + 0.2}
        ctx2 = {"threshold": 0.3}

        assert create_fingerprint(ctx1) == create_fingerprint(ctx2)

        # Redondeo a 10 decimales hace que sean iguales
        assert create_fingerprint(ctx1) == create_fingerprint(ctx2)

    def test_empty_context(self):
        """Contextos vacíos deben tener fingerprint consistente."""
        ctx1 = {}
        ctx2 = {}

        fp1 = create_fingerprint(ctx1)
        fp2 = create_fingerprint(ctx2)

        assert fp1 == fp2
        assert len(fp1) == 64  # SHA256 = 64 chars hex

    def test_none_values(self):
        """None debe manejarse correctamente."""
        ctx1 = {"param": None}
        ctx2 = {"param": None}

        assert create_fingerprint(ctx1) == create_fingerprint(ctx2)

    def test_bool_values(self):
        """Bools deben distinguirse."""
        ctx1 = {"enabled": True}
        ctx2 = {"enabled": False}

        assert create_fingerprint(ctx1) != create_fingerprint(ctx2)


class TestFingerprintStructures:
    """Tests para estructuras complejas."""

    def test_nested_dict(self):
        """Dicts nested deben funcionar."""
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
        """Orden de listas debe importar."""
        ctx1 = {"tools": ["search", "calculator"]}
        ctx2 = {"tools": ["calculator", "search"]}

        # Orden diferente = fingerprint diferente
        assert create_fingerprint(ctx1) != create_fingerprint(ctx2)

    def test_list_values(self):
        """Listas con mismos valores en mismo orden."""
        ctx1 = {"tools": ["search", "calculator"]}
        ctx2 = {"tools": ["search", "calculator"]}

        assert create_fingerprint(ctx1) == create_fingerprint(ctx2)

    def test_dict_key_order_irrelevant(self):
        """Orden de keys en dict no debe importar."""
        ctx1 = {"a": 1, "b": 2, "c": 3}
        ctx2 = {"c": 3, "a": 1, "b": 2}

        assert create_fingerprint(ctx1) == create_fingerprint(ctx2)


class TestFingerprintRealWorld:
    """Tests con casos de uso reales."""

    def test_agent_context(self):
        """Contexto típico de agente."""
        ctx = {
            "agent_id": "sql_analyzer",
            "version": "v1.2.3",
            "config": {"database": "production", "timeout": 30, "max_rows": 1000},
            "tools": ["query", "format"],
        }

        fp = create_fingerprint(ctx)
        assert len(fp) == 64

        # Mismo contexto debe dar mismo fingerprint
        assert create_fingerprint(ctx) == fp

    def test_slight_config_change(self):
        """Cambio mínimo en config debe cambiar fingerprint."""
        ctx1 = {"agent": "analyzer", "model": "gpt-4", "temperature": 0.0}
        ctx2 = {"agent": "analyzer", "model": "gpt-4", "temperature": 0.1}

        assert create_fingerprint(ctx1) != create_fingerprint(ctx2)

    def test_multiagent_context(self):
        """Contexto de sistema multi-agente."""
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
    """Tests para casos edge."""

    def test_very_long_values(self):
        """Valores muy largos."""
        ctx = {
            "description": "a" * 10000,  # String de 10k chars
            "value": 42,
        }

        fp = create_fingerprint(ctx)
        assert len(fp) == 64  # Siempre 64 chars

    def test_special_characters(self):
        """Caracteres especiales en strings."""
        ctx = {
            "query": "SELECT * FROM users WHERE name = 'O\"Reilly'",
            "param": "hello\nworld\t!",
        }

        fp1 = create_fingerprint(ctx)
        fp2 = create_fingerprint(ctx)
        assert fp1 == fp2

    def test_unicode_values(self):
        """Valores unicode."""
        ctx1 = {"text": "Hello 世界"}
        ctx2 = {"text": "Hello 世界"}

        assert create_fingerprint(ctx1) == create_fingerprint(ctx2)

    def test_deeply_nested(self):
        """Estructuras profundamente nested."""
        ctx = {"level1": {"level2": {"level3": {"level4": {"value": 42}}}}}

        fp = create_fingerprint(ctx)
        assert len(fp) == 64

    def test_mixed_number_types(self):
        """Mezcla de ints y floats."""
        ctx1 = {"int_val": 10, "float_val": 3.14, "int_as_float": 20.0}
        ctx2 = {
            "int_val": 10,
            "float_val": 3.14,
            "int_as_float": 20,  # Int, no float
        }

        assert create_fingerprint(ctx1) == create_fingerprint(ctx2)


class TestFingerprintDeterminism:
    """Tests de determinismo."""

    def test_multiple_calls_same_result(self):
        """Múltiples llamadas deben dar mismo resultado."""
        ctx = {"agent": "test", "params": {"a": 1, "b": 2}, "tools": ["t1", "t2"]}

        fingerprints = [create_fingerprint(ctx) for _ in range(100)]

        # Todos deben ser iguales
        assert len(set(fingerprints)) == 1

    def test_different_contexts_different_fingerprints(self):
        """Contextos diferentes deben tener fingerprints diferentes."""
        contexts = [
            {"agent": "a"},
            {"agent": "b"},
            {"agent": "a", "version": "v1"},
            {"agent": "a", "version": "v2"},
            {},
        ]

        fingerprints = [create_fingerprint(ctx) for ctx in contexts]

        # Todos deben ser únicos
        assert len(set(fingerprints)) == len(contexts)
