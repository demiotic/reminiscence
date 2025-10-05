import pytest
import pyarrow as pa
from memora.utils import create_fingerprint


class TestFingerprint:
    """Tests para utils/fingerprint.py."""

    def test_fingerprint_basic(self):
        """Fingerprint básico debe generar hash."""
        fp = create_fingerprint({"agent": "test"})

        assert isinstance(fp, str)
        assert len(fp) > 0

    def test_fingerprint_deterministic(self):
        """Mismo contexto debe generar mismo hash."""
        ctx = {"agent": "test", "model": "gpt-4"}

        fp1 = create_fingerprint(ctx)
        fp2 = create_fingerprint(ctx)

        assert fp1 == fp2

    def test_fingerprint_order_independent(self):
        """Orden de keys no debe afectar hash."""
        ctx1 = {"agent": "test", "model": "gpt-4"}
        ctx2 = {"model": "gpt-4", "agent": "test"}

        fp1 = create_fingerprint(ctx1)
        fp2 = create_fingerprint(ctx2)

        assert fp1 == fp2

    def test_fingerprint_different_values(self):
        """Valores diferentes deben generar hashes diferentes."""
        fp1 = create_fingerprint({"agent": "test1"})
        fp2 = create_fingerprint({"agent": "test2"})

        assert fp1 != fp2

    def test_fingerprint_nested(self):
        """Contexto nested debe funcionar."""
        ctx = {
            "agent": "test",
            "config": {"model": "gpt-4", "params": {"temperature": 0.7}},
        }

        fp = create_fingerprint(ctx)
        assert isinstance(fp, str)

    def test_fingerprint_empty(self):
        """Contexto vacío debe generar hash válido."""
        fp = create_fingerprint({})
        assert isinstance(fp, str)
