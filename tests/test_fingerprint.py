"""Tests para fingerprinting."""

import pytest
from memora.utils import create_fingerprint


def test_fingerprint_basic():
    """Test generación básica de fingerprint."""
    ctx = {"tools": ["search"]}
    fp = create_fingerprint(ctx)

    assert isinstance(fp, str)
    assert len(fp) == 16  # Hash truncado a 16 chars


def test_fingerprint_deterministic():
    """Test que fingerprint sea determinístico."""
    ctx = {"tools": ["search", "web_fetch"]}

    fp1 = create_fingerprint(ctx)
    fp2 = create_fingerprint(ctx)

    assert fp1 == fp2


def test_fingerprint_order_independent():
    """Test que orden de tools no importe."""
    ctx1 = {"tools": ["search", "web_fetch"]}
    ctx2 = {"tools": ["web_fetch", "search"]}

    fp1 = create_fingerprint(ctx1)
    fp2 = create_fingerprint(ctx2)

    assert fp1 == fp2  # Mismo fingerprint (sorted)


def test_fingerprint_different_contexts():
    """Test que contextos diferentes den fingerprints diferentes."""
    ctx1 = {"tools": ["search"]}
    ctx2 = {"tools": ["code_execute"]}

    fp1 = create_fingerprint(ctx1)
    fp2 = create_fingerprint(ctx2)

    assert fp1 != fp2


def test_fingerprint_with_constraints():
    """Test fingerprint con constraints."""
    ctx1 = {"tools": ["search"], "constraints": {"format": "markdown"}}
    ctx2 = {"tools": ["search"], "constraints": {"format": "json"}}

    fp1 = create_fingerprint(ctx1)
    fp2 = create_fingerprint(ctx2)

    assert fp1 != fp2  # Constraints diferentes


def test_fingerprint_with_entities():
    """Test fingerprint con entities."""
    ctx1 = {"tools": ["search"], "entities": ["Python", "FastAPI"]}
    ctx2 = {"tools": ["search"], "entities": ["FastAPI", "Python"]}

    fp1 = create_fingerprint(ctx1)
    fp2 = create_fingerprint(ctx2)

    assert fp1 == fp2  # Entities ordenadas


def test_fingerprint_empty_context():
    """Test fingerprint con contexto vacío."""
    ctx = {}
    fp = create_fingerprint(ctx)

    assert isinstance(fp, str)
    assert len(fp) == 16
