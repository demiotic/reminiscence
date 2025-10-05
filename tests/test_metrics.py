"""Tests para CacheMetrics."""

import pytest
from memora import CacheMetrics


def test_metrics_initialization():
    """Test inicialización de métricas."""
    metrics = CacheMetrics()
    assert metrics.hits == 0
    assert metrics.misses == 0
    assert metrics.total_latency_saved_ms == 0.0


def test_hit_rate_empty():
    """Test hit rate con 0 queries."""
    metrics = CacheMetrics()
    assert metrics.hit_rate() == 0.0


def test_hit_rate_calculation():
    """Test cálculo de hit rate."""
    metrics = CacheMetrics(hits=3, misses=7)
    assert metrics.hit_rate() == 0.3  # 30%


def test_hit_rate_perfect():
    """Test hit rate perfecto."""
    metrics = CacheMetrics(hits=10, misses=0)
    assert metrics.hit_rate() == 1.0  # 100%


def test_metrics_report():
    """Test reporte de métricas."""
    metrics = CacheMetrics(hits=5, misses=5, total_latency_saved_ms=10000)

    report = metrics.report()
    assert report["hits"] == 5
    assert report["misses"] == 5
    assert report["hit_rate"] == "50.00%"
    assert report["latency_saved_ms"] == "10000.0"
