"""Tests for text logging (json_logs=False)."""

import pytest


def test_config_has_text_logging(text_logging_env):
    """Config should have json_logs=False."""
    from memora import CacheConfig

    config = CacheConfig.load()
    assert config.json_logs is False


def test_initialization_logs_text(text_logging_env, capsys):
    """Should log initialization in text format."""
    from memora import Memora, CacheConfig

    config = CacheConfig.load()
    memora = Memora(config)

    captured = capsys.readouterr()
    assert "initializing_memora" in captured.out or "memora_ready" in captured.out


def test_cache_hit_logs_text(text_logging_env, capsys):
    """Should log cache hit in text format."""
    from memora import Memora, CacheConfig

    config = CacheConfig.load()
    memora = Memora(config)

    memora.store("test query", {"agent": "test"}, "test result")
    capsys.readouterr()

    result = memora.lookup("test query", {"agent": "test"})
    captured = capsys.readouterr()

    assert result.is_hit
    assert "cache_hit" in captured.out


def test_eviction_logs_text(text_logging_env, monkeypatch, capsys):
    """Should log eviction in text format."""
    from memora import Memora, CacheConfig

    monkeypatch.setenv("MEMORA_MAX_ENTRIES", "2")

    config = CacheConfig.load()
    memora = Memora(config)

    memora.store("q1", {"agent": "test"}, "r1")
    memora.store("q2", {"agent": "test"}, "r2")
    capsys.readouterr()

    memora.store("q3", {"agent": "test"}, "r3")
    captured = capsys.readouterr()

    assert memora.backend.count() == 2
    assert "evict" in captured.out.lower()


def test_operations_work_with_text_logging(text_logging_env):
    """All operations should work with text logging."""
    from memora import Memora, CacheConfig

    config = CacheConfig.load()
    memora = Memora(config)

    memora.store("test", {"agent": "test"}, "result")
    result = memora.lookup("test", {"agent": "test"})

    assert result.is_hit
    assert memora.backend.count() == 1
