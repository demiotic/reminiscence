"""Tests for JSON logging (json_logs=True)."""

import pytest


def test_config_has_json_logging(json_logging_env):
    """Config should have json_logs=True."""
    from memora import CacheConfig

    config = CacheConfig.load()
    assert config.json_logs is True


def test_initialization_logs_json(json_logging_env, capsys):
    """Should log initialization in JSON format."""
    from memora import Memora, CacheConfig

    config = CacheConfig.load()
    memora = Memora(config)

    captured = capsys.readouterr()

    # Should have JSON logs
    assert "{" in captured.out
    assert '"event"' in captured.out or '"level"' in captured.out


def test_cache_hit_logs_json(json_logging_env, capsys):
    """Should log cache hit in JSON format."""
    from memora import Memora, CacheConfig

    config = CacheConfig.load()
    memora = Memora(config)

    memora.store("test query", {"agent": "test"}, "test result")
    capsys.readouterr()

    result = memora.lookup("test query", {"agent": "test"})
    captured = capsys.readouterr()

    assert result.is_hit
    assert "{" in captured.out
    assert "cache_hit" in captured.out


def test_eviction_logs_json(json_logging_env, monkeypatch, capsys):
    """Should log eviction in JSON format."""
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
    assert "{" in captured.out


def test_operations_work_with_json_logging(json_logging_env):
    """All operations should work with JSON logging."""
    from memora import Memora, CacheConfig

    config = CacheConfig.load()
    memora = Memora(config)

    memora.store("test", {"agent": "test"}, "result")
    result = memora.lookup("test", {"agent": "test"})

    assert result.is_hit
    assert memora.backend.count() == 1
