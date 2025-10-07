"""Tests for text logging (json_logs=False)."""


def test_config_has_text_logging(text_logging_env):
    """Config should have json_logs=False."""
    from reminiscence import ReminiscenceConfig

    config = ReminiscenceConfig.load()
    assert config.json_logs is False


def test_initialization_logs_text(text_logging_env, capsys):
    """Should log initialization in text format."""
    from reminiscence import Reminiscence, ReminiscenceConfig

    config = ReminiscenceConfig.load()
    _ = Reminiscence(config)

    captured = capsys.readouterr()
    assert (
        "initializing_reminiscence" in captured.out
        or "reminiscence_ready" in captured.out
    )


def test_cache_hit_logs_text(text_logging_env, capsys):
    """Should log cache hit in text format."""
    from reminiscence import Reminiscence, ReminiscenceConfig

    config = ReminiscenceConfig.load()
    reminiscence = Reminiscence(config)

    reminiscence.store("test query", {"agent": "test"}, "test result")
    capsys.readouterr()

    result = reminiscence.lookup("test query", {"agent": "test"})
    captured = capsys.readouterr()

    assert result.is_hit
    assert "cache_hit" in captured.out


def test_eviction_logs_text(text_logging_env, monkeypatch, capsys):
    """Should log eviction in text format."""
    from reminiscence import Reminiscence, ReminiscenceConfig

    monkeypatch.setenv("REMINISCENCE_MAX_ENTRIES", "2")

    config = ReminiscenceConfig.load()
    reminiscence = Reminiscence(config)

    reminiscence.store("q1", {"agent": "test"}, "r1")
    reminiscence.store("q2", {"agent": "test"}, "r2")
    capsys.readouterr()

    reminiscence.store("q3", {"agent": "test"}, "r3")
    captured = capsys.readouterr()

    assert reminiscence.backend.count() == 2
    assert "evict" in captured.out.lower()


def test_operations_work_with_text_logging(text_logging_env):
    """All operations should work with text logging."""
    from reminiscence import Reminiscence, ReminiscenceConfig

    config = ReminiscenceConfig.load()
    reminiscence = Reminiscence(config)

    reminiscence.store("test", {"agent": "test"}, "result")
    result = reminiscence.lookup("test", {"agent": "test"})

    assert result.is_hit
    assert reminiscence.backend.count() == 1
