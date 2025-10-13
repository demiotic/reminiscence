"""Tests for JSON logging (json_logs=True)."""


def test_config_has_json_logging(json_logging_env):
    """Config should have json_logs=True."""
    from reminiscence import ReminiscenceConfig

    config = ReminiscenceConfig.load()
    assert config.json_logs is True


def test_initialization_logs_json(json_logging_env, capsys):
    """Should log initialization in JSON format."""
    from reminiscence import Reminiscence, ReminiscenceConfig
    from reminiscence.types import MultiModalInput

    config = ReminiscenceConfig.load()
    _ = Reminiscence(config)

    captured = capsys.readouterr()

    # Should have JSON logs
    assert "{" in captured.out
    assert '"event"' in captured.out or '"level"' in captured.out


def test_cache_hit_logs_json(json_logging_env, capsys):
    """Should log cache hit in JSON format."""
    from reminiscence import Reminiscence, ReminiscenceConfig
    from reminiscence.types import MultiModalInput

    config = ReminiscenceConfig.load()
    reminiscence = Reminiscence(config)

    reminiscence.store(
        MultiModalInput(text="test query"), {"agent": "test"}, "test result"
    )
    capsys.readouterr()

    result = reminiscence.lookup(MultiModalInput(text="test query"), {"agent": "test"})
    captured = capsys.readouterr()

    assert result.is_hit
    assert "{" in captured.out
    assert "cache_hit" in captured.out


def test_eviction_logs_json(json_logging_env, monkeypatch, capsys):
    """Should log eviction in JSON format."""
    from reminiscence import Reminiscence, ReminiscenceConfig
    from reminiscence.types import MultiModalInput

    monkeypatch.setenv("REMINISCENCE_MAX_ENTRIES", "2")

    config = ReminiscenceConfig.load()
    reminiscence = Reminiscence(config)

    reminiscence.store(MultiModalInput(text="q1"), {"agent": "test"}, "r1")
    reminiscence.store(MultiModalInput(text="q2"), {"agent": "test"}, "r2")
    capsys.readouterr()

    reminiscence.store(MultiModalInput(text="q3"), {"agent": "test"}, "r3")
    captured = capsys.readouterr()

    assert reminiscence.backend.count() == 2
    assert "{" in captured.out


def test_operations_work_with_json_logging(json_logging_env):
    """All operations should work with JSON logging."""
    from reminiscence import Reminiscence, ReminiscenceConfig
    from reminiscence.types import MultiModalInput

    config = ReminiscenceConfig.load()
    reminiscence = Reminiscence(config)

    reminiscence.store(MultiModalInput(text="test"), {"agent": "test"}, "result")
    result = reminiscence.lookup(MultiModalInput(text="test"), {"agent": "test"})

    assert result.is_hit
    assert reminiscence.backend.count() == 1
