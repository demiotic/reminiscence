"""Tests for reminiscence.config.ReminiscenceConfig."""

import os
from reminiscence import ReminiscenceConfig


class TestConfigDefaults:
    """Test default configuration values."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = ReminiscenceConfig()

        assert config.model_name is None
        assert config.embedding_backend == "fastembed"
        assert config.similarity_threshold == 0.80
        assert config.db_uri == "memory://"
        assert config.table_name == "semantic_cache"
        assert config.enable_metrics is True
        assert config.ttl_seconds is None
        assert config.log_level == "INFO"
        assert config.json_logs is False
        assert config.max_entries == 1_000
        assert config.eviction_policy == "fifo"


class TestConfigLoad:
    """Environment variable configuration tests."""

    def test_load_defaults(self, monkeypatch):
        """Config should use defaults when env vars not set."""
        # Clear all REMINISCENCE_* env vars
        for key in list(os.environ.keys()):
            if key.startswith("REMINISCENCE_"):
                monkeypatch.delenv(key, raising=False)

        config = ReminiscenceConfig.load()

        assert config.model_name is None
        assert config.similarity_threshold == 0.80
        assert config.embedding_backend == "fastembed"
        assert config.db_uri == "memory://"
        assert config.table_name == "semantic_cache"
        assert config.enable_metrics is True
        assert config.ttl_seconds is None
        assert config.log_level == "INFO"
        assert config.json_logs is False
        assert config.max_entries == 1_000
        assert config.eviction_policy == "fifo"

    def test_load_with_json_logs_enabled(self, monkeypatch):
        """Config should read json_logs from env var."""
        monkeypatch.setenv("REMINISCENCE_JSON_LOGS", "true")
        monkeypatch.setenv("REMINISCENCE_LOG_LEVEL", "WARNING")

        config = ReminiscenceConfig.load()

        assert config.json_logs is True
        assert config.log_level == "WARNING"

    def test_load_bool_parsing_variations(self, monkeypatch):
        """Test different boolean value formats."""
        # Test "true" variants
        for value in ["true", "True", "TRUE", "1", "yes", "Yes", "on"]:
            monkeypatch.setenv("REMINISCENCE_JSON_LOGS", value)
            config = ReminiscenceConfig.load()
            assert config.json_logs is True, f"Failed for value: {value}"

        # Test "false" variants
        for value in ["false", "False", "FALSE", "0", "no", "off", ""]:
            monkeypatch.setenv("REMINISCENCE_JSON_LOGS", value)
            config = ReminiscenceConfig.load()
            assert config.json_logs is False, f"Failed for value: {value}"

    def test_load_optional_int_none(self, monkeypatch):
        """Test parsing None for optional int fields."""
        monkeypatch.setenv("REMINISCENCE_TTL_SECONDS", "none")
        monkeypatch.setenv("REMINISCENCE_MAX_ENTRIES", "None")

        config = ReminiscenceConfig.load()

        assert config.ttl_seconds is None
        assert config.max_entries is None

    def test_load_preserves_unset_defaults(self, monkeypatch):
        """Only set env vars should override defaults."""
        # Clear all REMINISCENCE_* env vars first
        for key in list(os.environ.keys()):
            if key.startswith("REMINISCENCE_"):
                monkeypatch.delenv(key, raising=False)

        # Only set one env var
        monkeypatch.setenv("REMINISCENCE_JSON_LOGS", "true")

        config = ReminiscenceConfig.load()

        # This one should be changed
        assert config.json_logs is True

        # All others should be defaults
        assert config.db_uri == "memory://"
        assert config.max_entries == 1_000
        assert config.log_level == "INFO"
