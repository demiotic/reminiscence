"""Tests for memora.config.CacheConfig."""

import os
from memora import CacheConfig


class TestConfigDefaults:
    """Test default configuration values."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = CacheConfig()

        assert config.model_name == "paraphrase-multilingual-MiniLM-L12-v2"
        assert config.similarity_threshold == 0.85
        assert config.db_uri == "memory://"
        assert config.table_name == "semantic_cache"
        assert config.enable_metrics is True
        assert config.ttl_seconds is None
        assert config.log_level == "INFO"
        assert config.json_logs is False
        assert config.max_entries == 10_000
        assert config.max_result_size_bytes == 10_000_000
        assert config.eviction_policy == "fifo"


class TestConfigPresets:
    """Test configuration presets."""

    def test_for_production_preset(self):
        """Production preset should have correct values."""
        config = CacheConfig.for_production()

        assert config.db_uri == "./memora_cache"
        assert config.ttl_seconds == 3600
        assert config.enable_metrics is True
        assert config.log_level == "INFO"
        assert config.json_logs is True  # JSON logs in production
        assert config.auto_create_index is True
        assert config.index_threshold_entries == 1000
        assert config.index_num_partitions == 512
        assert config.max_entries == 50_000
        assert config.max_result_size_bytes == 10_000_000
        assert config.eviction_policy == "fifo"

    def test_for_production_custom_path(self):
        """Production preset should accept custom db path."""
        config = CacheConfig.for_production(db_path="/var/cache/memora")

        assert config.db_uri == "/var/cache/memora"
        assert config.json_logs is True

    def test_for_development_preset(self):
        """Development preset should have correct values."""
        config = CacheConfig.for_development()

        assert config.db_uri == "memory://"
        assert config.ttl_seconds == 300
        assert config.enable_metrics is True
        assert config.log_level == "DEBUG"
        assert config.json_logs is False  # Console logs in development
        assert config.auto_create_index is False
        assert config.max_entries == 1_000
        assert config.max_result_size_bytes == 5_000_000
        assert config.eviction_policy == "fifo"

    def test_presets_can_be_overridden(self):
        """Preset values can be modified after creation."""
        config = CacheConfig.for_production()

        # Override some values
        config.json_logs = False
        config.log_level = "DEBUG"
        config.max_entries = 100_000

        assert config.json_logs is False
        assert config.log_level == "DEBUG"
        assert config.max_entries == 100_000


class TestConfigFromEnv:
    """Environment variable configuration tests."""

    def test_from_env_defaults(self, monkeypatch):
        """Config should use defaults when env vars not set."""
        # Clear all MEMORA_* env vars
        for key in list(os.environ.keys()):
            if key.startswith("MEMORA_"):
                monkeypatch.delenv(key, raising=False)

        config = CacheConfig.from_env()

        assert config.model_name == "paraphrase-multilingual-MiniLM-L12-v2"
        assert config.similarity_threshold == 0.85
        assert config.db_uri == "memory://"
        assert config.table_name == "semantic_cache"
        assert config.enable_metrics is True
        assert config.ttl_seconds is None
        assert config.log_level == "INFO"
        assert config.json_logs is False
        assert config.max_entries == 10_000
        assert config.max_result_size_bytes == 10_000_000
        assert config.eviction_policy == "fifo"

    def test_from_env_with_json_logs_enabled(self, monkeypatch):
        """Config should read json_logs from env var."""
        monkeypatch.setenv("MEMORA_JSON_LOGS", "true")
        monkeypatch.setenv("MEMORA_LOG_LEVEL", "WARNING")

        config = CacheConfig.from_env()

        assert config.json_logs is True
        assert config.log_level == "WARNING"

    def test_from_env_with_all_vars_set(self, monkeypatch):
        """Config should read all env vars correctly."""
        monkeypatch.setenv("MEMORA_MODEL_NAME", "all-MiniLM-L6-v2")
        monkeypatch.setenv("MEMORA_SIMILARITY_THRESHOLD", "0.75")
        monkeypatch.setenv("MEMORA_DB_URI", "./test_cache.db")
        monkeypatch.setenv("MEMORA_TABLE_NAME", "custom_cache")
        monkeypatch.setenv("MEMORA_ENABLE_METRICS", "false")
        monkeypatch.setenv("MEMORA_TTL_SECONDS", "7200")
        monkeypatch.setenv("MEMORA_LOG_LEVEL", "debug")
        monkeypatch.setenv("MEMORA_JSON_LOGS", "1")  # Test "1" as true
        monkeypatch.setenv("MEMORA_MAX_ENTRIES", "50000")
        monkeypatch.setenv("MEMORA_MAX_RESULT_SIZE_BYTES", "5000000")
        monkeypatch.setenv("MEMORA_EVICTION_POLICY", "lru")
        monkeypatch.setenv("MEMORA_AUTO_CREATE_INDEX", "yes")  # Test "yes" as true

        config = CacheConfig.from_env()

        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.similarity_threshold == 0.75
        assert config.db_uri == "./test_cache.db"
        assert config.table_name == "custom_cache"
        assert config.enable_metrics is False
        assert config.ttl_seconds == 7200
        assert config.log_level == "DEBUG"  # Should be uppercased
        assert config.json_logs is True
        assert config.max_entries == 50_000
        assert config.max_result_size_bytes == 5_000_000
        assert config.eviction_policy == "lru"
        assert config.auto_create_index is True

    def test_from_env_bool_parsing_variations(self, monkeypatch):
        """Test different boolean value formats."""
        # Test "true" variants
        for value in ["true", "True", "TRUE", "1", "yes", "Yes", "on"]:
            monkeypatch.setenv("MEMORA_JSON_LOGS", value)
            config = CacheConfig.from_env()
            assert config.json_logs is True, f"Failed for value: {value}"

        # Test "false" variants
        for value in ["false", "False", "FALSE", "0", "no", "off", ""]:
            monkeypatch.setenv("MEMORA_JSON_LOGS", value)
            config = CacheConfig.from_env()
            assert config.json_logs is False, f"Failed for value: {value}"

    def test_from_env_optional_int_none(self, monkeypatch):
        """Test parsing None for optional int fields."""
        monkeypatch.setenv("MEMORA_TTL_SECONDS", "none")
        monkeypatch.setenv("MEMORA_MAX_ENTRIES", "None")

        config = CacheConfig.from_env()

        assert config.ttl_seconds is None
        assert config.max_entries is None

    def test_from_env_preserves_unset_defaults(self, monkeypatch):
        """Only set env vars should override defaults."""
        # Clear all MEMORA_* env vars first
        for key in list(os.environ.keys()):
            if key.startswith("MEMORA_"):
                monkeypatch.delenv(key, raising=False)

        # Only set one env var
        monkeypatch.setenv("MEMORA_JSON_LOGS", "true")

        config = CacheConfig.from_env()

        # This one should be changed
        assert config.json_logs is True

        # All others should be defaults
        assert config.db_uri == "memory://"
        assert config.max_entries == 10_000
        assert config.log_level == "INFO"
