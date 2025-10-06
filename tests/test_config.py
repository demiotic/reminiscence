"""Tests for memora.config.CacheConfig."""

import os
from memora import CacheConfig


class TestConfigDefaults:
    """Test default configuration values."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = CacheConfig()

        assert config.model_name == "paraphrase-multilingual-MiniLM-L12-v2"
        assert config.use_onnx is True
        assert config.onnx_model_file == "model.onnx"
        assert config.similarity_threshold == 0.85
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
        # Clear all MEMORA_* env vars
        for key in list(os.environ.keys()):
            if key.startswith("MEMORA_"):
                monkeypatch.delenv(key, raising=False)

        config = CacheConfig.load()

        assert config.model_name == "paraphrase-multilingual-MiniLM-L12-v2"
        assert config.use_onnx is True
        assert config.onnx_model_file == "model.onnx"
        assert config.similarity_threshold == 0.85
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
        monkeypatch.setenv("MEMORA_JSON_LOGS", "true")
        monkeypatch.setenv("MEMORA_LOG_LEVEL", "WARNING")

        config = CacheConfig.load()

        assert config.json_logs is True
        assert config.log_level == "WARNING"

    def test_load_with_all_vars_set(self, monkeypatch):
        """Config should read all env vars correctly."""
        monkeypatch.setenv("MEMORA_MODEL_NAME", "all-MiniLM-L6-v2")
        monkeypatch.setenv("MEMORA_USE_ONNX", "false")
        monkeypatch.setenv("MEMORA_ONNX_MODEL_FILE", "model_quint8_avx2.onnx")
        monkeypatch.setenv("MEMORA_SIMILARITY_THRESHOLD", "0.75")
        monkeypatch.setenv("MEMORA_DB_URI", "./test_cache.db")
        monkeypatch.setenv("MEMORA_TABLE_NAME", "custom_cache")
        monkeypatch.setenv("MEMORA_ENABLE_METRICS", "false")
        monkeypatch.setenv("MEMORA_TTL_SECONDS", "7200")
        monkeypatch.setenv("MEMORA_LOG_LEVEL", "debug")
        monkeypatch.setenv("MEMORA_JSON_LOGS", "1")
        monkeypatch.setenv("MEMORA_MAX_ENTRIES", "50000")
        monkeypatch.setenv("MEMORA_EVICTION_POLICY", "lru")
        monkeypatch.setenv("MEMORA_AUTO_CREATE_INDEX", "yes")

        config = CacheConfig.load()

        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.use_onnx is False
        assert config.onnx_model_file == "model_quint8_avx2.onnx"
        assert config.similarity_threshold == 0.75
        assert config.db_uri == "./test_cache.db"
        assert config.table_name == "custom_cache"
        assert config.enable_metrics is False
        assert config.ttl_seconds == 7200
        assert config.log_level == "DEBUG"
        assert config.json_logs is True
        assert config.max_entries == 50_000
        assert config.eviction_policy == "lru"
        assert config.auto_create_index is True

    def test_load_bool_parsing_variations(self, monkeypatch):
        """Test different boolean value formats."""
        # Test "true" variants
        for value in ["true", "True", "TRUE", "1", "yes", "Yes", "on"]:
            monkeypatch.setenv("MEMORA_JSON_LOGS", value)
            config = CacheConfig.load()
            assert config.json_logs is True, f"Failed for value: {value}"

        # Test "false" variants
        for value in ["false", "False", "FALSE", "0", "no", "off", ""]:
            monkeypatch.setenv("MEMORA_JSON_LOGS", value)
            config = CacheConfig.load()
            assert config.json_logs is False, f"Failed for value: {value}"

    def test_load_optional_int_none(self, monkeypatch):
        """Test parsing None for optional int fields."""
        monkeypatch.setenv("MEMORA_TTL_SECONDS", "none")
        monkeypatch.setenv("MEMORA_MAX_ENTRIES", "None")

        config = CacheConfig.load()

        assert config.ttl_seconds is None
        assert config.max_entries is None

    def test_load_preserves_unset_defaults(self, monkeypatch):
        """Only set env vars should override defaults."""
        # Clear all MEMORA_* env vars first
        for key in list(os.environ.keys()):
            if key.startswith("MEMORA_"):
                monkeypatch.delenv(key, raising=False)

        # Only set one env var
        monkeypatch.setenv("MEMORA_JSON_LOGS", "true")

        config = CacheConfig.load()

        # This one should be changed
        assert config.json_logs is True

        # All others should be defaults
        assert config.db_uri == "memory://"
        assert config.max_entries == 1_000
        assert config.log_level == "INFO"

    def test_load_onnx_defaults(self, monkeypatch):
        """Test ONNX configuration defaults."""
        # Clear env vars
        for key in list(os.environ.keys()):
            if key.startswith("MEMORA_"):
                monkeypatch.delenv(key, raising=False)

        config = CacheConfig.load()

        assert config.use_onnx is True
        assert config.onnx_model_file == "model.onnx"

    def test_load_onnx_custom(self, monkeypatch):
        """Test custom ONNX configuration."""
        monkeypatch.setenv("MEMORA_USE_ONNX", "true")
        monkeypatch.setenv("MEMORA_ONNX_MODEL_FILE", "model_qint8_avx512_vnni.onnx")

        config = CacheConfig.load()

        assert config.use_onnx is True
        assert config.onnx_model_file == "model_qint8_avx512_vnni.onnx"
