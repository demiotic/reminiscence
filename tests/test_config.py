"""Tests for reminiscence.config.ReminiscenceConfig."""

import os
import pytest
import tempfile
from pathlib import Path
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


class TestEncryptionConfig:
    """Tests for encryption configuration."""

    def test_encryption_disabled_by_default(self):
        """Encryption should be disabled by default."""
        config = ReminiscenceConfig()
        assert not config.encryption_enabled
        assert config.encryption_key is None
        assert config.encryption_backend is None

    def test_encryption_requires_key(self):
        """Should raise if encryption_enabled but no key."""
        with pytest.raises(ValueError, match="encryption_key is required"):
            ReminiscenceConfig(encryption_enabled=True)

    def test_auto_detect_age_private_key(self, age_private_key):
        """Should auto-detect age from private key."""
        config = ReminiscenceConfig(
            encryption_enabled=True,
            encryption_key=age_private_key,
        )
        assert config.encryption_backend == "age"

    def test_auto_detect_age_public_key(self, age_public_key):
        """Should auto-detect age from public key."""
        config = ReminiscenceConfig(
            encryption_enabled=True,
            encryption_key=age_public_key,
        )
        assert config.encryption_backend == "age"

    def test_auto_detect_aws_kms(self):
        """Should auto-detect AWS KMS from ARN."""
        config = ReminiscenceConfig(
            encryption_enabled=True,
            encryption_key="arn:aws:kms:us-east-1:123456789:key/abc-123",
        )
        assert config.encryption_backend == "aws-kms"

    def test_auto_detect_gcp_kms(self):
        """Should auto-detect GCP KMS from resource name."""
        config = ReminiscenceConfig(
            encryption_enabled=True,
            encryption_key="projects/my-project/locations/global/keyRings/ring/cryptoKeys/key",
        )
        assert config.encryption_backend == "gcp-kms"

    def test_auto_detect_azure_keyvault(self):
        """Should auto-detect Azure Key Vault from URI."""
        config = ReminiscenceConfig(
            encryption_enabled=True,
            encryption_key="https://my-vault.vault.azure.net/keys/my-key/version",
        )
        assert config.encryption_backend == "azure-keyvault"

    def test_auto_detect_vault(self):
        """Should auto-detect HashiCorp Vault from path."""
        config = ReminiscenceConfig(
            encryption_enabled=True,
            encryption_key="secret/data/cache-key",
        )
        assert config.encryption_backend == "vault"

    def test_explicit_backend_overrides_detection(self, age_private_key):
        """Explicit backend should override auto-detection."""
        config = ReminiscenceConfig(
            encryption_enabled=True,
            encryption_key=age_private_key,
            encryption_backend="age",
        )
        assert config.encryption_backend == "age"

    def test_read_key_from_file_with_file_prefix(self, age_private_key):
        """Should read key from file:// path."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".key") as f:
            f.write(age_private_key)
            key_path = f.name

        try:
            config = ReminiscenceConfig(
                encryption_enabled=True,
                encryption_key=f"file://{key_path}",
            )
            assert config.encryption_backend == "age"
            assert config.encryption_key == age_private_key
        finally:
            Path(key_path).unlink()

    def test_read_key_from_file_without_prefix(self, age_private_key):
        """Should read key from plain file path."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".key") as f:
            f.write(age_private_key)
            key_path = f.name

        try:
            config = ReminiscenceConfig(
                encryption_enabled=True,
                encryption_key=key_path,
            )
            assert config.encryption_backend == "age"
            assert config.encryption_key == age_private_key
        finally:
            Path(key_path).unlink()

    def test_invalid_key_format_raises(self):
        """Should raise if key format cannot be detected."""
        with pytest.raises(ValueError, match="Cannot auto-detect encryption backend"):
            ReminiscenceConfig(
                encryption_enabled=True,
                encryption_key="invalid-key-format-xyz",
            )

    def test_encryption_max_workers_default(self):
        """Should have default max_workers."""
        config = ReminiscenceConfig()
        assert config.encryption_max_workers == 4
