"""Tests for YAML configuration loading."""

import os
import tempfile
from pathlib import Path

import pytest

from reminiscence.config import ReminiscenceConfig


class TestYAMLConfiguration:
    """Test YAML configuration loading."""

    def test_load_from_yaml_flat_structure(self):
        """Test loading YAML with flat structure."""
        yaml_content = """
grpc_enabled: true
grpc_port: 9090
grpc_max_workers: 25
db_uri: ./test_cache.db
max_entries: 5000
eviction_policy: lru
similarity_threshold: 0.85
enable_metrics: true
log_level: DEBUG
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            config = ReminiscenceConfig.load_from_yaml(
                yaml_path, allow_env_override=False
            )

            assert config.grpc_enabled is True
            assert config.grpc_port == 9090
            assert config.grpc_max_workers == 25
            assert config.db_uri == "./test_cache.db"
            assert config.max_entries == 5000
            assert config.eviction_policy == "lru"
            assert config.similarity_threshold == 0.85
            assert config.enable_metrics is True
            assert config.log_level == "DEBUG"
        finally:
            os.unlink(yaml_path)

    def test_load_from_yaml_nested_structure(self):
        """Test loading YAML with nested structure."""
        yaml_content = """
grpc:
  enabled: true
  port: 8080
  max_workers: 50

embedding:
  backend: fastembed
  batch_size: 64

otel:
  enabled: true
  endpoint: http://localhost:4318/v1/metrics
  service_name: test-service
  export_interval_ms: 30000

compression:
  enabled: true
  algorithm: zstd
  level: 5

encryption:
  enabled: false
  max_workers: 8

index:
  threshold_entries: 512
  num_partitions: 128

db_uri: ./cache.db
max_entries: 100000
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            config = ReminiscenceConfig.load_from_yaml(
                yaml_path, allow_env_override=False
            )

            # gRPC settings
            assert config.grpc_enabled is True
            assert config.grpc_port == 8080
            assert config.grpc_max_workers == 50

            # Embedding settings
            assert config.embedding_backend == "fastembed"
            assert config.embedding_batch_size == 64

            # OTEL settings
            assert config.otel_enabled is True
            assert config.otel_endpoint == "http://localhost:4318/v1/metrics"
            assert config.otel_service_name == "test-service"
            assert config.otel_export_interval_ms == 30000

            # Compression settings
            assert config.compression_enabled is True
            assert config.compression_algorithm == "zstd"
            assert config.compression_level == 5

            # Encryption settings
            assert config.encryption_enabled is False
            assert config.encryption_max_workers == 8

            # Index settings
            assert config.index_threshold_entries == 512
            assert config.index_num_partitions == 128

            # Other settings
            assert config.db_uri == "./cache.db"
            assert config.max_entries == 100000
        finally:
            os.unlink(yaml_path)

    def test_load_from_yaml_with_env_override(self):
        """Test YAML loading with environment variable override."""
        yaml_content = """
grpc:
  enabled: true
  port: 8080
  max_workers: 50

db_uri: ./cache.db
max_entries: 10000
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        # Set environment variables to override YAML
        os.environ["REMINISCENCE_GRPC_PORT"] = "9999"
        os.environ["REMINISCENCE_MAX_ENTRIES"] = "20000"

        try:
            config = ReminiscenceConfig.load_from_yaml(
                yaml_path, allow_env_override=True
            )

            # Environment variables should override YAML values
            assert config.grpc_port == 9999
            assert config.max_entries == 20000

            # Other values from YAML should remain
            assert config.grpc_enabled is True
            assert config.grpc_max_workers == 50
            assert config.db_uri == "./cache.db"
        finally:
            os.unlink(yaml_path)
            del os.environ["REMINISCENCE_GRPC_PORT"]
            del os.environ["REMINISCENCE_MAX_ENTRIES"]

    def test_load_from_yaml_no_env_override(self):
        """Test YAML loading without environment variable override."""
        yaml_content = """
grpc_port: 8080
max_entries: 10000
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        # Set environment variables
        os.environ["REMINISCENCE_GRPC_PORT"] = "9999"
        os.environ["REMINISCENCE_MAX_ENTRIES"] = "20000"

        try:
            config = ReminiscenceConfig.load_from_yaml(
                yaml_path, allow_env_override=False
            )

            # YAML values should NOT be overridden
            assert config.grpc_port == 8080
            assert config.max_entries == 10000
        finally:
            os.unlink(yaml_path)
            del os.environ["REMINISCENCE_GRPC_PORT"]
            del os.environ["REMINISCENCE_MAX_ENTRIES"]

    def test_load_from_yaml_with_context_thresholds(self):
        """Test loading context_thresholds from YAML."""
        yaml_content = """
similarity_threshold: 0.80
context_thresholds:
  "model:gpt-4": 0.90
  "agent:sql": 0.95
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            config = ReminiscenceConfig.load_from_yaml(
                yaml_path, allow_env_override=False
            )

            assert config.similarity_threshold == 0.80
            assert config.context_thresholds == {
                "model:gpt-4": 0.90,
                "agent:sql": 0.95,
            }
        finally:
            os.unlink(yaml_path)

    def test_load_from_yaml_file_not_found(self):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            ReminiscenceConfig.load_from_yaml("nonexistent.yaml")

    def test_load_from_yaml_empty_file(self):
        """Test loading from empty YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            yaml_path = f.name

        try:
            # Empty file should return defaults
            config = ReminiscenceConfig.load_from_yaml(
                yaml_path, allow_env_override=False
            )
            defaults = ReminiscenceConfig()

            assert config.grpc_port == defaults.grpc_port
            assert config.max_entries == defaults.max_entries
        finally:
            os.unlink(yaml_path)

    def test_load_example_yaml_files(self):
        """Test loading example YAML files from examples directory."""
        example_files = [
            "reminiscence-dev.yaml",
            "reminiscence-prod.yaml",
            "reminiscence.yaml",
        ]

        examples_dir = Path(__file__).parent.parent / "examples"

        for filename in example_files:
            yaml_path = examples_dir / filename
            if yaml_path.exists():
                # Should load without errors
                config = ReminiscenceConfig.load_from_yaml(
                    yaml_path, allow_env_override=False
                )
                assert isinstance(config, ReminiscenceConfig)

    def test_flatten_yaml_nested(self):
        """Test _flatten_yaml with nested structure."""
        yaml_data = {
            "grpc": {"enabled": True, "port": 8080},
            "otel": {"enabled": False, "service_name": "test"},
        }

        flattened = ReminiscenceConfig._flatten_yaml(yaml_data)

        assert flattened == {
            "grpc_enabled": True,
            "grpc_port": 8080,
            "otel_enabled": False,
            "otel_service_name": "test",
        }

    def test_flatten_yaml_flat(self):
        """Test _flatten_yaml with flat structure."""
        yaml_data = {
            "grpc_enabled": True,
            "grpc_port": 8080,
            "db_uri": "./cache.db",
        }

        flattened = ReminiscenceConfig._flatten_yaml(yaml_data)

        assert flattened == yaml_data  # Should remain unchanged

    def test_flatten_yaml_mixed(self):
        """Test _flatten_yaml with mixed flat and nested structure."""
        yaml_data = {
            "grpc": {"enabled": True, "port": 8080},
            "db_uri": "./cache.db",
            "max_entries": 10000,
        }

        flattened = ReminiscenceConfig._flatten_yaml(yaml_data)

        assert flattened == {
            "grpc_enabled": True,
            "grpc_port": 8080,
            "db_uri": "./cache.db",
            "max_entries": 10000,
        }

    def test_apply_env_overrides_bool(self):
        """Test _apply_env_overrides with boolean values."""
        config_dict = {"grpc_enabled": False, "enable_metrics": True}

        os.environ["REMINISCENCE_GRPC_ENABLED"] = "true"
        os.environ["REMINISCENCE_ENABLE_METRICS"] = "false"

        try:
            overridden = ReminiscenceConfig._apply_env_overrides(config_dict)

            assert overridden["grpc_enabled"] is True
            assert overridden["enable_metrics"] is False
        finally:
            del os.environ["REMINISCENCE_GRPC_ENABLED"]
            del os.environ["REMINISCENCE_ENABLE_METRICS"]

    def test_apply_env_overrides_int(self):
        """Test _apply_env_overrides with integer values."""
        config_dict = {"grpc_port": 8080, "max_entries": 1000}

        os.environ["REMINISCENCE_GRPC_PORT"] = "9090"
        os.environ["REMINISCENCE_MAX_ENTRIES"] = "5000"

        try:
            overridden = ReminiscenceConfig._apply_env_overrides(config_dict)

            assert overridden["grpc_port"] == 9090
            assert overridden["max_entries"] == 5000
        finally:
            del os.environ["REMINISCENCE_GRPC_PORT"]
            del os.environ["REMINISCENCE_MAX_ENTRIES"]

    def test_apply_env_overrides_float(self):
        """Test _apply_env_overrides with float values."""
        config_dict = {"similarity_threshold": 0.80}

        os.environ["REMINISCENCE_SIMILARITY_THRESHOLD"] = "0.95"

        try:
            overridden = ReminiscenceConfig._apply_env_overrides(config_dict)

            assert overridden["similarity_threshold"] == 0.95
        finally:
            del os.environ["REMINISCENCE_SIMILARITY_THRESHOLD"]

    def test_apply_env_overrides_dict(self):
        """Test _apply_env_overrides with dict values."""
        config_dict = {"context_thresholds": {"model:gpt-4": 0.85}}

        os.environ["REMINISCENCE_CONTEXT_THRESHOLDS"] = '{"agent:sql": 0.95}'

        try:
            overridden = ReminiscenceConfig._apply_env_overrides(config_dict)

            assert overridden["context_thresholds"] == {"agent:sql": 0.95}
        finally:
            del os.environ["REMINISCENCE_CONTEXT_THRESHOLDS"]

    def test_apply_env_overrides_none(self):
        """Test _apply_env_overrides with None values."""
        config_dict = {"ttl_seconds": 3600}

        os.environ["REMINISCENCE_TTL_SECONDS"] = "none"

        try:
            overridden = ReminiscenceConfig._apply_env_overrides(config_dict)

            assert overridden["ttl_seconds"] is None
        finally:
            del os.environ["REMINISCENCE_TTL_SECONDS"]
