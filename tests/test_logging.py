"""Tests for structured logging functionality."""

import json
import logging
import sys
from io import StringIO

import pytest
from memora import Memora, CacheConfig


class TestLoggingConfiguration:
    """Test logging configuration options."""

    def test_json_logs_disabled_by_default(self):
        """Default config should have json_logs=False."""
        config = CacheConfig()
        assert config.json_logs is False

    def test_production_preset_enables_json_logs(self):
        """Production preset should enable JSON logs."""
        config = CacheConfig.for_production()
        assert config.json_logs is True

    def test_development_preset_disables_json_logs(self):
        """Development preset should use console logs."""
        config = CacheConfig.for_development()
        assert config.json_logs is False

    def test_json_logs_can_be_toggled(self):
        """json_logs setting should be configurable."""
        config = CacheConfig(json_logs=True)
        assert config.json_logs is True

        config = CacheConfig(json_logs=False)
        assert config.json_logs is False


class TestStructuredLogging:
    """Structured logging integration tests."""

    def test_structured_logging_with_json_enabled(self):
        """Test that JSON logs are produced when enabled."""
        # Capture stdout
        captured_output = StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output

        try:
            config = CacheConfig(
                db_uri="memory://",
                json_logs=True,
                log_level="INFO",
                enable_metrics=True,
            )
            memora = Memora(config)

            # Trigger log events
            memora.store("test query", {"agent": "test"}, "result")
            memora.lookup("test query", {"agent": "test"})

            # Restore stdout
            sys.stdout = original_stdout
            output = captured_output.getvalue()

            print(f"\n[DEBUG] Captured output:\n{output}")

            # Parse output lines as JSON
            lines = [line.strip() for line in output.split("\n") if line.strip()]
            json_logs_found = 0

            for line in lines:
                try:
                    log_entry = json.loads(line)
                    json_logs_found += 1
                    print(f"[DEBUG] Valid JSON log entry: {log_entry}")

                    # Verify standard structured log fields exist
                    assert "event" in log_entry or "message" in log_entry

                except json.JSONDecodeError:
                    # Some lines might be from other sources
                    continue

            # Should have at least one JSON log
            assert json_logs_found > 0, (
                f"Expected JSON logs but found none in:\n{output}"
            )

        finally:
            sys.stdout = original_stdout

    def test_structured_logging_with_console_logs(self):
        """Test that console logs work when json_logs=False."""
        config = CacheConfig(
            db_uri="memory://",
            json_logs=False,
            log_level="INFO",
            enable_metrics=True,
        )
        memora = Memora(config)

        # Should initialize without errors
        assert memora is not None
        assert memora.config.json_logs is False

        # Operations should work normally
        memora.store("test query", {"agent": "test"}, "result")
        result = memora.lookup("test query", {"agent": "test"})

        assert result.is_hit

    def test_from_env_json_logs_integration(self, monkeypatch):
        """Test full integration: env var → config → Memora."""
        monkeypatch.setenv("MEMORA_JSON_LOGS", "true")
        monkeypatch.setenv("MEMORA_LOG_LEVEL", "WARNING")
        monkeypatch.setenv("MEMORA_DB_URI", "memory://")

        config = CacheConfig.from_env()
        memora = Memora(config)

        assert memora.config.json_logs is True
        assert memora.config.log_level == "WARNING"

        # Should work normally
        memora.store("query", {"agent": "test"}, "result")
        result = memora.lookup("query", {"agent": "test"})
        assert result.is_hit


class TestLoggingLevels:
    """Test different logging levels."""

    def test_debug_level_logs_verbose_info(self):
        """DEBUG level should be configurable and work correctly."""
        config = CacheConfig(
            db_uri="memory://",
            log_level="DEBUG",
            json_logs=False,
            enable_metrics=True,
        )
        memora = Memora(config)

        assert memora.config.log_level == "DEBUG"

        # Should work without errors
        memora.store("query", {"agent": "test"}, "result")
        result = memora.lookup("query", {"agent": "test"})
        assert result.is_hit

    def test_info_level_default(self):
        """INFO should be the default log level."""
        config = CacheConfig(db_uri="memory://")
        memora = Memora(config)

        assert memora.config.log_level == "INFO"

        # Should work normally
        memora.store("query", {"agent": "test"}, "result")
        result = memora.lookup("query", {"agent": "test"})
        assert result.is_hit

    def test_warning_level_configuration(self):
        """WARNING level should be configurable."""
        config = CacheConfig(
            db_uri="memory://",
            log_level="WARNING",
            json_logs=False,
        )
        memora = Memora(config)

        assert memora.config.log_level == "WARNING"

        # Operations should work
        memora.store("query", {"agent": "test"}, "result")
        result = memora.lookup("query", {"agent": "test"})
        assert result.is_hit

    def test_error_level_configuration(self):
        """ERROR level should be configurable."""
        config = CacheConfig(
            db_uri="memory://",
            log_level="ERROR",
            json_logs=False,
        )
        memora = Memora(config)

        assert memora.config.log_level == "ERROR"

        # Operations should work
        memora.store("query", {"agent": "test"}, "result")
        result = memora.lookup("query", {"agent": "test"})
        assert result.is_hit

    def test_different_log_levels_all_functional(self):
        """Test that cache works correctly at all log levels."""
        log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]

        for level in log_levels:
            config = CacheConfig(
                db_uri="memory://",
                log_level=level,
                json_logs=False,
            )
            memora = Memora(config)

            assert memora.config.log_level == level

            # All should work normally
            memora.store(f"query_{level}", {"agent": "test"}, f"result_{level}")
            result = memora.lookup(f"query_{level}", {"agent": "test"})
            assert result.is_hit, f"Failed at log level: {level}"


class TestLoggingFunctionality:
    """Test that logging doesn't break core functionality."""

    def test_logging_with_unicode_queries(self, memora_memory):
        """Logging should handle unicode queries without errors."""
        query = "¿Qué es Python? 🐍 中文 العربية"
        memora_memory.store(query, {"agent": "test"}, "result")
        result = memora_memory.lookup(query, {"agent": "test"})

        # Should work without errors
        assert result.is_hit

    def test_logging_with_large_context(self, memora_memory):
        """Logging should handle large context objects."""
        large_context = {
            "agent": "test",
            "config": {"param_" + str(i): f"value_{i}" for i in range(100)},
            "tools": [f"tool_{i}" for i in range(50)],
        }

        memora_memory.store("query", large_context, "result")
        result = memora_memory.lookup("query", large_context)

        # Should work without errors
        assert result.is_hit

    def test_logging_during_eviction(self):
        """Eviction should work correctly with logging enabled."""
        import time

        config = CacheConfig(
            db_uri="memory://",
            max_entries=2,
            log_level="DEBUG",
            json_logs=False,
            enable_metrics=True,
        )
        memora = Memora(config)

        # Use SEMANTICALLY DISTINCT queries
        queries = [
            "The capital of France is Paris",  # Geography
            "Python is a programming language",  # Technology
            "Water boils at 100 degrees Celsius",  # Physics
        ]

        # Store 3 entries to trigger eviction
        for i, query in enumerate(queries):
            memora.store(query, {"agent": "test"}, f"result{i + 1}")
            time.sleep(0.01)

        # Verify eviction happened
        assert memora.table.count_rows() == 2

        # First entry should be evicted
        result1 = memora.lookup(queries[0], {"agent": "test"})
        assert result1.is_miss

        # Last two should remain
        result2 = memora.lookup(queries[1], {"agent": "test"})
        result3 = memora.lookup(queries[2], {"agent": "test"})
        assert result2.is_hit
        assert result3.is_hit

    def test_logging_during_errors(self, memora_memory, monkeypatch):
        """Errors should be handled gracefully with logging."""
        # Store data first
        memora_memory.store("initial", {"agent": "test"}, "data")

        # Force an error
        def failing_embed(text):
            raise RuntimeError("Embedding failed")

        monkeypatch.setattr(memora_memory, "_embed", failing_embed)

        # Should return MISS without crashing
        result = memora_memory.lookup("query", {"agent": "test"})
        assert result.is_miss

        # Error should be tracked in metrics
        assert memora_memory.metrics.lookup_errors >= 1
