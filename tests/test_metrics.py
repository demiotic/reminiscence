"""Tests for metrics tracking and exporters."""

import pytest
from reminiscence.metrics.tracker import CacheMetrics
from reminiscence.metrics.exporters import OpenTelemetryExporter
from reminiscence.config import ReminiscenceConfig


class TestCacheMetrics:
    """Tests for CacheMetrics tracker."""

    def test_initialization(self):
        """Test metrics initialization with defaults."""
        metrics = CacheMetrics()

        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.total_latency_saved_ms == 0.0
        assert metrics.evictions == 0
        assert metrics.storage_searches == 0
        assert metrics.embedding_generations == 0
        assert metrics.scheduler_runs == 0

    def test_hit_rate_calculation(self):
        """Test hit rate property calculation."""
        metrics = CacheMetrics()

        # No requests
        assert metrics.hit_rate == 0.0

        # 50% hit rate
        metrics.hits = 5
        metrics.misses = 5
        assert metrics.hit_rate == 0.5

        # 100% hit rate
        metrics.hits = 10
        metrics.misses = 0
        assert metrics.hit_rate == 1.0

        # 0% hit rate
        metrics.hits = 0
        metrics.misses = 10
        assert metrics.hit_rate == 0.0

    def test_eviction_rate_calculation(self):
        """Test eviction rate property calculation."""
        metrics = CacheMetrics()

        # No requests
        assert metrics.eviction_rate == 0.0

        # 10% eviction rate
        metrics.hits = 8
        metrics.misses = 2
        metrics.evictions = 1
        assert metrics.eviction_rate == 0.1

        # 50% eviction rate
        metrics.hits = 10
        metrics.misses = 0
        metrics.evictions = 5
        assert metrics.eviction_rate == 0.5

    def test_record_lookup_latency(self):
        """Test lookup latency recording."""
        metrics = CacheMetrics()

        metrics.record_lookup_latency(10.5)
        metrics.record_lookup_latency(20.3)
        metrics.record_lookup_latency(15.7)

        assert len(metrics.lookup_latencies_ms) == 3
        assert metrics.lookup_latencies_ms[0] == 10.5
        assert metrics.lookup_latencies_ms[1] == 20.3
        assert metrics.lookup_latencies_ms[2] == 15.7

    def test_record_lookup_latency_max_samples(self):
        """Test lookup latency respects max_samples limit."""
        metrics = CacheMetrics(max_samples=100)

        # Add more than max_samples
        for i in range(150):
            metrics.record_lookup_latency(float(i))

        # Should only keep last 100
        assert len(metrics.lookup_latencies_ms) == 100
        assert metrics.lookup_latencies_ms[0] == 50.0
        assert metrics.lookup_latencies_ms[-1] == 149.0

    def test_record_result_size(self):
        """Test result size recording."""
        metrics = CacheMetrics()

        metrics.record_result_size(1024)
        metrics.record_result_size(2048)
        metrics.record_result_size(512)

        assert len(metrics.result_sizes_bytes) == 3
        assert metrics.result_sizes_bytes[0] == 1024
        assert metrics.result_sizes_bytes[1] == 2048
        assert metrics.result_sizes_bytes[2] == 512

    def test_get_percentiles_empty(self):
        """Test percentiles with empty list."""
        metrics = CacheMetrics()

        percentiles = metrics.get_percentiles([])

        assert percentiles["p50"] == 0.0
        assert percentiles["p95"] == 0.0
        assert percentiles["p99"] == 0.0

    def test_get_percentiles_single_value(self):
        """Test percentiles with single value."""
        metrics = CacheMetrics()

        percentiles = metrics.get_percentiles([10.0])

        assert percentiles["p50"] == 10.0
        assert percentiles["p95"] == 10.0
        assert percentiles["p99"] == 10.0

    def test_get_percentiles_multiple_values(self):
        """Test percentiles with multiple values."""
        metrics = CacheMetrics()

        # Values from 0 to 99
        values = [float(i) for i in range(100)]
        percentiles = metrics.get_percentiles(values)

        assert percentiles["p50"] == 50.0
        assert percentiles["p95"] == 95.0
        assert percentiles["p99"] == 99.0

    def test_report_basic(self):
        """Test basic metrics report generation."""
        metrics = CacheMetrics()

        metrics.hits = 80
        metrics.misses = 20
        metrics.total_latency_saved_ms = 16000.0
        metrics.lookup_latencies_ms = [10.0, 15.0, 20.0, 25.0, 30.0]
        metrics.result_sizes_bytes = [100, 200, 300, 400, 500]

        report = metrics.report()

        assert report["hits"] == 80
        assert report["misses"] == 20
        assert report["total_requests"] == 100
        assert report["hit_rate"] == "80.00%"
        assert report["total_latency_saved_ms"] == 16000.0
        assert report["avg_latency_saved_ms"] == 200.0
        assert "lookup_latency_ms" in report
        assert "result_size_bytes" in report

    def test_report_with_eviction_metrics(self):
        """Test report includes eviction metrics."""
        metrics = CacheMetrics()

        metrics.hits = 100
        metrics.evictions = 10
        metrics.evictions_by_policy = {"lru": 10}
        metrics.evicted_entry_ages = [60.0, 120.0, 180.0]

        report = metrics.report()

        assert "eviction" in report
        assert report["eviction"]["total_evictions"] == 10
        assert report["eviction"]["eviction_rate"] == "10.00%"
        assert report["eviction"]["by_policy"] == {"lru": 10}
        assert "evicted_entry_age_seconds" in report["eviction"]

    def test_report_with_lfu_metrics(self):
        """Test report includes LFU-specific metrics."""
        metrics = CacheMetrics()

        metrics.hits = 100
        metrics.lfu_total_accesses = 500
        metrics.lfu_evicted_frequencies = [5, 10, 15, 20]

        report = metrics.report()

        assert "eviction" in report
        assert "lfu_metrics" in report["eviction"]
        assert report["eviction"]["lfu_metrics"]["total_accesses"] == 500
        assert "evicted_frequencies" in report["eviction"]["lfu_metrics"]

    def test_report_with_lru_metrics(self):
        """Test report includes LRU-specific metrics."""
        metrics = CacheMetrics()

        metrics.hits = 100
        metrics.lru_total_accesses = 300
        metrics.lru_evicted_recency_seconds = [30.0, 60.0, 90.0]

        report = metrics.report()

        assert "eviction" in report
        assert "lru_metrics" in report["eviction"]
        assert report["eviction"]["lru_metrics"]["total_accesses"] == 300
        assert "evicted_recency_seconds" in report["eviction"]["lru_metrics"]

    def test_report_with_storage_metrics(self):
        """Test report includes storage metrics."""
        metrics = CacheMetrics()

        metrics.storage_searches = 100
        metrics.storage_adds = 20
        metrics.storage_search_latencies_ms = [5.0, 10.0, 15.0]
        metrics.storage_add_latencies_ms = [2.0, 3.0, 4.0]
        metrics.storage_search_errors = 2
        metrics.storage_add_errors = 1

        report = metrics.report()

        assert "storage" in report
        assert report["storage"]["total_searches"] == 100
        assert report["storage"]["total_adds"] == 20
        assert "search_latency_ms" in report["storage"]
        assert "add_latency_ms" in report["storage"]
        assert report["storage"]["errors"]["search"] == 2
        assert report["storage"]["errors"]["add"] == 1

    def test_report_with_embedding_metrics(self):
        """Test report includes embedding metrics."""
        metrics = CacheMetrics()

        metrics.embedding_generations = 50
        metrics.embedding_latencies_ms = [20.0, 25.0, 30.0]
        metrics.embedding_errors = 1

        report = metrics.report()

        assert "embedding" in report
        assert report["embedding"]["total_generations"] == 50
        assert "latency_ms" in report["embedding"]
        assert report["embedding"]["errors"] == 1

    def test_report_with_scheduler_metrics(self):
        """Test report includes scheduler metrics."""
        metrics = CacheMetrics()

        metrics.scheduler_runs = 10
        metrics.scheduler_cleanup_latencies_ms = [100.0, 150.0, 200.0]
        metrics.scheduler_errors = 2

        report = metrics.report()

        assert "scheduler" in report
        assert report["scheduler"]["total_runs"] == 10
        assert "cleanup_latency_ms" in report["scheduler"]
        assert report["scheduler"]["errors"] == 2

    def test_reset(self):
        """Test metrics reset."""
        metrics = CacheMetrics()

        # Set various metrics
        metrics.hits = 100
        metrics.misses = 50
        metrics.lookup_latencies_ms = [10.0, 20.0]
        metrics.evictions = 5
        metrics.storage_searches = 200
        metrics.embedding_generations = 75
        metrics.scheduler_runs = 3

        # Reset
        metrics.reset()

        # Verify all reset to defaults
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert len(metrics.lookup_latencies_ms) == 0
        assert metrics.evictions == 0
        assert metrics.storage_searches == 0
        assert metrics.embedding_generations == 0
        assert metrics.scheduler_runs == 0

    def test_exporter_is_global_singleton(self):
        """Test that exporter is singleton per process (not per service)."""
        exp1 = OpenTelemetryExporter(service_name="service-a")
        exp2 = OpenTelemetryExporter(service_name="service-a")

        # Same instance (singleton)
        assert exp1 is exp2

        # Even with different service name, returns same instance
        # (OpenTelemetry is global per process)
        exp3 = OpenTelemetryExporter(service_name="service-b")
        assert exp1 is exp3

        # Service name is from first initialization
        assert exp1.service_name == "service-a"

    def test_initialization_default(self):
        """Test exporter initialization with defaults."""
        exporter = OpenTelemetryExporter()

        assert exporter.endpoint == "http://localhost:4318/v1/metrics"
        assert exporter.service_name == "reminiscence"
        assert exporter.headers == {}
        assert exporter.export_interval_ms == 60000

    def test_initialization_custom(self):
        """Test exporter initialization with custom values."""
        headers = {"Authorization": "Bearer token123"}

        exporter = OpenTelemetryExporter(
            endpoint="https://example.com/metrics",
            service_name="my-service",
            headers=headers,
            export_interval_ms=30000,
        )

        assert exporter.endpoint == "https://example.com/metrics"
        assert exporter.service_name == "my-service"
        assert exporter.headers == headers
        assert exporter.export_interval_ms == 30000

    def test_export_basic(self):
        """Test basic export functionality."""
        exporter = OpenTelemetryExporter()

        metrics_data = {
            "hits": 100,
            "misses": 20,
            "hit_rate": "83.33%",
            "errors": {
                "lookup": 2,
                "store": 1,
            },
        }

        # Should not raise
        exporter.export(metrics_data)

        # Verify internal counters updated
        assert exporter._last_hits == 100
        assert exporter._last_misses == 20
        assert exporter._last_lookup_errors == 2
        assert exporter._last_store_errors == 1

    def test_export_delta_calculation(self):
        """Test delta calculation between exports."""
        exporter = OpenTelemetryExporter()

        # First export
        metrics_data_1 = {
            "hits": 100,
            "misses": 20,
            "hit_rate": "83.33%",
            "errors": {"lookup": 2, "store": 1},
        }
        exporter.export(metrics_data_1)

        # Second export with increased values
        metrics_data_2 = {
            "hits": 150,  # +50
            "misses": 30,  # +10
            "hit_rate": "83.33%",
            "errors": {"lookup": 3, "store": 2},  # +1, +1
        }
        exporter.export(metrics_data_2)

        # Verify internal state tracks cumulative values
        assert exporter._last_hits == 150
        assert exporter._last_misses == 30
        assert exporter._last_lookup_errors == 3
        assert exporter._last_store_errors == 2

    def test_from_config_disabled(self):
        """Test creating exporter from config when disabled."""
        config = ReminiscenceConfig(otel_enabled=False)

        exporter = OpenTelemetryExporter.from_config(config)

        assert exporter is None

    def test_from_config_enabled(self):
        """Test creating exporter from config when enabled."""
        config = ReminiscenceConfig(
            otel_enabled=True,
            otel_endpoint="https://example.com/metrics",
            otel_service_name="test-service",
            otel_export_interval_ms=15000,
        )

        exporter = OpenTelemetryExporter.from_config(config)

        assert exporter is not None
        assert exporter.endpoint == "https://example.com/metrics"
        assert exporter.service_name == "test-service"
        assert exporter.export_interval_ms == 15000

    def test_from_config_with_headers(self):
        """Test creating exporter with parsed headers."""
        config = ReminiscenceConfig(
            otel_enabled=True,
            otel_headers="Authorization=Bearer token123,X-Custom=value456",
        )

        exporter = OpenTelemetryExporter.from_config(config)

        assert exporter is not None
        assert exporter.headers == {
            "Authorization": "Bearer token123",
            "X-Custom": "value456",
        }

    def test_from_config_with_malformed_headers(self):
        """Test creating exporter with malformed headers."""
        config = ReminiscenceConfig(
            otel_enabled=True,
            otel_headers="validkey=validvalue,invalidheader,another=valid",
        )

        exporter = OpenTelemetryExporter.from_config(config)

        assert exporter is not None
        # Should skip malformed header
        assert exporter.headers == {
            "validkey": "validvalue",
            "another": "valid",
        }


class TestConfigLoadFromEnvironment:
    """Tests for loading OTEL config from environment."""

    def test_otel_config_defaults(self):
        """Test OTEL config default values."""
        config = ReminiscenceConfig()

        assert config.otel_enabled is False
        assert config.otel_endpoint == "http://localhost:4318/v1/metrics"
        assert config.otel_headers is None
        assert config.otel_service_name == "reminiscence"
        assert config.otel_export_interval_ms == 60000

    def test_otel_config_from_env(self, monkeypatch):
        """Test loading OTEL config from environment variables."""
        monkeypatch.setenv("REMINISCENCE_OTEL_ENABLED", "true")
        monkeypatch.setenv("REMINISCENCE_OTEL_ENDPOINT", "https://custom.com/metrics")
        monkeypatch.setenv("REMINISCENCE_OTEL_HEADERS", "key1=value1,key2=value2")
        monkeypatch.setenv("REMINISCENCE_OTEL_SERVICE_NAME", "custom-service")
        monkeypatch.setenv("REMINISCENCE_OTEL_EXPORT_INTERVAL_MS", "30000")

        config = ReminiscenceConfig.load()

        assert config.otel_enabled is True
        assert config.otel_endpoint == "https://custom.com/metrics"
        assert config.otel_headers == "key1=value1,key2=value2"
        assert config.otel_service_name == "custom-service"
        assert config.otel_export_interval_ms == 30000


class TestMetricsIntegration:
    """Integration tests for metrics tracking across components."""

    def test_eviction_policy_integration(self):
        """Test metrics integration with eviction policies."""
        from reminiscence.eviction.lru import LRUPolicy

        metrics = CacheMetrics()
        policy = LRUPolicy(metrics=metrics)

        # Simulate some operations
        policy.on_insert("entry1")
        policy.on_insert("entry2")
        policy.on_access("entry1")

        # Evict oldest
        victim = policy.select_victim()
        policy.on_evict(victim)

        # Verify metrics updated
        assert metrics.evictions == 1
        assert "lru" in metrics.evictions_by_policy
        assert metrics.evictions_by_policy["lru"] == 1
        assert len(metrics.evicted_entry_ages) == 1

    def test_storage_metrics_integration(self):
        """Test metrics tracking in storage operations."""
        metrics = CacheMetrics()

        # Simulate storage operations
        metrics.storage_searches = 10
        metrics.storage_search_latencies_ms = [5.0, 10.0, 15.0]
        metrics.storage_adds = 5
        metrics.storage_add_latencies_ms = [2.0, 3.0]

        report = metrics.report()

        assert report["storage"]["total_searches"] == 10
        assert report["storage"]["total_adds"] == 5
        assert len(report["storage"]["search_latency_ms"]) > 0
        assert len(report["storage"]["add_latency_ms"]) > 0

    def test_end_to_end_metrics_flow(self):
        """Test complete metrics flow from tracking to export."""
        # Create metrics and exporter
        metrics = CacheMetrics()
        config = ReminiscenceConfig(
            otel_enabled=True,
            otel_export_interval_ms=5000,
        )
        exporter = OpenTelemetryExporter.from_config(config)

        # Simulate cache operations
        metrics.hits = 80
        metrics.misses = 20
        metrics.record_lookup_latency(15.0)
        metrics.record_result_size(1024)
        metrics.evictions = 5
        metrics.storage_searches = 100
        metrics.embedding_generations = 80

        # Generate report
        report = metrics.report()

        # Export metrics
        exporter.export(report)

        # Verify report structure
        assert "hits" in report
        assert "storage" in report
        assert "embedding" in report
        assert "eviction" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
