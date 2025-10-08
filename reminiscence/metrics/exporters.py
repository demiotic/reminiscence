"""Metrics exporters for external systems."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from opentelemetry import metrics
from opentelemetry.metrics import Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter


class MetricsExporter(ABC):
    """Abstract base for metrics exporters."""

    @abstractmethod
    def export(self, metrics: Dict[str, Any]):
        """Export metrics to external system."""
        pass

    @classmethod
    def _clear_instances(cls):
        """Clear all singleton instances (for testing only)."""
        cls._instance = None


class OpenTelemetryExporter(MetricsExporter):
    """
    OpenTelemetry metrics exporter (Singleton).

    Exports Reminiscence cache metrics to OTLP-compatible backends
    (Grafana, Prometheus via OTLP, Jaeger, SigNoz, etc.)

    This class uses a singleton pattern to ensure only one MeterProvider
    is configured globally, preventing conflicts with OpenTelemetry's
    global state management.

    Example:
        >>> # Automatic creation via Reminiscence
        >>> cache = Reminiscence()
        >>> if cache.otel_exporter:
        ...     # Exporter is ready, metrics will be exported automatically
        ...     pass

        >>> # Manual creation (not recommended, but safe due to singleton)
        >>> exporter = OpenTelemetryExporter.from_config(config)
        >>> # Second call returns the same instance
        >>> exporter2 = OpenTelemetryExporter.from_config(config)
        >>> assert exporter is exporter2  # True

    Supports authentication via headers for services like:
    - SigNoz (local or cloud)
    - Grafana Cloud (requires API token)
    - New Relic (requires license key)
    - Honeycomb (requires API key)
    - Elastic APM (requires secret token)
    """

    _instance: Optional["OpenTelemetryExporter"] = None

    def __new__(cls, *args, **kwargs):
        """
        Singleton pattern: only one instance globally.

        Ensures that multiple instantiation attempts return the same object,
        preventing MeterProvider conflicts.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Initialize flag BEFORE __init__ is called
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        endpoint: str = "http://localhost:4318/v1/metrics",
        service_name: str = "reminiscence",
        headers: Optional[Dict[str, str]] = None,
        export_interval_ms: int = 60000,
    ):
        """
        Initialize OpenTelemetry exporter (only once per process).

        Args:
            endpoint: OTLP endpoint URL
            service_name: Service name for telemetry
            headers: Optional HTTP headers for authentication
                     Example for Grafana Cloud:
                     {"Authorization": "Basic <base64_token>"}
            export_interval_ms: How often to export metrics in milliseconds
                                Default: 60000 (60 seconds)

        Note:
            Due to singleton pattern, only the first call to __init__ will
            actually configure the exporter. Subsequent calls are no-ops.
        """

        # Only initialize once (check instance flag, not class flag)
        if self._initialized:
            return  # Silent return on subsequent calls

        self.endpoint = endpoint
        self.service_name = service_name
        self.headers = headers or {}
        self.export_interval_ms = export_interval_ms

        # 1. Setup MeterProvider FIRST (before any get_meter call)
        self._setup_meter_provider()

        # 2. THEN get the meter (will use the real MeterProvider we just configured)
        self.meter = metrics.get_meter("reminiscence.cache", version="0.1.0")

        # 3. AND THEN create instruments
        self._create_instruments()

        # Cumulative counters for delta calculation
        self._last_hits = 0
        self._last_misses = 0
        self._last_lookup_errors = 0
        self._last_store_errors = 0

        # Mark as initialized
        self._initialized = True

    def _setup_meter_provider(self):
        """
        Setup MeterProvider with OTLP exporter BEFORE any meter is created.

        This ensures we configure OpenTelemetry correctly before any
        instrumentation code tries to use it.
        """
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME

        # Check current state
        current = metrics.get_meter_provider()
        current_type = current.__class__.__name__

        # Only warn if it's a real MeterProvider (not the default proxy)
        if current_type == "MeterProvider":
            from reminiscence.utils.logging import get_logger

            logger = get_logger(__name__)
            logger.warning(
                "meter_provider_already_configured",
                type=current_type,
                message="A real MeterProvider is already set, reusing it",
            )
            return

        # Configure the real MeterProvider
        resource = Resource.create({SERVICE_NAME: self.service_name})

        exporter = OTLPMetricExporter(
            endpoint=self.endpoint,
            headers=self.headers,
        )

        reader = PeriodicExportingMetricReader(
            exporter=exporter,
            export_interval_millis=self.export_interval_ms,
        )

        provider = MeterProvider(resource=resource, metric_readers=[reader])

        # Set it globally BEFORE calling get_meter()
        metrics.set_meter_provider(provider)

    def _create_instruments(self):
        """Create OpenTelemetry instruments for metrics."""

        # Counters for cumulative values
        self.hits_counter = self.meter.create_counter(
            name="cache_hits",
            description="Number of cache hits",
            unit="1",
        )

        self.misses_counter = self.meter.create_counter(
            name="cache_misses",
            description="Number of cache misses",
            unit="1",
        )

        self.lookup_errors_counter = self.meter.create_counter(
            name="cache_lookup_errors",
            description="Number of lookup errors",
            unit="1",
        )

        self.store_errors_counter = self.meter.create_counter(
            name="cache_store_errors",
            description="Number of store errors",
            unit="1",
        )

        # Histograms for distributions
        self.lookup_latency_histogram = self.meter.create_histogram(
            name="cache_lookup_latency",
            description="Cache lookup latency distribution",
            unit="ms",
        )

        self.result_size_histogram = self.meter.create_histogram(
            name="cache_result_size",
            description="Cached result size distribution",
            unit="bytes",
        )

        # Observable gauge for hit rate
        self._current_hit_rate = 0.0

        def get_hit_rate(options):
            """Callback that returns list of Observations."""
            return [Observation(value=self._current_hit_rate)]

        self.hit_rate_gauge = self.meter.create_observable_gauge(
            name="cache_hit_rate",
            callbacks=[get_hit_rate],
            description="Current cache hit rate (0.0-1.0)",
            unit="1",
        )

    def export(self, metrics_data: Dict[str, Any]):
        """
        Export metrics from CacheMetrics.report() to OpenTelemetry.

        This method is called periodically by the metrics export scheduler
        (if enabled via start_scheduler()) or can be called manually.

        Args:
            metrics_data: Output from CacheMetrics.report()

        Note:
            Metrics are sent using delta aggregation for counters, meaning
            only the change since the last export is sent, not cumulative values.
        """
        from reminiscence.utils.logging import get_logger

        logger = get_logger(__name__)

        # Calculate deltas for counters
        current_hits = metrics_data["hits"]
        current_misses = metrics_data["misses"]
        current_lookup_errors = metrics_data["errors"]["lookup"]
        current_store_errors = metrics_data["errors"]["store"]

        hits_delta = current_hits - self._last_hits
        misses_delta = current_misses - self._last_misses
        lookup_errors_delta = current_lookup_errors - self._last_lookup_errors
        store_errors_delta = current_store_errors - self._last_store_errors

        logger.info(
            "exporting_metrics_to_otel",
            hits_delta=hits_delta,
            misses_delta=misses_delta,
            lookup_errors_delta=lookup_errors_delta,
            store_errors_delta=store_errors_delta,
            current_hits=current_hits,
            current_misses=current_misses,
            hit_rate=metrics_data["hit_rate"],
        )

        # Update counters with deltas
        if hits_delta > 0:
            self.hits_counter.add(hits_delta)
            logger.debug("counter_added", metric="cache_hits", value=hits_delta)
        if misses_delta > 0:
            self.misses_counter.add(misses_delta)
            logger.debug("counter_added", metric="cache_misses", value=misses_delta)
        if lookup_errors_delta > 0:
            self.lookup_errors_counter.add(lookup_errors_delta)
            logger.debug(
                "counter_added", metric="cache_lookup_errors", value=lookup_errors_delta
            )
        if store_errors_delta > 0:
            self.store_errors_counter.add(store_errors_delta)
            logger.debug(
                "counter_added", metric="cache_store_errors", value=store_errors_delta
            )

        # Save current values for next delta
        self._last_hits = current_hits
        self._last_misses = current_misses
        self._last_lookup_errors = current_lookup_errors
        self._last_store_errors = current_store_errors

        # Update hit rate
        hit_rate_value = float(metrics_data["hit_rate"].rstrip("%")) / 100.0
        self._current_hit_rate = hit_rate_value

        logger.debug("hit_rate_updated", value=hit_rate_value)

        # Log summary
        counters_updated = (
            hits_delta > 0
            or misses_delta > 0
            or lookup_errors_delta > 0
            or store_errors_delta > 0
        )

        logger.info(
            "otel_export_complete",
            counters_updated=counters_updated,
            gauge_updated=True,  # Hit rate always updates
        )

    @classmethod
    def from_config(cls, config) -> Optional["OpenTelemetryExporter"]:
        """
        Create exporter from ReminiscenceConfig.

        Args:
            config: ReminiscenceConfig instance

        Returns:
            OpenTelemetryExporter instance or None if disabled

        Example:
            >>> config = ReminiscenceConfig.load()
            >>> exporter = OpenTelemetryExporter.from_config(config)
            >>> if exporter:
            ...     # Exporter is ready
            ...     pass
        """
        if not config.otel_enabled:
            return None

        # Parse headers from string format "key1=value1,key2=value2"
        headers = {}
        if config.otel_headers:
            for pair in config.otel_headers.split(","):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    headers[key.strip()] = value.strip()

        return cls(
            endpoint=config.otel_endpoint,
            service_name=config.otel_service_name,
            headers=headers if headers else None,
            export_interval_ms=config.otel_export_interval_ms,
        )

    @classmethod
    def reset(cls):
        """
        Reset the singleton instance (mainly for testing).

        Warning:
            This should only be used in tests. In production, the singleton
            should persist for the lifetime of the process.

        Example:
            >>> # In tests
            >>> OpenTelemetryExporter.reset()
            >>> exporter = OpenTelemetryExporter(service_name="test")
        """
        cls._instance = None


class PrometheusExporter(MetricsExporter):
    """
    Prometheus metrics exporter.

    TODO: Implement prometheus_client integration.

    Example:
        >>> from prometheus_client import Counter, Histogram, Gauge
        >>>
        >>> class PrometheusExporter(MetricsExporter):
        >>>     def __init__(self):
        >>>         self.cache_hits = Counter('reminiscence_cache_hits_total', 'Cache hits')
        >>>         self.cache_misses = Counter('reminiscence_cache_misses_total', 'Cache misses')
        >>>         self.lookup_latency = Histogram('reminiscence_lookup_latency_seconds', 'Lookup latency')
        >>>
        >>>     def export(self, metrics: Dict[str, Any]):
        >>>         self.cache_hits.inc(metrics['hits'])
        >>>         self.cache_misses.inc(metrics['misses'])
    """

    def export(self, metrics: Dict[str, Any]):
        raise NotImplementedError("Prometheus exporter not yet implemented")
