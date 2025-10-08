# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-10-08

### Added

#### Core Features
- **Semantic caching** with FastEmbed multilingual embeddings
  - Hybrid matching: semantic similarity + exact context matching
  - Configurable similarity thresholds (0.75-0.95)
  - Support for 384-dimensional embeddings (paraphrase-multilingual-MiniLM-L12-v2)
- **Multiple eviction policies**
  - FIFO (First In First Out) - Simple chronological eviction
  - LRU (Least Recently Used) - Evict least accessed entries
  - LFU (Least Frequently Used) - Evict entries with lowest access count
  - Configurable max entries with automatic eviction
- **TTL-based expiration**
  - Time-to-live support with configurable seconds
  - Automatic cleanup of expired entries
  - Manual invalidation by context or age
- **Decorator API** for automatic caching
  - `@cache.cached()` decorator with semantic + strict parameter matching
  - Auto-strict mode for non-string parameters
  - Static context support for fixed parameters
  - Preserved function metadata (`__name__`, `__doc__`)

#### Storage & Performance
- **LanceDB vector storage**
  - Persistent and in-memory modes
  - Singleton pattern per `db_uri` for cache pooling
  - IVF-PQ vector indexing for >1K entries
  - Auto-indexing with configurable thresholds
- **Type-safe serialization**
  - Native support for `str`, `int`, `float`, `bool`, `None`, `dict`, `list`
  - DataFrame support: `pandas.DataFrame`, `polars.DataFrame`
  - NumPy arrays with dtype preservation
  - Apache Arrow IPC for large payloads (>10MB)
  - Handles arbitrarily nested structures

#### Observability & Monitoring
- **OpenTelemetry metrics integration**
  - OTLP HTTP protocol support
  - Export to Grafana, Prometheus, Jaeger, SigNoz, any OTLP backend
  - Metrics: cache hits/misses, hit rate, lookup/store errors
  - Configurable export intervals and authentication headers
  - Automatic delta calculation for counters
  - Global singleton pattern per process
- **Structured logging**
  - JSON and text output formats
  - Configurable log levels (DEBUG, INFO, WARNING, ERROR)
  - structlog integration for production
  - ELK/Datadog/Grafana-ready
- **Health checks**
  - Component-level diagnostics (embedding, database, schedulers)
  - Error rate monitoring with thresholds
  - Kubernetes-ready liveness/readiness probes
  - Detailed metrics in responses
- **Metrics tracking**
  - Hit rate, total requests, cache entries
  - Latency percentiles (p50, p95, p99)
  - Error counts by operation type
  - Eviction and storage statistics
  - Scheduler execution metrics

#### Background Tasks
- **Unified scheduler manager**
  - Support for multiple concurrent schedulers
  - Background cleanup scheduler for TTL enforcement
  - Background metrics export scheduler for OpenTelemetry
  - Configurable intervals and initial delays
  - Graceful shutdown with timeout support
  - Context manager support for automatic cleanup
  - Statistics tracking (runs, deletions, errors per scheduler)

#### Configuration
- **Environment-based configuration** (12-factor app)
  - All settings configurable via environment variables
  - Docker/Kubernetes-friendly
  - Prefix: `REMINISCENCE_*`
- **Configuration options:**
  - Model and backend selection
  - Similarity thresholds
  - Storage paths (memory:// or file://)
  - Eviction policy and max entries
  - TTL and cleanup intervals
  - Logging format and level
  - OpenTelemetry endpoints and credentials
  - Vector indexing parameters

### Documentation
- **Comprehensive README**
  - Quick-start guide with decorator and manual usage
  - Configuration examples (development, production, Kubernetes)
  - Use cases and architecture diagrams
  - Performance benchmarks
  - Health check integration examples
  - OpenTelemetry setup for popular backends
- **API documentation**
  - All public methods documented with examples
  - Type hints for all functions
  - Detailed docstrings

### Testing
- **194 tests** with comprehensive coverage:
  - Core caching logic (lookup, store, invalidation)
  - Eviction policies (FIFO, LRU, LFU)
  - Serialization (JSON, Arrow, DataFrames, numpy)
  - Decorator functionality
  - TTL and cleanup
  - OpenTelemetry integration
  - Scheduler lifecycle and error handling
  - Storage singleton behavior
  - Concurrency and thread safety
  - End-to-end workflows

### Performance
- **Lookup latency:** 5-15ms with index, 10-50ms without
- **Store latency:** 5-10ms
- **Embedding generation:** 20-50ms (cached after first use)
- **Scales to 100K+ entries** with automatic indexing

### Dependencies
- **Core:** `lancedb`, `fastembed`, `orjson`, `pyarrow`, `structlog`
- **Optional:** `pandas`, `polars`, `numpy` (for DataFrame/array support)
- **Python:** 3.9+

---

[0.2.0]: https://github.com/yourusername/reminiscence/releases/tag/v0.2.0
[Unreleased]: https://github.com/yourusername/reminiscence/compare/v0.2.0...HEAD
