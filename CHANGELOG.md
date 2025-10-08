# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2025-10-09

### Added
- **Query modes** for flexible matching strategies
  - `semantic`: Normal semantic search with configurable threshold (default)
  - `exact`: Near-exact matching with threshold 0.9999 for SQL/API caching
  - `auto`: Intelligent mode that tries exact match first, falls back to semantic
  - Added `query_mode` parameter to `lookup()`, `store()`, and `cached()` decorator

### Changed
- **Always-on embeddings architecture** - Simplified caching approach
  - Embeddings now generated for all query modes (semantic, exact, auto)
  - Exact mode now uses threshold 0.9999 instead of nullable embeddings
- **API improvements**
  - Renamed decorator parameters: `query_param` → `query`, `strict_params` → `context_params`
  - Added `similarity_threshold` parameter to decorator
- **Test performance** - 10x faster test suite
  - Module-scoped fixtures reuse embedding model across tests
  - Test runtime reduced from ~120s to ~15s
  - Added fixture-based approach for integration tests

### Fixed
- Decorator parameter validation and type checking
- Test isolation issues with shared fixtures

## [0.2.0] - 2025-10-08

### Added

#### Core Features
- **Semantic caching** with FastEmbed multilingual embeddings
  - Hybrid matching: semantic similarity + exact context matching
  - Configurable similarity thresholds (0.75-0.95)
  - Support for 384-dimensional embeddings (paraphrase-multilingual-MiniLM-L12-v2)
- **Multiple eviction policies**
  - FIFO (First In First Out)
  - LRU (Least Recently Used)
  - LFU (Least Frequently Used)
  - Configurable max entries with automatic eviction
- **TTL-based expiration**
  - Time-to-live support with configurable seconds
  - Automatic cleanup of expired entries
  - Manual invalidation by context or age
- **Decorator API** for automatic caching
  - `@cache.cached()` decorator with semantic + context parameter matching
  - Auto-strict mode for non-string parameters
  - Static context support

#### Storage & Performance
- **LanceDB vector storage**
  - Persistent and in-memory modes
  - Singleton pattern per `db_uri`
  - IVF-PQ vector indexing for >1K entries
  - Auto-indexing with configurable thresholds
- **Type-safe serialization**
  - Native support for primitives, dicts, lists
  - DataFrame support: pandas, polars
  - NumPy arrays with dtype preservation
  - Apache Arrow IPC for large payloads

#### Observability & Monitoring
- **OpenTelemetry metrics integration**
  - OTLP HTTP protocol support
  - Export to Grafana, Prometheus, Jaeger, SigNoz
  - Metrics: hits/misses, hit rate, errors, latency
  - Configurable export intervals
- **Structured logging** with structlog
  - JSON and text output formats
  - Configurable log levels
  - Production-ready logging
- **Health checks**
  - Component-level diagnostics
  - Error rate monitoring
  - Kubernetes-ready probes

#### Background Tasks
- **Unified scheduler manager**
  - Multiple concurrent schedulers
  - Background cleanup for TTL enforcement
  - Background metrics export
  - Graceful shutdown support

#### Configuration
- **Environment-based configuration** (12-factor app)
  - All settings via environment variables
  - Docker/Kubernetes-friendly
  - Prefix: `REMINISCENCE_*`

### Testing
- **194 tests** with comprehensive coverage
- Core caching, eviction policies, serialization
- Decorator functionality, TTL, OpenTelemetry
- End-to-end workflows

### Performance
- Lookup latency: 5-15ms with index, 10-50ms without
- Store latency: 5-10ms
- Scales to 100K+ entries with automatic indexing

---

[Unreleased]: https://github.com/yourusername/reminiscence/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/yourusername/reminiscence/releases/tag/v0.3.0
[0.2.0]: https://github.com/yourusername/reminiscence/releases/tag/v0.2.0
