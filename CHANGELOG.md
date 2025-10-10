# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2025-10-00

### Added

- **Context-specific similarity thresholds**
  - `ReminiscenceConfig.context_thresholds` Dict[str, float] for different thresholds per context
  - `get_threshold_for_context()` helper method to resolve threshold dynamically
  - Pattern matching supports `key:value` format (e.g., `"agent:sql": 0.99`, `"model:gpt-4": 0.95`)
  - Automatically applied in `lookup()` and `lookup_batch()` when no explicit threshold provided
  - Environment variable: `REMINISCENCE_CONTEXT_THRESHOLDS` (JSON format)
  - Enables multi-agent caching with optimal precision per agent/model/task

- **Per-entry TTL (Time-To-Live)**
  - `CacheEntry.ttl_seconds` optional field for per-entry TTL override
  - `CacheEntry.is_expired` property to check expiration status
  - `CacheEntry.ttl_remaining` property showing remaining TTL in seconds
  - `store(ttl_seconds=...)` parameter to set custom TTL for individual entries
  - `store_batch(ttl_seconds=[...])` with different TTL per entry in batch operations
  - `LookupResult.ttl_remaining` field shows remaining TTL of matched entry
  - Falls back to global `config.ttl_seconds` when not specified

- **Bulk invalidation with pattern matching**
  - `BulkInvalidatePattern` dataclass for complex invalidation specifications
  - `invalidate_bulk(pattern)` for efficient pattern-based invalidation (single-scan approach)
  - Convenience methods:
    - `invalidate_by_prefix(prefix)` - invalidate by query prefix
    - `invalidate_by_regex(regex)` - invalidate by regex pattern
    - `invalidate_by_context(matches)` - invalidate by context pattern with wildcards
    - `invalidate_older_than(seconds)` - invalidate by age threshold
  - Pattern matching supports:
    - `query_regex`: Match queries by regex
    - `query_prefix`: Match queries starting with prefix
    - `query_suffix`: Match queries ending with suffix
    - `context_matches`: Match context with wildcard support (`agent_*`, `model:*`)
    - `older_than_seconds`: Match entries older than threshold
    - `similarity_below`: Match entries below similarity score
    - `entry_ids`: Match specific entry IDs list

- **Batch operations support**
  - `lookup_batch()` and `store_batch()` with batch embedding generation (3-5x faster)
  - Supports per-query contexts (list of dicts) or shared context (single dict)
  - Auto mode detection in batch operations (applies heuristic per query)
  - Context-specific thresholds automatically applied per query in batches
  - Per-entry TTL support in `store_batch(ttl_seconds=[...])`
  - Per-entry context thresholds in `store_batch(context_thresholds=[...])`
  - Stores `query_mode` in entry metadata for flexible routing
  - Performance: ~5% overhead for single items, 3-10x gains for batches

- **Result compression support**
  - Optional compression for cached results to reduce storage size
  - Enable with `REMINISCENCE_COMPRESSION_ENABLED=true`
  - Supported algorithms:
    - `zstd` (default): Compression levels 1-22, recommended level 3
    - `gzip`: Compression levels 1-9
    - `none`: Disable compression
  - Configure via `REMINISCENCE_COMPRESSION_ALGORITHM` and `REMINISCENCE_COMPRESSION_LEVEL`
  - Transparent compression/decompression at storage layer
  - Compatible with all result types (JSON, Arrow, DataFrames, NumPy arrays)
  - Works alongside encryption when both enabled

- **Encryption support for cached results**
  - Optional encryption for stored cache results using **age encryption**
  - Enable with `REMINISCENCE_ENCRYPTION_ENABLED=true`
  - Provide encryption key via `REMINISCENCE_ENCRYPTION_KEY` environment variable
  - Age encryption backend (`REMINISCENCE_ENCRYPTION_BACKEND=age`)
  - Configurable worker pool for parallel encryption (`REMINISCENCE_ENCRYPTION_MAX_WORKERS`, default: 4)
  - Transparent encryption/decryption at storage layer
  - Batch serialization with optional encryption for performance
  - Compatible with all result types and compression

- **Decorator batch support**
  - `batch_mode` parameter in decorator (default: True)
  - Auto-detects single vs batch calls transparently
  - Supports single context (dict) or per-item context (list)
  - Handles batch processing automatically in decorators

### Changed

- **Storage layer refactoring**
  - Split monolithic `lancedb.py` (1000+ lines) into modular architecture under `reminiscence/storage/lancedb/`:
    - `backend.py` - Main LanceDBBackend class with singleton pattern and dual-table management
    - `schema.py` - Arrow schema definitions for exact and semantic tables
    - `serialization.py` - Result serialization with pickle, compression, and encryption
    - `query_builder.py` - SQL query builder for context filtering, TTL checks, and cleanup
    - `table_manager.py` - Table lifecycle management (create, search, add, delete)
  - Each module now ~200 lines (down from 1000+), following Single Responsibility Principle
  - **No breaking changes**: Public API remains identical

- **Core API enhancement**
  - Added optional `embedder` parameter to `Reminiscence.__init__()` for dependency injection
  - Allows sharing embedder instances across multiple cache instances
  - Example: `cache = Reminiscence(config, embedder=shared_embedder)`

- **Types updated**
  - `CacheEntry` now includes optional `ttl_seconds` and `context_threshold` fields (backward compatible)
  - `LookupResult` now includes `ttl_remaining` field
  - `StoreRequest` updated with `ttl_seconds` and `context_threshold` fields
  - New `BulkInvalidatePattern` dataclass exported from types module

- **Configuration updated**
  - `ReminiscenceConfig.context_thresholds` added with empty dict default (backward compatible)
  - `ReminiscenceConfig.compression_enabled` added (default: False)
  - `ReminiscenceConfig.compression_algorithm` added (default: "zstd")
  - `ReminiscenceConfig.compression_level` added (default: 3)
  - Config validation ensures threshold values are in [0.0, 1.0] range
  - Config validation ensures compression levels are valid for chosen algorithm
  - `get_threshold_for_context()` method added for context-aware threshold resolution

- **Cache operations enhanced**
  - `lookup()` now uses context-specific thresholds when available
  - `lookup_batch()` applies per-query context thresholds automatically
  - `store()` signature extended with optional `ttl_seconds` and `context_threshold` parameters
  - `store_batch()` signature extended with list-based `ttl_seconds` and `context_thresholds`
  - `_process_hit()` checks per-entry TTL before falling back to global config
  - `_lookup_with_embedding()` updated to support context-specific thresholds
  - `check_availability()` now respects per-entry TTL

- **Logging improvements**
  - Added `context_specific` source indicator in threshold resolution logs
  - Added `ttl_seconds` logging in store operations
  - Added `ttl_remaining` logging in cache hit events
  - Better debugging for bulk invalidation operations

- **Storage API changes** (BREAKING)
  - `storage.add()` now reads `query_mode` from entry metadata instead of parameter
  - `storage.search()` no longer accepts "auto" mode (resolved upstream in cache layer)
  - Storage layer responsibility: physical storage, optional encryption/compression
  - Cache layer responsibility: mode detection and routing

### Fixed

- **Security improvements (Bandit compliance)**
  - Replaced hardcoded `/tmp` directory with `tempfile.gettempdir()` in FastEmbed cache path
  - Added debug logging for all try-except-pass blocks to improve error visibility:
    - Eviction cleanup failures during invalidation, TTL cleanup, and age-based cleanup
    - Result size measurement failures in storage operations
  - All silent exceptions now logged at DEBUG level for better observability

- **Type corrections**
  - `AvailabilityCheck` parameter names (`ttl_remaining_seconds` not `ttl_remaining`)
  - `store_batch()` passes full entries list to storage (not single entry)

- **Bulk invalidation fixes**
  - Fixed duplicate timestamp line in `invalidate_bulk()`
  - Corrected storage API usage (uses `delete_by_filter()` instead of non-existent `delete_entry()`)
  - Proper context JSON parsing in bulk operations

- **FastEmbed rate limiting**
  - Uses `local_files_only=True` first to prevent HuggingFace API rate limits
  - Automatic fallback to download if model not in cache
  - Better cache directory handling and logging

### Performance

- Bulk invalidation uses single table scan vs N individual delete operations
- Context-specific thresholds eliminate need for separate cache instances per agent
- FastEmbed cache checks prevent unnecessary network requests
- Batch operations leverage SIMD optimizations in ONNX runtime
- Compression reduces storage footprint by 60-80% (zstd level 3)

### Migration Notes

All changes are **100% backward compatible**. Existing code works unchanged:

- If `ttl_seconds` not specified, uses global `config.ttl_seconds`
- If `context_thresholds` not configured, uses `config.similarity_threshold`
- If `context_threshold` not set per-entry, uses config-level threshold
- Compression and encryption are disabled by default
- New fields in `CacheEntry` are optional with sensible defaults
- Batch methods work identically to before, new parameters are optional


## [0.4.0] - 2025-10-09

### Added
- **Auto query mode** with intelligent detection
  - Automatically detects query type: SQL/code/URLs use exact mode, natural language uses semantic
  - Reduces embedding costs by 30-50% for mixed workloads
  - Heuristics in `reminiscence/utils/query_detection.py`
  - Supports SQL, GraphQL, API endpoints, file paths, JSON, code snippets
  - Can be overridden with explicit `query_mode="semantic"` or `query_mode="exact"`
- **Batch embeddings API**
  - `embed_batch()` method for 3-5x performance improvement over sequential
  - `store_batch()` optimized to use batch embeddings
  - Configurable batch size via `REMINISCENCE_EMBEDDING_BATCH_SIZE` (default: 32)
- **Dual table architecture**
  - Separate `exact_cache` table for deterministic queries (SQL, code, APIs)
  - Separate `semantic_cache` table for fuzzy matching (natural language)
  - Auto mode intelligently routes to appropriate table
  - Both tables share same context matching logic

### Changed
- **Test infrastructure improvements**
  - Module-scoped fixtures for 75% faster test execution (~4min vs 15-20min)
  - OTEL collector Docker integration for real integration tests
  - Automatic container lifecycle management with `pytest_configure` and `pytest_sessionfinish`
  - 238 tests passing with 0 errors

## [0.3.0] - 2025-10-09

### Added
- **Query modes** for flexible matching strategies
  - `semantic`: Normal semantic search with configurable threshold (default)
  - `exact`: Near-exact matching with threshold 0.9999 for SQL/API caching
  - `auto`: Intelligent mode that tries exact match first, falls back to semantic
  - Added `query_mode` parameter to `lookup()`, `store()`, and `cached()` decorator
- **OpenTelemetry metrics integration**
  - OTLP HTTP protocol support for metrics export
  - Export to Grafana, Prometheus, Jaeger, SigNoz, any OTLP-compatible backend
  - Metrics: cache hits/misses, hit rate, lookup/store errors, latency percentiles
  - Configurable export intervals and authentication headers
  - Automatic delta calculation for counter metrics
  - Global singleton pattern per process to prevent duplicate exports
  - Background scheduler for automatic periodic export
  
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

[Unreleased]: https://github.com/yourusername/reminiscence/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/yourusername/reminiscence/releases/tag/v0.4.0
[0.3.0]: https://github.com/yourusername/reminiscence/releases/tag/v0.3.0
[0.2.0]: https://github.com/yourusername/reminiscence/releases/tag/v0.2.0
