---
title: Best Practices
description: Production deployment and operational best practices for Reminiscence
---

This guide covers best practices for deploying and operating Reminiscence in production.

## Configuration

### Use Environment Variables

```python
# ✓ Good: Load from environment
cache = Reminiscence()  # Loads from REMINISCENCE_* env vars

# ❌ Avoid: Hardcoded config
config = ReminiscenceConfig(
    db_uri="./cache.db",  # Hardcoded path
    otel_endpoint="http://localhost:4318"  # Hardcoded endpoint
)
```

### Separate Dev and Prod Configs

```python
# config.py
import os
from reminiscence import Reminiscence, ReminiscenceConfig

def get_config():
    env = os.getenv("ENV", "development")

    if env == "production":
        return ReminiscenceConfig(
            db_uri=os.getenv("CACHE_DB_URI", "./cache.db"),
            max_entries=100000,
            eviction_policy="lru",
            ttl_seconds=86400,
            auto_create_index=True,
            enable_metrics=True,
            otel_enabled=True,
            otel_endpoint=os.getenv("OTEL_ENDPOINT"),
            compression_enabled=True,
            log_level="INFO",
            json_logs=True
        )
    else:
        return ReminiscenceConfig(
            db_uri="memory://",
            log_level="DEBUG",
            enable_metrics=True
        )

cache = Reminiscence(config=get_config())
```

## Storage

### Use Persistent Storage in Production

```python
# ✓ Production: Persistent storage
config = ReminiscenceConfig(db_uri="/var/lib/reminiscence/cache.db")

# ✓ Development: In-memory
config = ReminiscenceConfig(db_uri="memory://")

# ❌ Avoid: Relative paths in production
config = ReminiscenceConfig(db_uri="./cache.db")  # CWD-dependent
```

### Set Appropriate Cache Size

```python
# Small service (<100 req/s)
config = ReminiscenceConfig(max_entries=10000)

# Medium service (100-1000 req/s)
config = ReminiscenceConfig(max_entries=100000)

# Large service (>1000 req/s)
config = ReminiscenceConfig(max_entries=1000000)

# Monitor and adjust based on:
# - Hit rate (target: > 70%)
# - Memory usage
# - Query patterns
```

### Enable Vector Indexing

```python
config = ReminiscenceConfig(
    auto_create_index=True,
    index_threshold_entries=256
)

# For large caches, create index manually on startup
if cache.backend.count() >= 256:
    cache.create_index()
```

## TTL and Eviction

### Set Appropriate TTL

```python
# Short-lived data (stock prices, weather)
config = ReminiscenceConfig(ttl_seconds=300)  # 5 minutes

# Medium-lived data (API responses, LLM calls)
config = ReminiscenceConfig(ttl_seconds=3600)  # 1 hour

# Long-lived data (translations, definitions)
config = ReminiscenceConfig(ttl_seconds=86400)  # 24 hours

# Eternal data (mathematical facts, constants)
config = ReminiscenceConfig(ttl_seconds=None)  # No expiration
```

### Choose the Right Eviction Policy

```python
# LRU: Best for most use cases
config = ReminiscenceConfig(eviction_policy="lru")
# Keeps frequently accessed entries

# LFU: For skewed access patterns
config = ReminiscenceConfig(eviction_policy="lfu")
# Keeps most frequently used entries

# FIFO: For highest throughput
config = ReminiscenceConfig(eviction_policy="fifo")
# Simplest, fastest eviction
```

## Decorator Usage (Recommended)

**Decorators provide the best developer experience** - use them by default:

```python
from reminiscence import Reminiscence

cache = Reminiscence()

# ✓ Excellent: Decorator (automatic, clean)
@cache.cached(query="prompt", context=["model"])
def call_llm(prompt: str, model: str):
    return expensive_llm_call(prompt, model)

# Use it naturally
answer = call_llm("What is AI?", model="gpt-4")

# ❌ Avoid: Manual lookup/store (verbose, error-prone)
def call_llm_manual(prompt: str, model: str):
    result = cache.lookup(prompt, {"model": model})
    if result.is_hit:
        return result.result
    answer = expensive_llm_call(prompt, model)
    cache.store(prompt, {"model": model}, answer)
    return answer
```

### Decorator Best Practices

```python
# ✓ Good: Clear query parameter
@cache.cached(query="question", context=["user_id"])
def answer_question(question: str, user_id: int):
    return generate_answer(question, user_id)

# ✓ Good: Static context for versioning
@cache.cached(
    query="request",
    context=["user_id"],
    static_context={"version": "v2"}
)
def api_handler(request: str, user_id: int):
    return process(request, user_id)

# ✓ Good: EXACT mode for deterministic queries
@cache.cached(
    query="sql",
    context=["database"],
    mode=QueryMode.EXACT
)
def execute_query(sql: str, database: str):
    return db.execute(sql, database)

# ❌ Avoid: Too many context parameters
@cache.cached(
    query="q",
    context=["user", "session", "request_id", "timestamp"]
)
def process(q: str, **kwargs):
    return do_work(q)  # Low hit rate!
```

## Monitoring and Observability

### Enable OpenTelemetry

```python
import os
from reminiscence import Reminiscence, ReminiscenceConfig

config = ReminiscenceConfig(
    enable_metrics=True,
    otel_enabled=True,
    otel_endpoint=os.getenv("OTEL_ENDPOINT"),
    otel_service_name=os.getenv("SERVICE_NAME", "cache"),
    otel_export_interval_ms=30000  # 30 seconds
)

cache = Reminiscence(config=config)
cache.start_scheduler(metrics_export_interval_seconds=30)
```

### Set Up Health Checks

```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/health/ready")
def readiness():
    health = cache.health_check()
    if health["status"] == "healthy":
        return jsonify({"status": "ready"}), 200
    return jsonify({"status": "not ready"}), 503

@app.route("/health/live")
def liveness():
    return jsonify({"status": "ok"}), 200
```

### Monitor Key Metrics

Track these critical metrics:

```python
# Hit rate (target: > 70%)
stats = cache.get_stats()
if stats["hit_rate"] < 0.7:
    logger.warning(f"Low hit rate: {stats['hit_rate']}")

# Lookup latency (target: P95 < 50ms)
if stats["lookup_latency_ms"]["p95"] > 50:
    logger.warning(f"High latency: {stats['lookup_latency_ms']['p95']}ms")

# Error rate (target: < 1%)
error_rate = (stats["errors"]["lookup"] + stats["errors"]["store"]) / stats["total_requests"]
if error_rate > 0.01:
    logger.error(f"High error rate: {error_rate * 100:.1f}%")
```

## Error Handling

### Don't Cache Errors by Default

```python
# ✓ Good: Errors not cached by default
@cache.cached(query="request")
def api_call(request: str):
    response = external_api(request)
    response.raise_for_status()  # Raises on error
    return response.json()

# ✓ Explicitly cache errors if needed
@cache.cached(query="request", allow_errors=True)
def api_call_with_error_cache(request: str):
    try:
        return external_api(request).json()
    except APIError as e:
        return {"error": str(e)}  # Will be cached
```

### Implement Circuit Breakers

```python
from datetime import datetime, timedelta

class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure = None
        self.state = "closed"  # closed, open, half-open

    def call(self, func, *args, **kwargs):
        if self.state == "open":
            if datetime.now() - self.last_failure > timedelta(seconds=self.timeout):
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")

        try:
            result = func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure = datetime.now()
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            raise

breaker = CircuitBreaker()

@cache.cached(query="request")
def api_call(request: str):
    return breaker.call(external_api, request)
```

## Security

### Encrypt Sensitive Data

```python
config = ReminiscenceConfig(
    encryption_enabled=True,
    encryption_key=os.getenv("CACHE_ENCRYPTION_KEY"),
    encryption_backend="age"
)

# Never log or expose encryption keys
# Store in secrets management (AWS Secrets Manager, HashiCorp Vault, etc.)
```

### Isolate Multi-Tenant Caches

```python
# ✓ Good: Tenant ID in context
@cache.cached(query="request", context=["tenant_id", "user_id"])
def multi_tenant_api(request: str, tenant_id: str, user_id: int):
    return process_request(request, tenant_id, user_id)

# ❌ Bad: No tenant isolation
@cache.cached(query="request")
def unsafe_api(request: str):
    return process_request(request)  # Cross-tenant data leakage!
```

### Validate Context Values

```python
def sanitize_context(context: dict) -> dict:
    """Remove sensitive data from context"""
    sensitive_keys = {"password", "token", "secret", "api_key"}
    return {
        k: v for k, v in context.items()
        if k.lower() not in sensitive_keys
    }

@cache.cached(query="request", context=["user_id"])
def safe_api(request: str, user_id: int, api_key: str):
    # api_key not in context - won't be cached
    return external_api(request, user_id, api_key)
```

## Graceful Shutdown

### Stop Schedulers Cleanly

```python
import signal
import sys

def signal_handler(sig, frame):
    print("Shutting down gracefully...")
    cache.stop_scheduler(timeout=10.0)

    # Export final metrics
    if cache.otel_exporter and cache.metrics:
        cache.otel_exporter.export(cache.metrics.report())

    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Start schedulers
cache.start_scheduler()

# Or use context manager
with Reminiscence() as cache:
    cache.start_scheduler()
    # ... use cache ...
# Schedulers stopped automatically
```

## Backup and Recovery

### Export Cache Regularly

```python
import schedule
import time

def backup_cache():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    cache.export_to_file(f"/backups/cache-{timestamp}.parquet")
    logger.info(f"Cache backed up to cache-{timestamp}.parquet")

# Schedule daily backups
schedule.every().day.at("02:00").do(backup_cache)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### Restore from Backup

```python
# On service start, restore from latest backup
import glob
import os

def restore_latest_backup():
    backups = glob.glob("/backups/cache-*.parquet")
    if backups:
        latest = max(backups, key=os.path.getctime)
        logger.info(f"Restoring cache from {latest}")
        cache.import_from_file(latest)

# Restore before starting service
restore_latest_backup()
```

## Deployment Patterns

### Rolling Deployment

```yaml
# Kubernetes Deployment with RollingUpdate
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cache-service
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0  # Zero downtime

  template:
    spec:
      containers:
      - name: app
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 15"]  # Drain connections

        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
```

### Blue-Green Deployment

```bash
# Deploy new version
kubectl apply -f deployment-green.yaml

# Wait for ready
kubectl wait --for=condition=ready pod -l version=green

# Switch traffic
kubectl patch service cache-service -p '{"spec":{"selector":{"version":"green"}}}'

# Monitor for issues
# If OK, delete blue deployment
# If issues, rollback to blue
```

## Logging

### Structured Logging

```python
config = ReminiscenceConfig(
    log_level="INFO",
    json_logs=True  # Structured logs for production
)

# Logs are JSON-formatted:
# {"event": "cache_hit", "similarity": 0.92, "latency_ms": 12.5, ...}
```

### Log Sampling

```python
import random

def should_log_sample(rate=0.01):
    return random.random() < rate

@cache.cached(query="request")
def api_handler(request: str):
    result = process(request)

    # Log 1% of requests
    if should_log_sample():
        logger.info(
            "request_processed",
            request_preview=request[:100],
            result_size=len(str(result))
        )

    return result
```

## Testing

### Unit Tests

```python
import pytest
from reminiscence import Reminiscence

@pytest.fixture
def cache():
    """Create fresh cache for each test"""
    return Reminiscence()

def test_cache_hit(cache):
    cache.store("query", {}, "result")
    result = cache.lookup("query", {})
    assert result.is_hit
    assert result.result == "result"

def test_cache_miss(cache):
    result = cache.lookup("unknown", {})
    assert not result.is_hit
```

### Integration Tests

```python
def test_llm_caching_integration():
    cache = Reminiscence()

    @cache.cached(query="prompt", context=["model"])
    def call_llm(prompt: str, model: str):
        return f"Response for: {prompt}"

    # First call
    result1 = call_llm("What is AI?", model="gpt-4")

    # Similar query should hit cache
    result2 = call_llm("Explain AI", model="gpt-4")

    # Verify caching worked
    stats = cache.get_stats()
    assert stats["hits"] > 0
```

## Common Pitfalls

### 1. Caching with High-Cardinality Context

```python
# ❌ Bad: Unique timestamp in context
@cache.cached(query="q", context=["timestamp"])
def process(q: str, timestamp: float):
    return do_work(q)
# Every call has different timestamp = no cache reuse!

# ✓ Good: Use TTL instead
cache.store(query, {}, result, ttl_seconds=60)
```

### 2. Not Using Batch Operations

```python
# ❌ Slow: Loop
for query in queries:
    cache.lookup(query, context)

# ✓ Fast: Batch
requests = [LookupRequest(query=q, context=context) for q in queries]
cache.lookup_batch(requests)
```

### 3. Forgetting Vector Indexing

```python
# ❌ Slow lookups without index
config = ReminiscenceConfig(auto_create_index=False)

# ✓ Fast lookups with index
config = ReminiscenceConfig(auto_create_index=True)
```

### 4. Not Monitoring Hit Rate

```python
# ✓ Monitor and alert on low hit rate
if cache.get_stats()["hit_rate"] < 0.5:
    alert("Low cache hit rate")
```

### 5. Caching Everything

```python
# ❌ Don't cache highly dynamic data
@cache.cached(query="request")
def get_current_timestamp(request: str):
    return time.time()  # Always changing!

# ✓ Cache stable data
@cache.cached(query="user_id")
def get_user_profile(user_id: int):
    return db.query_user(user_id)  # Rarely changes
```

## Capacity Planning

### Estimate Cache Size

```python
# Average result size
avg_result_size = 2000  # bytes

# Target cache entries
target_entries = 100000

# Overhead per entry (metadata, embedding, etc.)
overhead_per_entry = 1500  # bytes

# Total memory needed
total_bytes = target_entries * (avg_result_size + overhead_per_entry)
total_mb = total_bytes / (1024 * 1024)

print(f"Estimated memory: {total_mb:.1f} MB")
```

### Monitor Growth

```python
import time

def monitor_cache_growth():
    while True:
        stats = cache.get_stats()
        utilization = stats["cache_entries"] / stats["max_entries"]

        if utilization > 0.9:
            logger.warning(
                f"Cache near capacity: {utilization * 100:.1f}%"
            )

        time.sleep(60)
```

## Next Steps

- [OpenTelemetry](/production/opentelemetry/) - Metrics and monitoring
- [Health Checks](/production/health-checks/) - Health monitoring
- [Performance](/production/performance/) - Optimization techniques
- [Examples](/examples/llm-apps/) - Real-world examples
