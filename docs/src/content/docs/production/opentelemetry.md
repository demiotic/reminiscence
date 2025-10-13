---
title: OpenTelemetry
description: Configure metrics export to OpenTelemetry collectors
---

Reminiscence supports OpenTelemetry for production monitoring. This guide covers setup, configuration, and integration with popular observability platforms.

## Quick Start

```python
from reminiscence import Reminiscence, ReminiscenceConfig

config = ReminiscenceConfig(
    enable_metrics=True,
    otel_enabled=True,
    otel_endpoint="http://localhost:4318/v1/metrics",
    otel_service_name="my-service"
)

cache = Reminiscence(config=config)

# Start metrics export (every 60s by default)
cache.start_scheduler(
    metrics_export_interval_seconds=60
)
```

## Configuration

### Enable OpenTelemetry

```python
config = ReminiscenceConfig(
    enable_metrics=True,       # Required for OTEL
    otel_enabled=True,
    otel_endpoint="http://otel-collector:4318/v1/metrics",
    otel_service_name="reminiscence-cache",
    otel_export_interval_ms=30000  # Export every 30s
)

# Via environment
# REMINISCENCE_ENABLE_METRICS=true
# REMINISCENCE_OTEL_ENABLED=true
# REMINISCENCE_OTEL_ENDPOINT=http://otel-collector:4318/v1/metrics
# REMINISCENCE_OTEL_SERVICE_NAME=reminiscence-cache
# REMINISCENCE_OTEL_EXPORT_INTERVAL_MS=30000
```

### OTLP Endpoint

Point to your OpenTelemetry collector:

```python
# Local collector
config = ReminiscenceConfig(
    otel_endpoint="http://localhost:4318/v1/metrics"
)

# Kubernetes service
config = ReminiscenceConfig(
    otel_endpoint="http://otel-collector.observability:4318/v1/metrics"
)

# Cloud endpoint (Datadog, New Relic, etc.)
config = ReminiscenceConfig(
    otel_endpoint="https://api.datadoghq.com/api/v2/metrics"
)
```

### Custom Headers

Add authentication or routing headers:

```python
config = ReminiscenceConfig(
    otel_endpoint="https://api.datadoghq.com/api/v2/metrics",
    otel_headers="DD-API-KEY=your-api-key,x-tenant=acme"
)

# Via environment
# REMINISCENCE_OTEL_HEADERS=DD-API-KEY=key,x-tenant=acme
```

## Exported Metrics

Reminiscence exports the following metrics:

### Cache Metrics

```python
# Counter metrics
reminiscence.cache.hits               # Total cache hits
reminiscence.cache.misses             # Total cache misses
reminiscence.cache.requests           # Total requests (hits + misses)

# Gauge metrics
reminiscence.cache.hit_rate           # Hit rate (0.0-1.0)
reminiscence.cache.entries            # Current cache size
reminiscence.cache.latency_saved_ms   # Total latency saved

# Histogram metrics
reminiscence.lookup.latency           # Lookup latency distribution
reminiscence.result.size              # Cached result size distribution
```

### Eviction Metrics

```python
# Counter metrics
reminiscence.eviction.total           # Total evictions
reminiscence.eviction.by_policy       # Evictions by policy (fifo/lru/lfu)

# Histogram metrics
reminiscence.eviction.entry_age       # Age of evicted entries
reminiscence.eviction.lfu.frequency   # LFU: Frequency of evicted entries
reminiscence.eviction.lru.recency     # LRU: Recency of evicted entries
```

### Storage Metrics

```python
# Counter metrics
reminiscence.storage.searches         # Total search operations
reminiscence.storage.adds             # Total add operations
reminiscence.storage.errors           # Storage operation errors

# Histogram metrics
reminiscence.storage.search_latency   # Search latency distribution
reminiscence.storage.add_latency      # Add latency distribution
```

### Embedding Metrics

```python
# Counter metrics
reminiscence.embedding.generations    # Total embeddings generated
reminiscence.embedding.errors         # Embedding generation errors

# Histogram metrics
reminiscence.embedding.latency        # Embedding generation latency
```

### Scheduler Metrics

```python
# Counter metrics
reminiscence.scheduler.runs           # Total scheduler runs
reminiscence.scheduler.errors         # Scheduler errors

# Histogram metrics
reminiscence.scheduler.cleanup_latency  # Cleanup operation latency
```

## Prometheus Integration

### OpenTelemetry Collector Config

```yaml
# otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      http:
        endpoint: 0.0.0.0:4318

exporters:
  prometheus:
    endpoint: "0.0.0.0:8889"
    namespace: reminiscence

service:
  pipelines:
    metrics:
      receivers: [otlp]
      exporters: [prometheus]
```

### Prometheus Queries

```promql
# Hit rate over time
reminiscence_cache_hit_rate

# Request rate
rate(reminiscence_cache_requests[5m])

# P95 lookup latency
histogram_quantile(0.95,
  rate(reminiscence_lookup_latency_bucket[5m])
)

# Eviction rate
rate(reminiscence_eviction_total[5m])

# Cache size
reminiscence_cache_entries

# Error rate
rate(reminiscence_storage_errors[5m])
```

### Grafana Dashboard

Example Grafana dashboard queries:

```json
{
  "dashboard": {
    "title": "Reminiscence Cache",
    "panels": [
      {
        "title": "Hit Rate",
        "targets": [{
          "expr": "reminiscence_cache_hit_rate"
        }]
      },
      {
        "title": "Request Rate",
        "targets": [{
          "expr": "rate(reminiscence_cache_requests[5m])"
        }]
      },
      {
        "title": "Lookup Latency (P50, P95, P99)",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(reminiscence_lookup_latency_bucket[5m]))",
            "legendFormat": "P50"
          },
          {
            "expr": "histogram_quantile(0.95, rate(reminiscence_lookup_latency_bucket[5m]))",
            "legendFormat": "P95"
          },
          {
            "expr": "histogram_quantile(0.99, rate(reminiscence_lookup_latency_bucket[5m]))",
            "legendFormat": "P99"
          }
        ]
      }
    ]
  }
}
```

## Cloud Platform Integration

### Datadog

```python
from reminiscence import Reminiscence, ReminiscenceConfig

config = ReminiscenceConfig(
    otel_enabled=True,
    otel_endpoint="https://api.datadoghq.com/api/v2/metrics",
    otel_headers=f"DD-API-KEY={os.getenv('DD_API_KEY')}",
    otel_service_name="reminiscence"
)

cache = Reminiscence(config=config)
cache.start_scheduler(metrics_export_interval_seconds=30)
```

### New Relic

```python
config = ReminiscenceConfig(
    otel_enabled=True,
    otel_endpoint="https://otlp.nr-data.net:4318/v1/metrics",
    otel_headers=f"api-key={os.getenv('NEW_RELIC_LICENSE_KEY')}",
    otel_service_name="reminiscence"
)
```

### Honeycomb

```python
config = ReminiscenceConfig(
    otel_enabled=True,
    otel_endpoint="https://api.honeycomb.io/v1/metrics",
    otel_headers=f"x-honeycomb-team={os.getenv('HONEYCOMB_API_KEY')}",
    otel_service_name="reminiscence"
)
```

### Grafana Cloud

```python
config = ReminiscenceConfig(
    otel_enabled=True,
    otel_endpoint="https://otlp-gateway-prod-us-central-0.grafana.net/otlp/v1/metrics",
    otel_headers=f"Authorization=Basic {base64.b64encode(f'{instance_id}:{api_key}'.encode()).decode()}",
    otel_service_name="reminiscence"
)
```

## Kubernetes Deployment

### Deployment with OpenTelemetry Sidecar

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: reminiscence-cache
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: app
        image: my-app:latest
        env:
        - name: REMINISCENCE_OTEL_ENABLED
          value: "true"
        - name: REMINISCENCE_OTEL_ENDPOINT
          value: "http://localhost:4318/v1/metrics"
        - name: REMINISCENCE_OTEL_SERVICE_NAME
          value: "reminiscence-cache"

      - name: otel-collector
        image: otel/opentelemetry-collector:latest
        ports:
        - containerPort: 4318
        volumeMounts:
        - name: otel-config
          mountPath: /etc/otel
        command: ["--config=/etc/otel/config.yaml"]

      volumes:
      - name: otel-config
        configMap:
          name: otel-collector-config
```

### ConfigMap for OTEL Collector

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: otel-collector-config
data:
  config.yaml: |
    receivers:
      otlp:
        protocols:
          http:
            endpoint: 0.0.0.0:4318

    exporters:
      prometheus:
        endpoint: "0.0.0.0:8889"

      logging:
        loglevel: info

    service:
      pipelines:
        metrics:
          receivers: [otlp]
          exporters: [prometheus, logging]
```

## Docker Compose Setup

```yaml
version: '3.8'

services:
  app:
    image: my-app:latest
    environment:
      - REMINISCENCE_OTEL_ENABLED=true
      - REMINISCENCE_OTEL_ENDPOINT=http://otel-collector:4318/v1/metrics
      - REMINISCENCE_OTEL_SERVICE_NAME=reminiscence
    depends_on:
      - otel-collector

  otel-collector:
    image: otel/opentelemetry-collector:latest
    command: ["--config=/etc/otel/config.yaml"]
    volumes:
      - ./otel-config.yaml:/etc/otel/config.yaml
    ports:
      - "4318:4318"  # OTLP HTTP
      - "8889:8889"  # Prometheus scrape endpoint

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
    depends_on:
      - otel-collector

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
```

## Alerting

### Prometheus Alerting Rules

```yaml
# alert-rules.yml
groups:
  - name: reminiscence
    interval: 30s
    rules:
      - alert: LowHitRate
        expr: reminiscence_cache_hit_rate < 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Cache hit rate below 50%"

      - alert: HighErrorRate
        expr: rate(reminiscence_storage_errors[5m]) > 0.01
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Storage error rate above 1%"

      - alert: HighLookupLatency
        expr: histogram_quantile(0.95, rate(reminiscence_lookup_latency_bucket[5m])) > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P95 lookup latency above 100ms"

      - alert: SchedulerErrors
        expr: rate(reminiscence_scheduler_errors[5m]) > 0
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Scheduler experiencing errors"
```

## Monitoring Best Practices

### 1. Track Hit Rate

```promql
# Target: > 70% hit rate
reminiscence_cache_hit_rate > 0.7
```

**Action if low:**
- Increase cache size (`max_entries`)
- Tune similarity threshold
- Check TTL settings

### 2. Monitor Latency

```promql
# Target: P95 < 50ms
histogram_quantile(0.95, rate(reminiscence_lookup_latency_bucket[5m])) < 0.050
```

**Action if high:**
- Enable vector indexing (`auto_create_index=True`)
- Optimize query complexity
- Check storage backend performance

### 3. Watch Error Rate

```promql
# Target: < 0.1% error rate
rate(reminiscence_storage_errors[5m]) < 0.001
```

**Action if high:**
- Check storage backend health
- Review error logs
- Verify database connectivity

### 4. Track Eviction Rate

```promql
# Target: Minimal evictions
rate(reminiscence_eviction_total[5m])
```

**Action if high:**
- Increase `max_entries`
- Reduce TTL if appropriate
- Consider LRU/LFU policy

### 5. Monitor Cache Size

```promql
# Ensure not hitting max
reminiscence_cache_entries / reminiscence_max_entries < 0.9
```

**Action if near max:**
- Increase `max_entries`
- Enable more aggressive TTL
- Review eviction policy

## Troubleshooting

### No Metrics Exported

```python
# Check OTEL configuration
health = cache.health_check()
otel_check = health["checks"]["opentelemetry"]

if not otel_check["ok"]:
    print(f"OTEL issue: {otel_check['details']}")

# Verify metrics enabled
assert cache.config.enable_metrics == True
assert cache.config.otel_enabled == True

# Check exporter initialization
assert cache.otel_exporter is not None
```

### Connection Errors

```python
# Test endpoint connectivity
import requests

try:
    response = requests.post(
        cache.config.otel_endpoint,
        json={},
        timeout=5
    )
    print(f"Endpoint reachable: {response.status_code}")
except Exception as e:
    print(f"Connection error: {e}")
```

### High Export Latency

```python
# Reduce export frequency
cache.start_scheduler(
    metrics_export_interval_seconds=120  # 2 minutes
)

# Or export manually at controlled intervals
if time.time() - last_export > 60:
    metrics = cache.get_stats()
    cache.otel_exporter.export(metrics)
    last_export = time.time()
```

## Next Steps

- [Health Checks](/production/health-checks/) - Monitoring cache health
- [Performance](/production/performance/) - Optimization techniques
- [Best Practices](/production/best-practices/) - Production deployment guide
