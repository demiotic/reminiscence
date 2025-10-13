---
title: Health Checks
description: Monitoring and verifying Reminiscence cache health in production
---

Health checks help ensure your cache is functioning correctly in production. This guide covers health check implementation and monitoring strategies.

## Quick Start

```python
from reminiscence import Reminiscence

cache = Reminiscence()

# Perform health check
health = cache.health_check()

if health["status"] == "healthy":
    print("✓ Cache is healthy")
else:
    print("✗ Cache has issues")
    for component, check in health["checks"].items():
        if not check["ok"]:
            print(f"  - {component}: {check.get('error', check.get('details'))}")
```

## Health Check Response

The health check returns comprehensive status:

```python
{
    "status": "healthy",  # or "unhealthy"
    "timestamp": 1704067200000,
    "checks": {
        "embedding": {
            "ok": true,
            "error": null
        },
        "database": {
            "ok": true,
            "error": null
        },
        "error_rate": {
            "ok": true,
            "details": "Error rate: 0.5% (5/1000 requests)"
        },
        "schedulers": {
            "ok": true,
            "details": "2/2 schedulers running"
        },
        "opentelemetry": {
            "ok": true,
            "details": "Enabled (service: my-service, endpoint: http://collector:4318)"
        }
    },
    "metrics": {
        "total_entries": 5000,
        "recent_errors": {
            "lookup": 3,
            "store": 2
        }
    }
}
```

## Component Checks

### Embedding Check

Verifies the embedding model is functioning:

```python
health = cache.health_check()
embedding_check = health["checks"]["embedding"]

if not embedding_check["ok"]:
    print(f"Embedding error: {embedding_check['error']}")
    # - Model not loaded
    # - Embedding dimension mismatch
    # - Model file corrupted
```

**What it tests:**
- Generates test embedding
- Verifies embedding dimension matches expected
- Catches model loading errors

### Database Check

Verifies storage backend is accessible:

```python
database_check = health["checks"]["database"]

if not database_check["ok"]:
    print(f"Database error: {database_check['error']}")
    # - Connection failed
    # - Read/write errors
    # - Disk space issues
```

**What it tests:**
- Counts entries in database
- Reads sample data (if entries exist)
- Catches storage backend errors

### Error Rate Check

Monitors recent error rates:

```python
error_check = health["checks"]["error_rate"]

if not error_check["ok"]:
    print(f"High error rate: {error_check['details']}")
    # "High error rate: 12.5% (125/1000 requests)"
```

**Thresholds:**
- **Healthy**: < 10% error rate
- **Unhealthy**: ≥ 10% error rate
- **Insufficient data**: < 10 total requests

### Scheduler Check

Monitors background scheduler status:

```python
scheduler_check = health["checks"]["schedulers"]

if not scheduler_check["ok"]:
    print(f"Scheduler issues: {scheduler_check['details']}")
    # "1/2 running with 5 errors"
```

**What it checks:**
- Number of running schedulers
- Total errors across all schedulers
- Last run timestamps

### OpenTelemetry Check

Verifies metrics export configuration:

```python
otel_check = health["checks"]["opentelemetry"]

if not otel_check["ok"]:
    print(f"OTEL issue: {otel_check['details']}")
    # "Enabled but exporter failed to initialize"
```

**States:**
- **OK**: Exporter initialized and configured
- **Not OK**: Enabled but initialization failed
- **Disabled**: OTEL not enabled

## HTTP Health Endpoint

### Flask Example

```python
from flask import Flask, jsonify
from reminiscence import Reminiscence

app = Flask(__name__)
cache = Reminiscence()

@app.route("/health")
def health():
    check = cache.health_check()
    status_code = 200 if check["status"] == "healthy" else 503
    return jsonify(check), status_code

@app.route("/health/live")
def liveness():
    """Kubernetes liveness probe"""
    return jsonify({"status": "ok"}), 200

@app.route("/health/ready")
def readiness():
    """Kubernetes readiness probe"""
    check = cache.health_check()
    if check["status"] == "healthy":
        return jsonify({"status": "ready"}), 200
    return jsonify({"status": "not ready"}), 503
```

### FastAPI Example

```python
from fastapi import FastAPI, status
from reminiscence import Reminiscence

app = FastAPI()
cache = Reminiscence()

@app.get("/health")
async def health():
    check = cache.health_check()
    if check["status"] == "healthy":
        return check
    return JSONResponse(content=check, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

@app.get("/health/live")
async def liveness():
    """Liveness probe"""
    return {"status": "ok"}

@app.get("/health/ready")
async def readiness():
    """Readiness probe"""
    check = cache.health_check()
    if check["status"] == "healthy":
        return {"status": "ready"}
    return JSONResponse(
        content={"status": "not ready"},
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE
    )
```

## Kubernetes Integration

### Liveness and Readiness Probes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: reminiscence-cache
spec:
  template:
    spec:
      containers:
      - name: app
        image: my-app:latest
        ports:
        - containerPort: 8080

        livenessProbe:
          httpGet:
            path: /health/live
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 30
          timeoutSeconds: 5
          failureThreshold: 3

        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 2
```

### Startup Probe

```yaml
startupProbe:
  httpGet:
    path: /health/ready
    port: 8080
  initialDelaySeconds: 0
  periodSeconds: 5
  timeoutSeconds: 5
  failureThreshold: 30  # 30 * 5s = 150s max startup time
```

## Docker Health Checks

### Dockerfile HEALTHCHECK

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "app.py"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  cache:
    image: my-app:latest
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 10s
```

## Monitoring Strategies

### 1. Periodic Health Checks

```python
import time
import logging

def monitor_cache_health(cache, interval_seconds=60):
    """Monitor cache health periodically"""
    while True:
        health = cache.health_check()

        if health["status"] != "healthy":
            logging.error(
                "Cache unhealthy",
                extra={
                    "checks": health["checks"],
                    "metrics": health["metrics"]
                }
            )

            # Send alert
            alert_on_call(health)

        time.sleep(interval_seconds)

# Run in background thread
import threading
monitor_thread = threading.Thread(
    target=monitor_cache_health,
    args=(cache,),
    daemon=True
)
monitor_thread.start()
```

### 2. Alerting on Health Changes

```python
import requests

def check_and_alert(cache, last_status={"status": "healthy"}):
    """Alert on health status changes"""
    current = cache.health_check()

    # Status changed from healthy to unhealthy
    if last_status["status"] == "healthy" and current["status"] == "unhealthy":
        send_alert(
            severity="critical",
            message="Cache became unhealthy",
            details=current["checks"]
        )

    # Status recovered
    elif last_status["status"] == "unhealthy" and current["status"] == "healthy":
        send_alert(
            severity="info",
            message="Cache recovered",
            details=current["checks"]
        )

    last_status["status"] = current["status"]
    return current

def send_alert(severity, message, details):
    # Send to PagerDuty, Slack, etc.
    requests.post(
        "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
        json={
            "text": f"[{severity.upper()}] {message}",
            "attachments": [{
                "text": str(details)
            }]
        }
    )
```

### 3. Metrics-Based Monitoring

Combine with metrics for comprehensive monitoring:

```python
def comprehensive_health_check(cache):
    """Extended health check with metrics"""
    health = cache.health_check()
    stats = cache.get_stats()

    # Check hit rate
    if stats["hit_rate"] < 0.5:
        health["checks"]["performance"] = {
            "ok": False,
            "details": f"Low hit rate: {stats['hit_rate']}"
        }
        health["status"] = "unhealthy"

    # Check cache size
    if stats["cache_entries"] / stats["max_entries"] > 0.95:
        health["checks"]["capacity"] = {
            "ok": False,
            "details": "Cache near capacity"
        }
        health["status"] = "unhealthy"

    return health
```

## Common Health Issues

### Issue 1: Embedding Model Not Loaded

**Symptoms:**
```python
{
    "embedding": {
        "ok": false,
        "error": "Model file not found"
    }
}
```

**Solutions:**
- Verify model files exist
- Check `REMINISCENCE_MODEL_NAME` configuration
- Ensure sufficient disk space
- Try `warm_up_embedder=True`

### Issue 2: Database Connection Failed

**Symptoms:**
```python
{
    "database": {
        "ok": false,
        "error": "Connection refused"
    }
}
```

**Solutions:**
- Check `db_uri` configuration
- Verify file permissions for persistent storage
- Ensure parent directory exists
- Check disk space

### Issue 3: High Error Rate

**Symptoms:**
```python
{
    "error_rate": {
        "ok": false,
        "details": "High error rate: 15.2% (152/1000 requests)"
    }
}
```

**Solutions:**
- Check application logs for error patterns
- Verify storage backend health
- Review embedding model performance
- Check network connectivity

### Issue 4: Scheduler Errors

**Symptoms:**
```python
{
    "schedulers": {
        "ok": false,
        "details": "2/2 running with 12 errors"
    }
}
```

**Solutions:**
- Check scheduler logs
- Verify TTL configuration
- Review OTEL endpoint connectivity
- Increase scheduler timeouts

### Issue 5: OTEL Export Failed

**Symptoms:**
```python
{
    "opentelemetry": {
        "ok": false,
        "details": "Enabled but exporter failed to initialize"
    }
}
```

**Solutions:**
- Verify `otel_endpoint` URL
- Check `otel_headers` authentication
- Test endpoint connectivity
- Review firewall rules

## Health Check Best Practices

### 1. Separate Liveness and Readiness

```python
@app.route("/health/live")
def liveness():
    """Liveness: Process is running"""
    return {"status": "ok"}, 200

@app.route("/health/ready")
def readiness():
    """Readiness: Ready to serve traffic"""
    health = cache.health_check()
    if health["status"] == "healthy":
        return {"status": "ready"}, 200
    return {"status": "not ready"}, 503
```

### 2. Set Appropriate Timeouts

```yaml
readinessProbe:
  timeoutSeconds: 5  # Don't wait too long
  periodSeconds: 10  # Check frequently
  failureThreshold: 2  # Mark unhealthy quickly
```

### 3. Monitor Health Check Latency

```python
import time

start = time.time()
health = cache.health_check()
latency_ms = (time.time() - start) * 1000

if latency_ms > 1000:
    logging.warning(
        f"Slow health check: {latency_ms:.1f}ms"
    )
```

### 4. Cache Health Results Briefly

```python
from functools import lru_cache
import time

@lru_cache(maxsize=1)
def get_cached_health(timestamp):
    return cache.health_check()

@app.route("/health")
def health():
    # Cache health check for 10 seconds
    current_bucket = int(time.time() / 10)
    return get_cached_health(current_bucket)
```

### 5. Include Version Information

```python
@app.route("/health")
def health():
    check = cache.health_check()
    check["version"] = {
        "app": os.getenv("APP_VERSION", "unknown"),
        "reminiscence": reminiscence.__version__
    }
    return check
```

## Debugging Health Issues

### Enable Debug Logging

```python
from reminiscence import Reminiscence, ReminiscenceConfig

config = ReminiscenceConfig(
    log_level="DEBUG",
    json_logs=True
)

cache = Reminiscence(config=config)
health = cache.health_check()
# Detailed logs will show what failed
```

### Manual Component Testing

```python
# Test embedding manually
try:
    embedding = cache.embedder.embed("test")
    print(f"✓ Embedding: {len(embedding)} dimensions")
except Exception as e:
    print(f"✗ Embedding failed: {e}")

# Test storage manually
try:
    count = cache.backend.count()
    print(f"✓ Storage: {count} entries")
except Exception as e:
    print(f"✗ Storage failed: {e}")

# Test metrics manually
if cache.metrics:
    stats = cache.metrics.report()
    print(f"✓ Metrics: {stats['total_requests']} requests")
```

## Next Steps

- [OpenTelemetry](/production/opentelemetry/) - Metrics and monitoring
- [Performance](/production/performance/) - Optimization techniques
- [Best Practices](/production/best-practices/) - Production deployment guide
