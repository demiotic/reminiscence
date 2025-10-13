---
title: Background Tasks
description: Configuring schedulers for automatic cleanup and metrics export
---

Reminiscence supports background schedulers for automatic TTL cleanup and metrics export. This guide covers scheduler configuration and management.

## Quick Start

```python
from reminiscence import Reminiscence, ReminiscenceConfig

# Configure with TTL
config = ReminiscenceConfig(
    ttl_seconds=3600,  # 1 hour expiration
    otel_enabled=True  # Enable metrics export
)

cache = Reminiscence(config=config)

# Start background schedulers
cache.start_scheduler()

# Use cache...
result = cache.lookup(query, context)

# Stop schedulers when done
cache.stop_scheduler()
```

## Cleanup Scheduler

Automatically removes expired entries based on TTL.

### Basic Setup

```python
from reminiscence import Reminiscence, ReminiscenceConfig

config = ReminiscenceConfig(
    ttl_seconds=3600  # Enable TTL
)

cache = Reminiscence(config=config)

# Start cleanup scheduler (runs every hour by default)
cache.start_scheduler()
```

### Custom Interval

```python
# Check for expired entries every 30 minutes
cache.start_scheduler(
    interval_seconds=1800  # 30 minutes
)
```

### Initial Delay

```python
# Wait 5 minutes before first cleanup
cache.start_scheduler(
    interval_seconds=3600,
    initial_delay_seconds=300  # 5 minutes
)
```

### Manual Cleanup

Run cleanup manually without scheduler:

```python
# Remove expired entries now
deleted = cache.cleanup_expired()
print(f"Removed {deleted} expired entries")
```

## Metrics Export Scheduler

Automatically exports metrics to OpenTelemetry collector.

### Basic Setup

```python
config = ReminiscenceConfig(
    enable_metrics=True,
    otel_enabled=True,
    otel_endpoint="http://localhost:4318/v1/metrics"
)

cache = Reminiscence(config=config)

# Start metrics export (runs every 60s by default)
cache.start_scheduler(
    metrics_export_interval_seconds=60
)
```

### Custom Export Interval

```python
# Export metrics every 10 seconds
cache.start_scheduler(
    metrics_export_interval_seconds=10
)
```

### Export Without Scheduler

Export metrics manually:

```python
from reminiscence.metrics.exporters import OpenTelemetryExporter

# Get current metrics
metrics = cache.get_stats()

# Export manually
if cache.otel_exporter:
    cache.otel_exporter.export(metrics)
```

## Combined Schedulers

Run both cleanup and metrics export:

```python
config = ReminiscenceConfig(
    ttl_seconds=3600,
    enable_metrics=True,
    otel_enabled=True,
    otel_endpoint="http://collector:4318/v1/metrics"
)

cache = Reminiscence(config=config)

# Start both schedulers
cache.start_scheduler(
    interval_seconds=3600,       # Cleanup every hour
    initial_delay_seconds=60,    # First cleanup after 1 minute
    metrics_export_interval_seconds=30,  # Export every 30s
    metrics_initial_delay_seconds=0      # Export immediately
)
```

## Context Manager

Automatically stop schedulers on exit:

```python
with Reminiscence(config=config) as cache:
    cache.start_scheduler()

    # Use cache...
    result = cache.lookup(query, context)

# Schedulers automatically stopped here
```

## Scheduler Statistics

Monitor scheduler performance:

```python
cache.start_scheduler()

# Use cache for a while...

# Get scheduler stats
stats = cache.get_scheduler_stats()

if stats:
    for name, scheduler_stats in stats.items():
        print(f"\nScheduler: {name}")
        print(f"  Running: {scheduler_stats['running']}")
        print(f"  Total runs: {scheduler_stats['total_runs']}")
        print(f"  Total deleted: {scheduler_stats.get('total_deleted', 0)}")
        print(f"  Errors: {scheduler_stats['errors']}")
        print(f"  Last run: {scheduler_stats['last_run_time']}")
```

## Stopping Schedulers

### Graceful Shutdown

```python
# Stop with 5 second timeout (default)
cache.stop_scheduler()

# Stop with custom timeout
cache.stop_scheduler(timeout=10.0)  # 10 seconds
```

### Force Stop

If schedulers don't stop within timeout, they're forcefully terminated:

```python
cache.stop_scheduler(timeout=1.0)  # Short timeout
# Logs warning if schedulers don't stop in time
```

## Configuration via Environment

Set scheduler intervals via environment variables:

```bash
# Cleanup interval
export REMINISCENCE_CLEANUP_INTERVAL_SECONDS=1800

# Initial delay before first cleanup
export REMINISCENCE_CLEANUP_INITIAL_DELAY=60

# Metrics export interval
export REMINISCENCE_OTEL_EXPORT_INTERVAL_MS=30000  # 30 seconds
```

```python
# Load config from environment
cache = Reminiscence()

# Start with configured intervals
cache.start_scheduler()
```

## Production Setup

Recommended scheduler configuration for production:

```python
from reminiscence import Reminiscence, ReminiscenceConfig

config = ReminiscenceConfig(
    # Storage
    db_uri="./cache.db",

    # TTL cleanup
    ttl_seconds=86400,  # 24 hours
    cleanup_interval_seconds=3600,  # Check every hour
    cleanup_initial_delay=300,  # First check after 5 minutes

    # Metrics
    enable_metrics=True,
    otel_enabled=True,
    otel_endpoint="http://otel-collector:4318/v1/metrics",
    otel_export_interval_ms=30000,  # Export every 30s
    otel_service_name="my-service"
)

cache = Reminiscence(config=config)

# Start both schedulers
cache.start_scheduler(
    interval_seconds=config.cleanup_interval_seconds,
    initial_delay_seconds=config.cleanup_initial_delay,
    metrics_export_interval_seconds=config.otel_export_interval_ms / 1000
)
```

## Health Checks

Monitor scheduler health:

```python
health = cache.health_check()

# Check scheduler status
scheduler_health = health["checks"]["schedulers"]

if scheduler_health["ok"]:
    print(f"✓ Schedulers healthy: {scheduler_health['details']}")
else:
    print(f"✗ Scheduler issues: {scheduler_health['details']}")
```

## Scheduler Patterns

### Pattern 1: Long-Running Service

```python
import signal
import sys

cache = Reminiscence(config=config)
cache.start_scheduler()

def signal_handler(sig, frame):
    print("\nShutting down...")
    cache.stop_scheduler(timeout=10.0)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

print("Service running. Press Ctrl+C to stop.")
signal.pause()
```

### Pattern 2: Kubernetes Deployment

```python
# k8s_app.py
import os
from reminiscence import Reminiscence, ReminiscenceConfig

config = ReminiscenceConfig(
    db_uri=os.getenv("CACHE_DB_URI", "./cache.db"),
    ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "86400")),
    cleanup_interval_seconds=int(os.getenv("CLEANUP_INTERVAL", "3600")),
    otel_enabled=os.getenv("OTEL_ENABLED", "true").lower() == "true",
    otel_endpoint=os.getenv("OTEL_ENDPOINT", "http://otel-collector:4318/v1/metrics"),
    otel_service_name=os.getenv("SERVICE_NAME", "cache-service")
)

cache = Reminiscence(config=config)
cache.start_scheduler()

# Flask/FastAPI app...
app = create_app(cache)

# Graceful shutdown
@app.on_event("shutdown")
async def shutdown():
    cache.stop_scheduler()
```

### Pattern 3: Development Mode

```python
# No schedulers in development
if os.getenv("ENV") == "production":
    cache.start_scheduler()
else:
    # Manual cleanup in development
    print("Dev mode - no background schedulers")
```

### Pattern 4: Systemd Service

```python
#!/usr/bin/env python3
# /usr/local/bin/cache-service

from reminiscence import Reminiscence, ReminiscenceConfig
import signal
import sys
import time

def main():
    config = ReminiscenceConfig(
        db_uri="/var/lib/reminiscence/cache.db",
        ttl_seconds=86400,
        cleanup_interval_seconds=3600,
        otel_enabled=True
    )

    cache = Reminiscence(config=config)
    cache.start_scheduler()

    def shutdown(sig, frame):
        cache.stop_scheduler(timeout=30.0)
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    # Keep service running
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
```

```ini
# /etc/systemd/system/reminiscence-cache.service
[Unit]
Description=Reminiscence Cache Service
After=network.target

[Service]
Type=simple
User=cache
Group=cache
ExecStart=/usr/local/bin/cache-service
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

## Monitoring Schedulers

### Logging

Schedulers log their activity:

```python
# Enable debug logging
config = ReminiscenceConfig(
    log_level="DEBUG",
    json_logs=True
)

cache = Reminiscence(config=config)
cache.start_scheduler()

# Logs will show:
# - scheduler_started
# - cleanup_completed
# - metrics_exported
# - scheduler_stopped
```

### Metrics

Track scheduler metrics via OpenTelemetry:

```python
# Scheduler metrics included in exports
# - reminiscence.scheduler.runs (counter)
# - reminiscence.scheduler.cleanup_latency (histogram)
# - reminiscence.scheduler.errors (counter)
```

### Prometheus Example

Query scheduler metrics:

```promql
# Total cleanup runs
reminiscence_scheduler_runs_total{scheduler="cache_cleanup"}

# Cleanup errors
rate(reminiscence_scheduler_errors_total[5m])

# Cleanup latency (95th percentile)
histogram_quantile(0.95,
  rate(reminiscence_scheduler_cleanup_latency_bucket[5m])
)
```

## Troubleshooting

### Scheduler Not Running

```python
stats = cache.get_scheduler_stats()

if not stats:
    print("No schedulers configured")
    print(f"TTL enabled: {cache.config.ttl_seconds is not None}")
    print(f"OTEL enabled: {cache.config.otel_enabled}")
```

### High Cleanup Latency

```python
# Reduce cleanup interval
cache.start_scheduler(
    interval_seconds=7200  # Run less frequently
)

# Or increase max_entries to reduce eviction pressure
config = ReminiscenceConfig(max_entries=100000)
```

### Scheduler Errors

```python
health = cache.health_check()

schedulers = health["checks"]["schedulers"]
if not schedulers["ok"]:
    print(f"Scheduler issues: {schedulers['details']}")

    # Check individual scheduler stats
    stats = cache.get_scheduler_stats()
    for name, s in stats.items():
        if s["errors"] > 0:
            print(f"Scheduler {name} has {s['errors']} errors")
```

## Performance Considerations

### Cleanup Frequency

Balance between responsiveness and overhead:

```python
# High frequency (responsive but more overhead)
cache.start_scheduler(interval_seconds=300)  # 5 minutes

# Low frequency (less overhead but slower cleanup)
cache.start_scheduler(interval_seconds=7200)  # 2 hours

# Recommended: 1 hour
cache.start_scheduler(interval_seconds=3600)  # Default
```

### Metrics Export Frequency

```python
# High frequency (real-time monitoring, more network traffic)
cache.start_scheduler(metrics_export_interval_seconds=10)

# Low frequency (less overhead, delayed visibility)
cache.start_scheduler(metrics_export_interval_seconds=300)

# Recommended: 30-60 seconds
cache.start_scheduler(metrics_export_interval_seconds=30)
```

## Testing Schedulers

```python
import time

# Start scheduler
cache.start_scheduler(
    interval_seconds=5,  # Short interval for testing
    initial_delay_seconds=0
)

# Wait for first run
time.sleep(6)

# Check stats
stats = cache.get_scheduler_stats()
assert stats["cache_cleanup"]["total_runs"] > 0

# Stop
cache.stop_scheduler()
```

## Next Steps

- [Configuration](/guides/configuration/) - All scheduler configuration options
- [OpenTelemetry](/production/opentelemetry/) - Metrics export setup
- [Health Checks](/production/health-checks/) - Monitoring scheduler health
- [Best Practices](/production/best-practices/) - Production deployment guide
