"""Test direct HTTP export to SigNoz."""

import requests
import time
from dotenv import load_dotenv
from reminiscence import Reminiscence, ReminiscenceConfig

load_dotenv()


def export_metrics_direct(metrics_data, endpoint, service_name):
    """Export metrics directly via HTTP POST."""

    payload = {
        "resourceMetrics": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": service_name}}
                    ]
                },
                "scopeMetrics": [
                    {
                        "scope": {"name": "reminiscence.cache", "version": "0.1.0"},
                        "metrics": [
                            {
                                "name": "cache.hits",
                                "sum": {
                                    "dataPoints": [
                                        {
                                            "asInt": metrics_data["hits"],
                                            "timeUnixNano": int(time.time() * 1e9),
                                        }
                                    ],
                                    "aggregationTemporality": 2,
                                    "isMonotonic": True,
                                },
                            },
                            {
                                "name": "cache.misses",
                                "sum": {
                                    "dataPoints": [
                                        {
                                            "asInt": metrics_data["misses"],
                                            "timeUnixNano": int(time.time() * 1e9),
                                        }
                                    ],
                                    "aggregationTemporality": 2,
                                    "isMonotonic": True,
                                },
                            },
                        ],
                    }
                ],
            }
        ]
    }

    try:
        response = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=5,
        )
        print(f"✅ Export status: {response.status_code}")
        if response.status_code != 200:
            print(f"   Response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False


def main():
    config = ReminiscenceConfig.load()
    cache = Reminiscence(config)

    # Generate traffic
    print("🔄 Generating traffic...")
    for i in range(10):
        cache.store(f"q{i}", {"a": "t"}, f"r{i}")
        cache.lookup(f"q{i}", {"a": "t"})

    # Get metrics
    stats = cache.get_stats()
    print(f"\n📊 Hits: {stats['hits']}, Misses: {stats['misses']}")

    # Export directly
    print(f"\n📤 Exporting to {config.otel_endpoint}...")
    success = export_metrics_direct(
        stats, config.otel_endpoint, config.otel_service_name
    )

    if success:
        print("\n✅ Metrics exported! Wait 30s then check http://localhost:3301")
    else:
        print("\n❌ Export failed!")


if __name__ == "__main__":
    main()
