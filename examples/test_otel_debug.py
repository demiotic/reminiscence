"""Test minimalista OTLP."""

import time
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME

print("1️⃣ Configurando MeterProvider...")

resource = Resource.create({SERVICE_NAME: "test-direct"})
exporter = OTLPMetricExporter(endpoint="http://localhost:4318/v1/metrics")
reader = PeriodicExportingMetricReader(exporter=exporter, export_interval_millis=5000)
provider = MeterProvider(resource=resource, metric_readers=[reader])
metrics.set_meter_provider(provider)

print("2️⃣ Creando contador...")
meter = metrics.get_meter("test", version="1.0")
counter = meter.create_counter("test_counter", description="Test counter")

print("3️⃣ Incrementando contador...")
for i in range(10):
    counter.add(1)
    print(f"  Count: {i + 1}")

print("4️⃣ Force flush...")
provider.force_flush(timeout_millis=5000)

print("5️⃣ Esperando 10 segundos...")
time.sleep(10)

print("✅ Done! Verifica ClickHouse:")
print("   SELECT * FROM signoz_metrics.distributed_time_series_v4_1day")
print("   WHERE resource_attrs['service.name'] = 'test-direct'")
