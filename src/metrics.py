"""Métricas mejoradas para Memora."""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class CacheMetrics:
    """
    Métricas de rendimiento del caché.

    Rastrea hits/misses, latencias y tamaños de resultados.
    """

    hits: int = 0
    misses: int = 0
    total_latency_saved_ms: float = 0.0

    lookup_latencies_ms: List[float] = field(default_factory=list)

    store_errors: int = 0
    lookup_errors: int = 0

    result_sizes_bytes: List[int] = field(default_factory=list)

    # Límite de samples para evitar memory leak
    max_samples: int = 10000

    def record_lookup_latency(self, latency_ms: float):
        """Registra latencia de lookup."""
        self.lookup_latencies_ms.append(latency_ms)

        # Limitar samples
        if len(self.lookup_latencies_ms) > self.max_samples:
            self.lookup_latencies_ms = self.lookup_latencies_ms[-self.max_samples :]

    def record_result_size(self, size_bytes: int):
        """Registra tamaño de resultado cacheado."""
        self.result_sizes_bytes.append(size_bytes)

        if len(self.result_sizes_bytes) > self.max_samples:
            self.result_sizes_bytes = self.result_sizes_bytes[-self.max_samples :]

    @property
    def hit_rate(self) -> float:
        """Tasa de hits (0.0 - 1.0)."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def total_requests(self) -> int:
        """Total de requests."""
        return self.hits + self.misses

    def get_percentiles(self, values: List[float]) -> Dict[str, float]:
        """Calcula percentiles de una lista de valores."""
        if not values:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

        sorted_vals = sorted(values)
        n = len(sorted_vals)

        return {
            "p50": sorted_vals[int(n * 0.50)] if n > 0 else 0.0,
            "p95": sorted_vals[int(n * 0.95)] if n > 0 else 0.0,
            "p99": sorted_vals[int(n * 0.99)] if n > 0 else 0.0,
        }

    def report(self) -> Dict[str, Any]:
        """
        Genera reporte completo de métricas.

        Returns:
            Dict con todas las métricas
        """
        latency_percentiles = self.get_percentiles(self.lookup_latencies_ms)
        size_percentiles = self.get_percentiles(
            [float(s) for s in self.result_sizes_bytes]
        )

        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": self.total_requests,
            "hit_rate": f"{self.hit_rate * 100:.2f}%",  # ← CAMBIO AQUÍ: convertir a string con %
            "total_latency_saved_ms": round(self.total_latency_saved_ms, 1),
            "avg_latency_saved_ms": round(
                self.total_latency_saved_ms / self.hits if self.hits > 0 else 0.0, 1
            ),
            "lookup_latency_ms": {
                "p50": round(latency_percentiles["p50"], 2),
                "p95": round(latency_percentiles["p95"], 2),
                "p99": round(latency_percentiles["p99"], 2),
                "samples": len(self.lookup_latencies_ms),
            },
            "errors": {
                "lookup": self.lookup_errors,
                "store": self.store_errors,
            },
            "result_size_bytes": {
                "p50": int(size_percentiles["p50"]),
                "p95": int(size_percentiles["p95"]),
                "p99": int(size_percentiles["p99"]),
                "samples": len(self.result_sizes_bytes),
            },
        }

    def reset(self):
        """Resetea todas las métricas."""
        self.hits = 0
        self.misses = 0
        self.total_latency_saved_ms = 0.0
        self.lookup_latencies_ms.clear()
        self.result_sizes_bytes.clear()
