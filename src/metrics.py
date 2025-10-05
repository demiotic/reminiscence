"""Métricas de performance del caché."""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class CacheMetrics:
    """Métricas de performance del caché."""

    hits: int = 0
    misses: int = 0
    total_latency_saved_ms: float = 0.0

    def hit_rate(self) -> float:
        """Retorna el hit rate (0.0 a 1.0)."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def report(self) -> Dict[str, Any]:
        """Retorna reporte completo."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate():.2%}",
            "latency_saved_ms": f"{self.total_latency_saved_ms:.1f}",
        }
