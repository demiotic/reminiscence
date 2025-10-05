"""Tipos de datos para Memora."""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import pyarrow as pa


@dataclass
class CacheEntry:
    """
    Entrada individual en el caché.

    Representa un resultado almacenado con su metadata asociada.
    """

    query_text: str
    context_hash: str
    embedding: pa.Array  # FixedSizeListArray de float32
    result: Any
    timestamp: int
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LookupResult:
    """
    Resultado de una operación de búsqueda en caché.

    Attributes:
        hit: True si se encontró match válido
        result: Datos recuperados (None si miss)
        similarity: Score de similitud (0-1)
        matched_query: Query original que hizo match
        age_seconds: Antigüedad de la entrada
    """

    hit: bool
    result: Optional[Any] = None
    similarity: Optional[float] = None
    matched_query: Optional[str] = None
    age_seconds: Optional[int] = None

    @property
    def is_hit(self) -> bool:
        """Alias para compatibilidad con diferentes estilos de código."""
        return self.hit

    @property
    def is_miss(self) -> bool:
        """Inverso de is_hit."""
        return not self.hit


@dataclass
class AvailabilityCheck:
    """
    Resultado de verificación de disponibilidad.

    Usado por planificadores para saber si existe caché sin recuperar datos.
    """

    available: bool
    age_seconds: Optional[int] = None
    ttl_remaining_seconds: Optional[int] = None
    similarity: Optional[float] = None

    @property
    def is_fresh(self) -> bool:
        """Retorna True si la entrada es reciente (< 50% del TTL consumido)."""
        if self.ttl_remaining_seconds is None or self.age_seconds is None:
            return True
        total_ttl = self.age_seconds + self.ttl_remaining_seconds
        return self.age_seconds < (total_ttl * 0.5)


@dataclass
class StoreRequest:
    """
    Request para almacenar en caché (usado en modo remoto).
    """

    query: str
    context: Dict[str, Any]
    result: Any
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LookupRequest:
    """
    Request de búsqueda (usado en modo remoto).
    """

    query: str
    context: Optional[Dict[str, Any]] = None
    similarity_threshold: Optional[float] = None


@dataclass
class InvalidateRequest:
    """
    Request de invalidación.
    """

    query: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    older_than_seconds: Optional[int] = None
