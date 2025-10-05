"""Configuración del caché."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CacheConfig:
    """
    Configuración para SemanticCache.

    Attributes:
        model_name: Modelo de sentence-transformers
            - 'paraphrase-multilingual-MiniLM-L12-v2' (default): 384 dims, 50+ idiomas
            - 'all-MiniLM-L6-v2': 384 dims, solo inglés (más rápido)
            - 'paraphrase-multilingual-mpnet-base-v2': 768 dims, mejor calidad
        similarity_threshold: Umbral de similitud coseno
            - 0.75-0.80 para modelos multilingües (recomendado)
            - 0.85-0.90 para modelos monolingües
        db_uri: URI de LanceDB
            - 'memory://' para in-memory (no persiste)
            - './cache.db' para persistencia en disco
        table_name: Nombre de la tabla en LanceDB
        enable_metrics: Si True, recolecta métricas de performance
        ttl_seconds: Time-to-live en segundos (None = sin expiración)
        log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR)
        cleanup_threshold: Threshold para limpieza automática
        auto_create_index: Si True, crea índice automáticamente
        index_threshold_entries: Mínimo de entradas antes de crear índice
        index_num_partitions: Número de particiones IVF para el índice
    """

    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    similarity_threshold: float = 0.75
    db_uri: str = "memory://"
    table_name: str = "semantic_cache"
    enable_metrics: bool = True
    ttl_seconds: Optional[int] = None
    log_level: str = "INFO"
    cleanup_threshold: float = 0.3

    # Configuración de índices vectoriales
    auto_create_index: bool = False
    index_threshold_entries: int = 256
    index_num_partitions: int = 256

    @classmethod
    def for_production(cls, db_path: str = "./cache.db") -> "CacheConfig":
        """
        Configuración optimizada para producción.

        Args:
            db_path: Path para persistencia del caché

        Returns:
            CacheConfig configurado para producción
        """
        return cls(
            db_uri=db_path,
            ttl_seconds=86400,  # 24 horas
            enable_metrics=True,
            log_level="INFO",
            auto_create_index=True,
            index_threshold_entries=1000,
            index_num_partitions=512,
        )

    @classmethod
    def for_development(cls) -> "CacheConfig":
        """
        Configuración para desarrollo/testing.

        Returns:
            CacheConfig configurado para desarrollo
        """
        return cls(
            db_uri="memory://",
            enable_metrics=True,
            log_level="DEBUG",
            auto_create_index=False,  # Manual en dev
        )
