"""Core de Memora - Caché semántica consultable."""

import logging
import time
from typing import Any, Dict, List, Optional

import lancedb
import pyarrow as pa
import pyarrow.compute as pc
from sentence_transformers import SentenceTransformer

from .config import CacheConfig
from .metrics import CacheMetrics
from .types import CacheEntry, LookupResult, AvailabilityCheck
from .utils import create_fingerprint, cosine_similarity, serialize, deserialize

logger = logging.getLogger(__name__)


class Memora:
    """
    Caché semántica para sistemas multiagente.

    Diseño: Solo almacenamiento y consulta. NO ejecuta lógica.

    API Principal:
    - lookup(): Busca resultado existente
    - store(): Guarda nuevo resultado
    - check_availability(): Verifica disponibilidad sin recuperar datos
    - invalidate(): Marca entradas como inválidas
    - create_index(): Crea índice vectorial para búsquedas rápidas

    Example:
        >>> memora = Memora(CacheConfig.for_development())
        >>>
        >>> # Consultar
        >>> result = memora.lookup(
        ...     query="Analiza ventas Q3",
        ...     context={"agent": "sql", "db": "prod"}
        ... )
        >>>
        >>> if result.is_hit:
        ...     print(result.result)
        ... else:
        ...     # Ejecutar agente externamente
        ...     data = execute_agent(...)
        ...     memora.store(query, context, data)
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """Inicializa Memora."""
        self.config = config or CacheConfig()

        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        logger.info(f"Inicializando Memora | modelo={self.config.model_name}")

        # Inicializar componentes
        self.model = SentenceTransformer(self.config.model_name)
        self.db = lancedb.connect(self.config.db_uri)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.metrics = CacheMetrics() if self.config.enable_metrics else None

        # Schema - timestamp en milisegundos, result y metadata como binary
        self.schema = pa.schema(
            [
                pa.field("query_text", pa.string()),
                pa.field("context_hash", pa.string()),
                pa.field("embedding", pa.list_(pa.float32(), self.embedding_dim)),
                pa.field("result", pa.binary()),  # bytes, no string
                pa.field("timestamp", pa.int64()),  # Milisegundos desde epoch
                pa.field("metadata", pa.binary()),  # bytes, no string
            ]
        )

        self.table = self._init_table()
        self._index_created = False

        logger.info(
            f"Memora listo | dim={self.embedding_dim}, "
            f"threshold={self.config.similarity_threshold}, "
            f"entries={self.table.count_rows()}"
        )

        # Auto-crear índice si está configurado
        if self.config.auto_create_index:
            self._maybe_auto_create_index()

    def _init_table(self):
        """Inicializa o abre tabla LanceDB."""
        try:
            table = self.db.open_table(self.config.table_name)
            logger.debug(f"Tabla '{self.config.table_name}' abierta")
            return table
        except Exception:
            table = self.db.create_table(
                self.config.table_name, schema=self.schema, mode="overwrite"
            )
            logger.debug(f"Tabla '{self.config.table_name}' creada")
            return table

    def _maybe_auto_create_index(self):
        """Crea índice automáticamente si se alcanza el threshold."""
        if self._index_created:
            return

        row_count = self.table.count_rows()
        if row_count >= self.config.index_threshold_entries:
            logger.info(
                f"Auto-creando índice: {row_count} >= {self.config.index_threshold_entries}"
            )
            self.create_index(num_partitions=self.config.index_num_partitions)
            self._index_created = True

    def _embed(self, text: str) -> List[float]:
        """Genera embedding L2-normalizado."""
        embedding_np = self.model.encode(
            text, convert_to_numpy=True, normalize_embeddings=True
        )
        return embedding_np.tolist()

    def _current_timestamp_ms(self) -> int:
        """Retorna timestamp actual en milisegundos."""
        return int(time.time() * 1000)

    def _is_expired(self, timestamp_ms: int) -> bool:
        """Verifica expiración según TTL."""
        if self.config.ttl_seconds is None:
            return False
        age_ms = self._current_timestamp_ms() - timestamp_ms
        age_seconds = age_ms / 1000
        return age_seconds > self.config.ttl_seconds

    def lookup(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> LookupResult:
        """
        Busca entrada en caché por similitud semántica.

        Args:
            query: Query del usuario
            context: Contexto (agent_id, tools, params, etc.)
            similarity_threshold: Override del threshold global

        Returns:
            LookupResult con hit/miss y datos asociados
        """
        start_time = time.time()
        context = context or {}
        threshold = similarity_threshold or self.config.similarity_threshold

        # Cache vacío
        if self.table.count_rows() == 0:
            logger.debug("Caché vacío")
            if self.metrics:
                self.metrics.misses += 1
            return LookupResult(hit=False)

        # Preparar búsqueda
        context_hash = create_fingerprint(context)
        query_embedding = self._embed(query)

        try:
            # Buscar candidatos (sin filtro de contexto en search, lo haremos después)
            search_results = (
                self.table.search(query_embedding)
                .limit(50)  # Más candidatos para filtrar después
                .to_arrow()
            )

            if len(search_results) == 0:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(f"MISS | Search sin resultados | {elapsed_ms:.1f}ms")
                if self.metrics:
                    self.metrics.misses += 1
                    self.metrics.record_lookup_latency(elapsed_ms)
                return LookupResult(hit=False)

            # Filtrar por context_hash manualmente (más confiable)
            mask_context = pc.equal(search_results["context_hash"], context_hash)
            search_results = search_results.filter(mask_context)

            if len(search_results) == 0:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(
                    f"MISS | Ningún resultado con contexto matching | {elapsed_ms:.1f}ms"
                )
                if self.metrics:
                    self.metrics.misses += 1
                    self.metrics.record_lookup_latency(elapsed_ms)
                return LookupResult(hit=False)

            # Filtrar por TTL (timestamp en ms)
            if self.config.ttl_seconds is not None:
                cutoff_ms = self._current_timestamp_ms() - int(
                    self.config.ttl_seconds * 1000
                )
                mask_ttl = pc.greater(search_results["timestamp"], cutoff_ms)
                search_results = search_results.filter(mask_ttl)

                if len(search_results) == 0:
                    logger.debug("MISS | Resultados expiraron")
                    if self.metrics:
                        self.metrics.misses += 1
                    return LookupResult(hit=False)

            # El primer resultado es el más similar
            best_idx = 0
            best_query = search_results["query_text"][best_idx].as_py()
            best_embedding = search_results["embedding"][best_idx].as_py()
            best_sim = cosine_similarity(query_embedding, best_embedding)

            # Evaluar threshold
            if best_sim < threshold:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.info(
                    f"MISS | sim={best_sim:.3f} < threshold={threshold} | "
                    f"query='{query[:50]}...' | {elapsed_ms:.1f}ms"
                )
                if self.metrics:
                    self.metrics.misses += 1
                    self.metrics.record_lookup_latency(elapsed_ms)
                return LookupResult(hit=False, similarity=best_sim)

            # HIT - Deserializar resultado
            result_bytes = search_results["result"][best_idx].as_py()
            result_data = deserialize(result_bytes)

            timestamp_ms = search_results["timestamp"][best_idx].as_py()
            age_ms = self._current_timestamp_ms() - timestamp_ms
            age_seconds = int(age_ms / 1000)
            elapsed_ms = (time.time() - start_time) * 1000

            logger.info(
                f"HIT | sim={best_sim:.3f} | "
                f"query='{query[:50]}...' → '{best_query[:50]}...' | "
                f"age={age_seconds}s | {elapsed_ms:.1f}ms"
            )

            if self.metrics:
                self.metrics.hits += 1
                self.metrics.total_latency_saved_ms += 2000
                self.metrics.record_lookup_latency(elapsed_ms)

            return LookupResult(
                hit=True,
                result=result_data,
                similarity=best_sim,
                matched_query=best_query,
                age_seconds=age_seconds,
            )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(f"Error en lookup: {e} | {elapsed_ms:.1f}ms", exc_info=True)
            if self.metrics:
                self.metrics.misses += 1
                self.metrics.record_lookup_latency(elapsed_ms)
            return LookupResult(hit=False)

    def store(
        self,
        query: str,
        context: Dict[str, Any],
        result: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Almacena resultado en caché.

        Args:
            query: Query del usuario
            context: Contexto del agente
            result: Resultado a cachear (str, dict, etc.)
            metadata: Metadata adicional (opcional)
        """
        context_hash = create_fingerprint(context)
        embedding = self._embed(query)
        timestamp = self._current_timestamp_ms()  # Milisegundos

        # Serializar result y metadata a bytes
        result_bytes = serialize(result)
        metadata_bytes = serialize(metadata) if metadata else b""

        # Registrar tamaño en métricas
        if self.metrics:
            self.metrics.record_result_size(len(result_bytes))

        data = [
            {
                "query_text": query,
                "context_hash": context_hash,
                "embedding": embedding,
                "result": result_bytes,
                "timestamp": timestamp,
                "metadata": metadata_bytes,
            }
        ]

        self.table.add(data)
        logger.debug(f"Stored | ctx_hash={context_hash[:8]} | query='{query[:50]}...'")

        # Auto-crear índice si es necesario
        if self.config.auto_create_index:
            self._maybe_auto_create_index()

    def check_availability(
        self,
        query: str,
        context: Dict[str, Any],
        similarity_threshold: Optional[float] = None,
    ) -> AvailabilityCheck:
        """
        Verifica disponibilidad sin recuperar datos completos.

        Útil para planificadores que solo necesitan saber si existe cache.

        Args:
            query: Query a verificar
            context: Contexto
            similarity_threshold: Override del threshold

        Returns:
            AvailabilityCheck con metadata mínima
        """
        result = self.lookup(query, context, similarity_threshold)

        if not result.is_hit:
            return AvailabilityCheck(available=False)

        ttl_remaining = None
        if self.config.ttl_seconds and result.age_seconds is not None:
            ttl_remaining = self.config.ttl_seconds - result.age_seconds

        return AvailabilityCheck(
            available=True,
            age_seconds=result.age_seconds,
            ttl_remaining_seconds=ttl_remaining,
            similarity=result.similarity,
        )

    def invalidate(
        self,
        query: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        older_than_seconds: Optional[float] = None,
    ) -> int:
        """
        Invalida entradas del caché.

        Args:
            query: Si especificado, invalida matches semánticos
            context: Si especificado, invalida por context_hash
            older_than_seconds: Si especificado, invalida entries antiguas (acepta decimales)

        Returns:
            Número de entradas invalidadas
        """
        if query is None and context is None and older_than_seconds is None:
            logger.warning("invalidate() llamado sin criterios, ignorando")
            return 0

        before = self.table.count_rows()

        # Invalidar por edad
        if older_than_seconds is not None:
            # Convertir segundos a milisegundos
            cutoff_ms = self._current_timestamp_ms() - int(older_than_seconds * 1000)
            return self._cleanup_by_timestamp(cutoff_ms, before)

        # Invalidar por contexto
        if context is not None:
            context_hash = create_fingerprint(context)
            return self._delete_by_hash(context_hash, before)

        # Invalidar por query (semántico)
        if query is not None:
            logger.warning("Invalidación semántica no implementada aún")
            return 0

    def _delete_by_hash(self, context_hash: str, before: int) -> int:
        """Elimina entries con context_hash específico."""
        if self.config.db_uri == "memory://":
            arrow_table = self.table.to_arrow()
            mask = pc.not_equal(arrow_table["context_hash"], context_hash)
            filtered = arrow_table.filter(mask)

            self.table = self.db.create_table(
                self.config.table_name,
                data=filtered if len(filtered) > 0 else None,
                schema=self.schema if len(filtered) == 0 else None,
                mode="overwrite",
            )
        else:
            self.table.delete(f"context_hash = '{context_hash}'")
            try:
                self.table.compact_files()
            except AttributeError:
                pass

        after = self.table.count_rows()
        deleted = before - after
        logger.info(f"Invalidated {deleted} entries with ctx_hash={context_hash[:8]}")
        return deleted

    def _cleanup_by_timestamp(self, cutoff_ms: int, before: int) -> int:
        """Elimina entries con timestamp <= cutoff_ms (más antiguas que cutoff)."""
        if self.config.db_uri == "memory://":
            arrow_table = self.table.to_arrow()
            # Mantener solo entries MÁS NUEVAS que cutoff
            mask = pc.greater(arrow_table["timestamp"], cutoff_ms)
            filtered = arrow_table.filter(mask)

            self.table = self.db.create_table(
                self.config.table_name,
                data=filtered if len(filtered) > 0 else None,
                schema=self.schema if len(filtered) == 0 else None,
                mode="overwrite",
            )
        else:
            # Borrar entries MÁS VIEJAS que cutoff
            self.table.delete(f"timestamp <= {cutoff_ms}")
            try:
                self.table.compact_files()
            except AttributeError:
                pass

        after = self.table.count_rows()
        deleted = before - after
        logger.info(f"Cleaned up {deleted} expired entries (cutoff: {cutoff_ms}ms)")
        return deleted

    def cleanup_expired(self) -> int:
        """
        Limpia entries expiradas según TTL configurado.

        Returns:
            Número de entries eliminadas
        """
        if self.config.ttl_seconds is None:
            logger.warning("No TTL configurado, skip cleanup")
            return 0

        cutoff_ms = self._current_timestamp_ms() - int(self.config.ttl_seconds * 1000)
        before = self.table.count_rows()

        return self._cleanup_by_timestamp(cutoff_ms, before)

    def create_index(
        self,
        num_partitions: int = 256,
        num_sub_vectors: Optional[int] = None,
    ) -> None:
        """
        Crea índice IVF-PQ para búsquedas vectoriales rápidas.

        IMPORTANTE: Solo útil con >256 entries. Para menos, usa ANN lineal.

        Args:
            num_partitions: Número de clusters IVF (default: 256)
            num_sub_vectors: Subvectores para PQ (default: embedding_dim // 4)

        Example:
            >>> memora = Memora(CacheConfig.for_production())
            >>> # ... agregar >1000 entradas ...
            >>> memora.create_index(num_partitions=512)
        """
        row_count = self.table.count_rows()

        if row_count < 256:
            logger.warning(
                f"Solo {row_count} entradas - índice no recomendado. "
                "Se requieren al menos 256 entradas."
            )
            return

        if num_sub_vectors is None:
            num_sub_vectors = max(1, self.embedding_dim // 4)

        logger.info(
            f"Creando índice vectorial: partitions={num_partitions}, "
            f"sub_vectors={num_sub_vectors}, entries={row_count}"
        )

        try:
            self.table.create_index(
                num_partitions=num_partitions,
                num_sub_vectors=num_sub_vectors,
            )
            self._index_created = True
            logger.info("Índice creado exitosamente")
        except Exception as e:
            logger.error(f"Error al crear índice: {e}", exc_info=True)
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del caché."""
        stats = {
            "total_entries": self.table.count_rows(),
            "threshold": self.config.similarity_threshold,
            "embedding_dim": self.embedding_dim,
            "model": self.config.model_name,
            "ttl_seconds": self.config.ttl_seconds,
            "storage": self.config.db_uri,
            "index_created": self._index_created,
        }

        if self.metrics:
            stats.update(self.metrics.report())

        return stats

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Retorna estadísticas del índice vectorial.

        Returns:
            Dict con metadata del índice (o None si no existe)
        """
        return {
            "has_index": self._index_created,
            "total_entries": self.table.count_rows(),
            "note": "LanceDB no expone métricas detalladas del índice",
        }
