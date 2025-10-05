"""Core del caché semántico."""

from functools import wraps
import inspect
import logging
import time
from typing import Any, Callable, Dict, Optional

import numpy as np
import lancedb
import pyarrow as pa
import pyarrow.compute as pc
from sentence_transformers import SentenceTransformer

from .config import CacheConfig
from .metrics import CacheMetrics
from .utils import create_fingerprint, cosine_similarity


logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Caché semántica multilingüe para respuestas de LLM.

    Características:
    - Soporte para 50+ idiomas
    - Fingerprinting basado en contexto (tools, constraints)
    - Métricas de performance integradas
    - TTL automático (opcional)

    Example:
        >>> from memora import SemanticCache, CacheConfig
        >>>
        >>> config = CacheConfig()
        >>> cache = SemanticCache(config)
        >>> response = cache.get_or_compute(
        ...     query="Explícame Python",
        ...     context={"tools": ["search"]},
        ...     compute_fn=lambda q, ctx: llm.generate(q)
        ... )
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Inicializa el caché semántico.

        Args:
            config: Configuración del caché. Si None, usa defaults.
        """
        self.config = config or CacheConfig()

        # Configurar logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        logger.info(f"Inicializando con modelo {self.config.model_name}")

        self.model = SentenceTransformer(self.config.model_name)
        self.db = lancedb.connect(self.config.db_uri)
        self.table = None
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.metrics = CacheMetrics() if self.config.enable_metrics else None

        self.schema = pa.schema(
            [
                pa.field("query_text", pa.string()),
                pa.field("ctx_hash", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.embedding_dim)),
                pa.field("response", pa.string()),
                pa.field("timestamp", pa.int64()),
            ]
        )

        self._init_table()

        logger.info(
            f"Cache listo | dim={self.embedding_dim}, "
            f"threshold={self.config.similarity_threshold}, "
            f"entries={self.table.count_rows()}"
        )

    def _init_table(self):
        """Inicializa o abre la tabla de LanceDB."""
        try:
            self.table = self.db.open_table(self.config.table_name)
            logger.debug(f"Tabla '{self.config.table_name}' abierta")
        except Exception:
            self.table = self.db.create_table(
                self.config.table_name, schema=self.schema, mode="overwrite"
            )
            logger.debug(f"Tabla '{self.config.table_name}' creada")

    def _get_embedding(self, text: str) -> np.ndarray:
        """Genera embedding L2-normalizado."""
        embedding = self.model.encode(
            text, convert_to_numpy=True, normalize_embeddings=True
        )
        return embedding.astype(np.float32)

    def _is_expired(self, timestamp: int) -> bool:
        """Verifica si una entrada expiró según TTL."""
        if self.config.ttl_seconds is None:
            return False
        age_seconds = int(time.time()) - timestamp
        expired = age_seconds > self.config.ttl_seconds
        if expired:
            logger.debug(
                f"Entry expirada: edad={age_seconds}s > TTL={self.config.ttl_seconds}s"
            )
        return expired

    def cached(
        self, context: Optional[Dict[str, Any]] = None, extract_from_args: bool = True
    ):
        """
        Decorador inteligente que combina contexto estático y dinámico.

        Args:
            context: Contexto estático (opcional)
            extract_from_args: Si True, extrae contexto de los argumentos de la función

        Example:
            >>> # Contexto extraído automáticamente de args
            >>> @cache.cached()
            >>> def ask_gpt(query: str, tools: list, model: str) -> str:
            >>>     return call_openai(query, model)
            >>>
            >>> ask_gpt("What is ML?", tools=["search"], model="gpt-4")
            >>> # Contexto: {"tools": ["search"], "model": "gpt-4"}
        """
        base_context = context or {}

        def decorator(func: Callable) -> Callable:
            # ← AQUÍ SE USA INSPECT
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            @wraps(func)
            def wrapper(*args, **kwargs) -> str:
                # Bind args a nombres de parámetros
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()

                # Extract query (primer parámetro)
                query = bound.arguments.get(params[0], "")

                # Build context
                runtime_context = {}

                # Extraer de args (si está habilitado)
                if extract_from_args:
                    extracted = {
                        k: v
                        for k, v in bound.arguments.items()
                        if k != params[0]  # Skip query param
                    }
                    runtime_context.update(extracted)

                # Merge con contexto estático (override)
                runtime_context.update(base_context)

                logger.debug(f"Context: {runtime_context}")

                return self.get_or_compute(
                    query=query,
                    context=runtime_context,
                    compute_fn=lambda q, ctx: func(*args, **kwargs),
                )

            return wrapper

        return decorator

    def get(self, query: str, context: Dict[str, Any]) -> Optional[str]:
        """
        Busca en caché usando similitud coseno.

        Args:
            query: Query del usuario
            context: Contexto (tools, constraints, entities)

        Returns:
            Respuesta cacheada o None si no hay hit
        """
        start_time = time.time()

        if self.table.count_rows() == 0:
            logger.debug("Cache vacío")
            if self.metrics:
                self.metrics.misses += 1
            return None

        ctx_hash = create_fingerprint(context)
        query_embedding = self._get_embedding(query)

        try:
            arrow_table = self.table.to_arrow()
            mask = pc.equal(arrow_table["ctx_hash"], ctx_hash)
            filtered_table = arrow_table.filter(mask)

            num_candidates = len(filtered_table)

            if num_candidates == 0:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(f"MISS | Sin candidatos | {elapsed_ms:.1f}ms")
                if self.metrics:
                    self.metrics.misses += 1
                return None

            # Filtrar entries expiradas ANTES de buscar
            if self.config.ttl_seconds is not None:
                current_time = int(time.time())
                cutoff = current_time - self.config.ttl_seconds

                logger.debug(
                    f"Filtro TTL: current_time={current_time}, "
                    f"cutoff={cutoff}, ttl={self.config.ttl_seconds}s"
                )

                # Debug: ver timestamps ANTES de filtrar
                for i in range(len(filtered_table)):
                    ts = filtered_table["timestamp"][i].as_py()
                    expired = ts <= cutoff
                    logger.debug(
                        f"Entry {i}: timestamp={ts}, "
                        f"age={(current_time - ts)}s, expired={expired}"
                    )

                # Filtrar: mantener solo entries con timestamp > cutoff
                mask_ttl = pc.greater(filtered_table["timestamp"], cutoff)
                filtered_table = filtered_table.filter(mask_ttl)
                num_candidates = len(filtered_table)

                logger.debug(
                    f"Después del filtro TTL: {num_candidates} candidatos restantes"
                )

                if num_candidates == 0:
                    elapsed_ms = (time.time() - start_time) * 1000
                    logger.debug(
                        f"MISS | Todas las entries expiraron | {elapsed_ms:.1f}ms"
                    )
                    if self.metrics:
                        self.metrics.misses += 1
                    return None

            logger.debug(f"Evaluando {num_candidates} candidatos")

            best_similarity = -1.0
            best_idx = -1
            best_query = None

            for i in range(num_candidates):
                candidate_vec = np.array(
                    filtered_table["vector"][i].as_py(), dtype=np.float32
                )
                similarity = cosine_similarity(query_embedding, candidate_vec)

                logger.debug(f"Candidato {i}: similarity={similarity:.4f}")

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_idx = i
                    best_query = filtered_table["query_text"][i].as_py()

            if best_idx == -1 or best_similarity < self.config.similarity_threshold:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.info(
                    f"MISS | sim={best_similarity:.3f} < threshold={self.config.similarity_threshold} "
                    f"| query='{query[:50]}...' | {elapsed_ms:.1f}ms"
                )
                if self.metrics:
                    self.metrics.misses += 1
                return None

            best_response = filtered_table["response"][best_idx].as_py()
            elapsed_ms = (time.time() - start_time) * 1000

            logger.info(
                f"HIT | sim={best_similarity:.3f} | "
                f"query='{query[:50]}...' → match='{best_query[:50]}...' | {elapsed_ms:.1f}ms"
            )

            if self.metrics:
                self.metrics.hits += 1
                self.metrics.total_latency_saved_ms += 2000

            return best_response

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(f"Error en búsqueda: {e} | {elapsed_ms:.1f}ms", exc_info=True)
            if self.metrics:
                self.metrics.misses += 1
            return None

    def set(self, query: str, context: Dict[str, Any], response: str):
        """Guarda en caché."""
        ctx_hash = create_fingerprint(context)
        query_vector = self._get_embedding(query).tolist()
        timestamp = int(time.time())

        data = [
            {
                "query_text": query,
                "ctx_hash": ctx_hash,
                "vector": query_vector,
                "response": response,
                "timestamp": timestamp,
            }
        ]

        self.table.add(data)
        logger.debug(f"Entry guardada | ctx_hash={ctx_hash}")

    def get_or_compute(
        self,
        query: str,
        context: Dict[str, Any],
        compute_fn: Callable[[str, Dict], str],
    ) -> str:
        """
        Busca en caché o computa si no existe.

        Args:
            query: Query del usuario
            context: Contexto
            compute_fn: Función que llama al LLM

        Returns:
            Respuesta (cacheada o computada)
        """
        cached = self.get(query, context)
        if cached is not None:
            return cached

        logger.debug("Computando respuesta")
        start_time = time.time()
        response = compute_fn(query, context)
        elapsed_ms = (time.time() - start_time) * 1000
        logger.debug(f"LLM respondió en {elapsed_ms:.1f}ms")

        self.set(query, context, response)
        return response

    def cleanup_old_entries(self, max_age_seconds: Optional[int] = None):
        """
        Limpia entries expiradas usando LanceDB nativo cuando es posible.

        En disco: usa delete() + compact_files() (eficiente)
        En memoria: usa recreation (único método disponible)
        """
        if max_age_seconds is None:
            max_age_seconds = self.config.ttl_seconds

        if max_age_seconds is None:
            logger.warning("No TTL configurado")
            return 0

        cutoff = int(time.time()) - max_age_seconds
        before = self.table.count_rows()

        if before == 0:
            return 0

        # Estrategia según storage backend
        if self.config.db_uri == "memory://":
            # Memory: no soporta delete nativo, usar recreation
            return self._cleanup_memory(cutoff, before)
        else:
            # Disco: usar delete + compact nativo
            return self._cleanup_disk(cutoff, before)

    def _cleanup_disk(self, cutoff: int, before: int) -> int:
        """Cleanup para storage persistente (usa LanceDB nativo)."""
        try:
            logger.debug(f"Cleanup (disk): usando delete nativo")

            # LanceDB delete con predicado
            self.table.delete(f"timestamp <= {cutoff}")

            # Compactar físicamente (opcional pero recomendado)
            try:
                self.table.compact_files()
                logger.debug("Compaction completada")
            except AttributeError:
                # compact_files() no disponible en versión antigua
                logger.debug(
                    "compact_files() no disponible, usando cleanup_old_versions"
                )
                try:
                    self.table.cleanup_old_versions()
                except AttributeError:
                    logger.debug("cleanup no disponible, skip")

            after = self.table.count_rows()
            deleted = before - after
            logger.info(f"Cleanup (disk): {deleted} entries eliminadas")
            return deleted

        except Exception as e:
            logger.warning(f"Delete nativo falló: {e}, fallback a recreation")
            return self._cleanup_memory(cutoff, before)

    def _cleanup_memory(self, cutoff: int, before: int) -> int:
        """Cleanup para memory:// (usa table recreation)."""
        logger.debug("Cleanup (memory): usando recreation")

        arrow_table = self.table.to_arrow()
        mask_valid = pc.greater(arrow_table["timestamp"], cutoff)
        filtered = arrow_table.filter(mask_valid)

        expired_count = before - len(filtered)

        # Solo recrear si vale la pena (threshold)
        threshold = getattr(self.config, "cleanup_threshold", 0.3)
        if expired_count / before < threshold:
            logger.debug(f"Skip recreation: {expired_count}/{before} < threshold")
            return 0

        # Recrear tabla
        self.table = self.db.create_table(
            self.config.table_name,
            data=filtered if len(filtered) > 0 else None,
            schema=self.schema if len(filtered) == 0 else None,
            mode="overwrite",
        )

        after = self.table.count_rows()
        deleted = before - after
        logger.info(f"Cleanup (memory): {deleted} entries eliminadas via recreation")
        return deleted

    def get_stats(self) -> Dict[str, Any]:
        """Retorna métricas del caché."""
        try:
            model_name = (
                self.model.model_card_data.model_id
                if hasattr(self.model, "model_card_data")
                else "unknown"
            )
        except:
            model_name = str(type(self.model).__name__)

        stats = {
            "total_entries": self.table.count_rows(),
            "threshold": self.config.similarity_threshold,
            "embedding_dim": self.embedding_dim,
            "model": model_name,
            "ttl_seconds": self.config.ttl_seconds,
        }

        if self.metrics:
            stats.update(self.metrics.report())

        return stats
