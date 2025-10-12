"""Core LanceDB backend with singleton pattern and dual tables."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import lancedb
import pyarrow as pa

from ..base import StorageBackend
from ..schemas import create_exact_schema, create_semantic_schema
from ...compression import create_compressor
from ...serialization import ResultSerializer
from ...types import CacheEntry
from ...utils.fingerprint import compute_query_hash, create_fingerprint
from ...utils.logging import get_logger
from .deletion import DeletionMixin
from .indexing import IndexingMixin
from .search import SearchMixin

logger = get_logger(__name__)


class LanceDBBackend(SearchMixin, DeletionMixin, IndexingMixin, StorageBackend):
    """LanceDB implementation with dual exact/semantic tables.

    Uses mixins for:
    - SearchMixin: search(), _search_exact(), _search_semantic()
    - DeletionMixin: delete_by_id(), delete_by_filter(), clear()
    - IndexingMixin: has_index(), create_index(), maybe_auto_create_index()
    """

    _instances: Dict[tuple, "LanceDBBackend"] = {}

    def __new__(cls, config, embedding_dim: int, metrics=None):
        """Create or return existing instance for the same db_uri + embedding_dim.

        Args:
            config: Configuration object.
            embedding_dim: Dimension of embedding vectors.
            metrics: Optional metrics tracker.

        Returns:
            Singleton instance for the given (db_uri, embedding_dim) pair.
        """
        key = (config.db_uri, embedding_dim)

        if key not in cls._instances:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instances[key] = instance
            logger.debug(
                "storage_backend_created",
                db_uri=config.db_uri,
                embedding_dim=embedding_dim,
            )
        else:
            logger.debug(
                "storage_backend_reused",
                db_uri=config.db_uri,
                embedding_dim=embedding_dim,
            )

        return cls._instances[key]

    def __init__(self, config, embedding_dim: int, metrics=None):
        """Initialize LanceDB backend with dual tables.

        Args:
            config: Configuration object.
            embedding_dim: Dimension of embedding vectors.
            metrics: Optional metrics tracker.
        """
        if self._initialized:
            logger.debug("storage_backend_already_initialized", db_uri=config.db_uri)
            return

        init_start = time.perf_counter()

        self.config = config
        self.embedding_dim = embedding_dim
        self.metrics = metrics

        logger.debug("connecting_to_lancedb", db_uri=config.db_uri)
        self.db = lancedb.connect(config.db_uri)

        logger.debug("creating_schemas", embedding_dim=embedding_dim)
        self.exact_schema = create_exact_schema()
        self.semantic_schema = create_semantic_schema(embedding_dim)

        self._exact_table_name = f"{config.table_name}_exact"
        self._semantic_table_name = f"{config.table_name}_semantic"

        self.exact_table = self._init_table(self._exact_table_name, self.exact_schema)
        self.semantic_table = self._init_table(
            self._semantic_table_name, self.semantic_schema
        )

        self.table = self.semantic_table
        self.schema = self.semantic_schema

        # Initialize encryption if enabled
        encryptor = None
        if config.encryption_enabled:
            logger.debug("initializing_encryption", backend=config.encryption_backend)
            if config.encryption_backend == "age":
                from reminiscence.encryption import AgeEncryption

                encryptor = AgeEncryption(
                    key=config.encryption_key,
                    max_workers=config.encryption_max_workers,
                )
            else:
                raise ValueError(
                    f"Unsupported encryption backend: {config.encryption_backend}. "
                    f"Currently only 'age' is supported."
                )
            logger.info(
                "encryption_initialized",
                backend=config.encryption_backend,
                max_workers=config.encryption_max_workers,
            )

        # Initialize compression if enabled
        compressor = None
        if config.compression_enabled:
            logger.debug(
                "initializing_compression",
                algorithm=config.compression_algorithm,
                level=config.compression_level,
            )
            compressor = create_compressor(
                algorithm=config.compression_algorithm,
                level=config.compression_level,
            )
            logger.info(
                "compression_initialized",
                algorithm=config.compression_algorithm,
                level=config.compression_level,
            )

        logger.debug(
            "initializing_serializer",
            has_encryptor=encryptor is not None,
            has_compressor=compressor is not None,
        )
        self.serializer = ResultSerializer(encryptor=encryptor, compressor=compressor)

        self._index_created = False
        self._initialized = True

        init_ms = (time.perf_counter() - init_start) * 1000
        logger.info(
            "storage_backend_initialized",
            db_uri=config.db_uri,
            exact_table=self._exact_table_name,
            semantic_table=self._semantic_table_name,
            embedding_dim=embedding_dim,
            encryption=config.encryption_enabled,
            compression=config.compression_enabled,
            init_ms=round(init_ms, 1),
        )

    def _init_table(self, table_name: str, schema: pa.Schema):
        """Initialize or open a specific table.

        Args:
            table_name: Name of the table.
            schema: PyArrow schema for the table.

        Returns:
            LanceDB table instance.
        """
        try:
            table = self.db.open_table(table_name)
            logger.debug("table_opened", name=table_name, rows=table.count_rows())
            return table
        except Exception:
            table = self.db.create_table(table_name, schema=schema, mode="overwrite")
            logger.debug("table_created", name=table_name)
            return table

    @classmethod
    def _clear_instances(cls) -> None:
        """Clear all singleton instances (for testing only)."""
        logger.debug("clearing_storage_instances", count=len(cls._instances))
        cls._instances.clear()

    def _generate_id(self, entry: CacheEntry) -> str:
        """Generate unique ID for entry using SHA256 hash.

        Args:
            entry: Cache entry.

        Returns:
            Unique identifier string.
        """
        return compute_query_hash(entry.query_text, entry.context)

    def count(self) -> int:
        """Get total entries across both tables.

        Returns:
            Total number of cache entries.
        """
        exact_count = self.exact_table.count_rows()
        semantic_count = self.semantic_table.count_rows()
        total = exact_count + semantic_count

        logger.debug(
            "storage_count",
            exact=exact_count,
            semantic=semantic_count,
            total=total,
        )
        return total

    def add(self, entries: List[CacheEntry]) -> None:
        """Add entries to appropriate tables.

        Args:
            entries: List of cache entries to store.
        """
        start = time.perf_counter()

        if not entries:
            logger.debug("add_skipped_empty_entries")
            return

        logger.debug(
            "add_start",
            entries=len(entries),
            has_encryptor=self.serializer.encryptor is not None,
            has_compressor=self.serializer.compressor is not None,
        )

        prep_start = time.perf_counter()
        query_texts = [e.query_text for e in entries]
        contexts = [e.context for e in entries]
        timestamps = [e.timestamp for e in entries]

        context_jsons = [json.dumps(c, sort_keys=True) for c in contexts]
        context_hashes = [create_fingerprint(c) for c in contexts]

        prep_ms = (time.perf_counter() - prep_start) * 1000
        logger.debug(
            "add_preparation_complete",
            entries=len(entries),
            prep_ms=round(prep_ms, 1),
        )

        exact_data = []
        semantic_data = []

        results = [e.result for e in entries]
        serialize_start = time.perf_counter()
        logger.debug(
            "serialization_start",
            count=len(results),
            batch=True,
        )

        try:
            serialized_results = self.serializer.serialize_batch(results)
            serialize_ms = (time.perf_counter() - serialize_start) * 1000
            logger.debug(
                "batch_serialization_complete",
                count=len(serialized_results),
                latency_ms=round(serialize_ms, 1),
                per_item_ms=round(serialize_ms / len(results), 2) if results else 0,
            )
        except Exception as e:
            serialize_ms = (time.perf_counter() - serialize_start) * 1000
            logger.error(
                "batch_serialization_failed",
                error=str(e),
                error_type=type(e).__name__,
                latency_ms=round(serialize_ms, 1),
            )
            if self.metrics:
                if not hasattr(self.metrics, "storage_add_errors"):
                    self.metrics.storage_add_errors = 0
                self.metrics.storage_add_errors += 1
            return

        build_start = time.perf_counter()
        for i, entry in enumerate(entries):
            ser_result, result_type = serialized_results[i]
            if ser_result is None:
                continue

            base_data = {
                "id": self._generate_id(entry),
                "query_text": query_texts[i],
                "context": context_jsons[i],
                "context_hash": context_hashes[i],
                "result": ser_result,
                "result_type": result_type,
                "timestamp": timestamps[i],
                "metadata": json.dumps(entry.metadata) if entry.metadata else "{}",
            }

            if entry.metadata and "query_mode" in entry.metadata:
                detected_mode = entry.metadata["query_mode"]
            else:
                detected_mode = "semantic"
                logger.warning(
                    "entry_missing_query_mode",
                    index=i,
                    query_preview=query_texts[i][:50],
                    fallback="semantic",
                )

            logger.debug(
                "entry_routing",
                index=i,
                query_preview=query_texts[i][:50],
                mode=detected_mode,
            )

            if detected_mode == "exact":
                exact_data.append(
                    {
                        **base_data,
                        "query_hash": compute_query_hash(query_texts[i], contexts[i]),
                    }
                )
            else:
                semantic_data.append(
                    {
                        **base_data,
                        "embedding": entry.embedding,
                    }
                )

        build_ms = (time.perf_counter() - build_start) * 1000
        logger.debug(
            "data_dicts_built",
            exact=len(exact_data),
            semantic=len(semantic_data),
            latency_ms=round(build_ms, 1),
        )

        total_added = 0

        if exact_data:
            exact_add_start = time.perf_counter()
            try:
                self.exact_table.add(exact_data)
                exact_add_ms = (time.perf_counter() - exact_add_start) * 1000
                total_added += len(exact_data)
                logger.debug(
                    "exact_table_add_success",
                    count=len(exact_data),
                    latency_ms=round(exact_add_ms, 1),
                )
            except Exception as e:
                exact_add_ms = (time.perf_counter() - exact_add_start) * 1000
                logger.error(
                    "exact_table_add_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    latency_ms=round(exact_add_ms, 1),
                    exc_info=True,
                )
                if self.metrics:
                    if not hasattr(self.metrics, "storage_add_errors"):
                        self.metrics.storage_add_errors = 0
                    self.metrics.storage_add_errors += 1

        if semantic_data:
            semantic_add_start = time.perf_counter()
            try:
                self.semantic_table.add(semantic_data)
                semantic_add_ms = (time.perf_counter() - semantic_add_start) * 1000
                total_added += len(semantic_data)
                logger.debug(
                    "semantic_table_add_success",
                    count=len(semantic_data),
                    latency_ms=round(semantic_add_ms, 1),
                )
            except Exception as e:
                semantic_add_ms = (time.perf_counter() - semantic_add_start) * 1000
                logger.error(
                    "semantic_table_add_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    latency_ms=round(semantic_add_ms, 1),
                    exc_info=True,
                )
                if self.metrics:
                    if not hasattr(self.metrics, "storage_add_errors"):
                        self.metrics.storage_add_errors = 0
                    self.metrics.storage_add_errors += 1

        elapsed_ms = (time.perf_counter() - start) * 1000

        if self.metrics and total_added > 0:
            if not hasattr(self.metrics, "storage_adds"):
                self.metrics.storage_adds = 0
            self.metrics.storage_adds += total_added

            if not hasattr(self.metrics, "storage_add_latencies_ms"):
                from collections import deque

                self.metrics.storage_add_latencies_ms = deque(maxlen=1000)
            self.metrics.storage_add_latencies_ms.append(elapsed_ms)

        logger.info(
            "storage_add_complete",
            exact_entries=len(exact_data),
            semantic_entries=len(semantic_data),
            total_added=total_added,
            total_ms=round(elapsed_ms, 1),
            per_item_ms=round(elapsed_ms / len(entries), 2) if entries else 0,
        )

    def _arrow_row_to_cache_entry(
        self, arrow_table: pa.Table, index: int, similarity: float
    ) -> Optional[CacheEntry]:
        """Convert Arrow row to CacheEntry.

        Args:
            arrow_table: Arrow table containing the row.
            index: Row index to convert.
            similarity: Similarity score for this entry.

        Returns:
            CacheEntry or None if conversion fails.
        """
        try:
            context_dict = json.loads(arrow_table["context"][index].as_py())
            result_data = arrow_table["result"][index].as_py()
            result_type = arrow_table["result_type"][index].as_py()

            deserialize_start = time.perf_counter()
            result_obj = self.serializer.deserialize(result_data, result_type)
            deserialize_ms = (time.perf_counter() - deserialize_start) * 1000

            if deserialize_ms > 10:
                logger.debug(
                    "deserialization_slow",
                    latency_ms=round(deserialize_ms, 1),
                    result_type=result_type,
                )

            metadata_str = arrow_table["metadata"][index].as_py()
            metadata_obj = (
                json.loads(metadata_str)
                if metadata_str and metadata_str != "{}"
                else None
            )

            embedding = None
            if "embedding" in arrow_table.schema.names:
                embedding = list(arrow_table["embedding"][index])

            from ...types import MultiModalInput

            return CacheEntry(
                query=MultiModalInput(text=arrow_table["query_text"][index].as_py()),
                context=context_dict,
                embedding=embedding,
                result=result_obj,
                timestamp=arrow_table["timestamp"][index].as_py(),
                similarity=similarity,
                metadata=metadata_obj,
            )
        except Exception as e:
            logger.error(
                "arrow_conversion_failed",
                index=index,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            return None

    def to_arrow(self):
        """Convert semantic table to Arrow table.

        Returns:
            PyArrow Table with semantic cache entries.
        """
        return self.semantic_table.to_arrow()

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage-specific statistics.

        Returns:
            Dictionary with storage metrics and configuration.
        """
        if not self.metrics:
            return {}

        search_latencies = getattr(self.metrics, "storage_search_latencies_ms", [])
        add_latencies = getattr(self.metrics, "storage_add_latencies_ms", [])

        avg_search = (
            sum(search_latencies) / len(search_latencies) if search_latencies else 0
        )
        avg_add = sum(add_latencies) / len(add_latencies) if add_latencies else 0

        return {
            "total_entries": self.count(),
            "exact_entries": self.exact_table.count_rows(),
            "semantic_entries": self.semantic_table.count_rows(),
            "total_searches": getattr(self.metrics, "storage_searches", 0),
            "total_adds": getattr(self.metrics, "storage_adds", 0),
            "avg_search_latency_ms": round(avg_search, 2),
            "avg_add_latency_ms": round(avg_add, 2),
            "search_errors": getattr(self.metrics, "storage_search_errors", 0),
            "add_errors": getattr(self.metrics, "storage_add_errors", 0),
            "index_created": self._index_created,
            "encryption_enabled": self.serializer.encryptor is not None,
            "compression_enabled": self.serializer.compressor is not None,
            "compression_algorithm": (
                self.serializer.compressor.algorithm
                if self.serializer.compressor
                else None
            ),
        }
