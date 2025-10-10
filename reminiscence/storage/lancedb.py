"""LanceDB storage backend with hybrid exact/semantic tables."""

import json
import time
from typing import List, Dict, Any
import lancedb
import pyarrow as pa

from .base import StorageBackend
from .schemas import create_exact_schema, create_semantic_schema
from ..types import CacheEntry
from ..utils.logging import get_logger
from ..utils.fingerprint import create_fingerprint, compute_query_hash
from ..serialization import ResultSerializer
from ..compression import create_compressor

logger = get_logger(__name__)


class LanceDBBackend(StorageBackend):
    """LanceDB implementation with dual exact/semantic tables."""

    _instances: Dict[tuple, "LanceDBBackend"] = {}

    def __new__(cls, config, embedding_dim: int, metrics=None):
        """Create or return existing instance for the same db_uri + embedding_dim."""
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
        """Initialize LanceDB backend with dual tables."""
        if self._initialized:
            logger.debug("storage_backend_already_initialized", db_uri=config.db_uri)
            return

        init_start = time.perf_counter()

        self.config = config
        self.embedding_dim = embedding_dim
        self.metrics = metrics

        logger.debug("connecting_to_lancedb", db_uri=config.db_uri)
        self.db = lancedb.connect(config.db_uri)

        # Use imported schema factories
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
        """Initialize or open a specific table."""
        try:
            table = self.db.open_table(table_name)
            logger.debug("table_opened", name=table_name, rows=table.count_rows())
            return table
        except Exception:
            table = self.db.create_table(table_name, schema=schema, mode="overwrite")
            logger.debug("table_created", name=table_name)
            return table

    @classmethod
    def _clear_instances(cls):
        """Clear all singleton instances (for testing only)."""
        logger.debug("clearing_storage_instances", count=len(cls._instances))
        cls._instances.clear()

    def _generate_id(self, entry: CacheEntry) -> str:
        """Generate unique ID for entry using SHA256 hash."""
        return compute_query_hash(entry.query_text, entry.context)

    def count(self) -> int:
        """Get total entries across both tables."""
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

    def add(self, entries: List[CacheEntry]):
        """Add entries to appropriate tables."""
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

    def search(
        self,
        embedding: List[float],
        context: Dict[str, Any],
        limit: int = 50,
        similarity_threshold: float = 0.85,
        query_mode: str = "semantic",
        query_text: str = None,
    ) -> List[CacheEntry]:
        """Search with mode-based routing."""
        start = time.perf_counter()

        logger.debug(
            "search_start",
            query_mode=query_mode,
            limit=limit,
            similarity_threshold=similarity_threshold,
            has_embedding=embedding is not None,
            has_query_text=query_text is not None,
        )

        if query_mode == "exact":
            results = self._search_exact(query_text, context)
        elif query_mode == "semantic":
            results = self._search_semantic(
                embedding, context, limit, similarity_threshold
            )
        else:
            logger.warning(
                "unexpected_query_mode",
                mode=query_mode,
                expected=["exact", "semantic"],
                fallback="semantic",
                note="Mode should be resolved upstream in lookup/store",
            )
            results = self._search_semantic(
                embedding, context, limit, similarity_threshold
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        if self.metrics:
            if not hasattr(self.metrics, "storage_searches"):
                self.metrics.storage_searches = 0
            self.metrics.storage_searches += 1

            if not hasattr(self.metrics, "storage_search_latencies_ms"):
                from collections import deque

                self.metrics.storage_search_latencies_ms = deque(maxlen=1000)
            self.metrics.storage_search_latencies_ms.append(elapsed_ms)

        logger.debug(
            "search_complete",
            mode=query_mode,
            results_count=len(results),
            latency_ms=round(elapsed_ms, 2),
        )

        return results

    def _search_exact(
        self, query_text: str, context: Dict[str, Any]
    ) -> List[CacheEntry]:
        """Exact match using hash lookup."""
        if not query_text:
            logger.debug("exact_search_skipped_no_query")
            return []

        search_start = time.perf_counter()
        query_hash = compute_query_hash(query_text, context)
        context_hash = create_fingerprint(context)

        logger.debug(
            "exact_search_start",
            query_hash=query_hash[:16],
            context_hash=context_hash[:16],
        )

        try:
            results = (
                self.exact_table.search()
                .where(
                    f"query_hash = '{query_hash}' AND context_hash = '{context_hash}'"
                )
                .limit(1)
                .to_arrow()
            )

            search_ms = (time.perf_counter() - search_start) * 1000

            if len(results) == 0:
                logger.debug("exact_search_miss", latency_ms=round(search_ms, 1))
                return []

            entry = self._arrow_row_to_cache_entry(results, 0, similarity=1.0)
            logger.debug("exact_search_hit", latency_ms=round(search_ms, 1))
            return [entry] if entry else []

        except Exception as e:
            search_ms = (time.perf_counter() - search_start) * 1000
            logger.error(
                "exact_search_failed",
                error=str(e),
                error_type=type(e).__name__,
                latency_ms=round(search_ms, 1),
            )
            if self.metrics:
                if not hasattr(self.metrics, "storage_search_errors"):
                    self.metrics.storage_search_errors = 0
                self.metrics.storage_search_errors += 1
            return []

    def _search_semantic(
        self,
        embedding: List[float],
        context: Dict[str, Any],
        limit: int,
        similarity_threshold: float,
    ) -> List[CacheEntry]:
        """Semantic search with vectorized filtering."""
        import pyarrow.compute as pc

        search_start = time.perf_counter()
        context_hash = (
            create_fingerprint(context) if context else create_fingerprint({})
        )
        where_clause = f"context_hash = '{context_hash}'"

        logger.debug(
            "semantic_search_start",
            context_hash=context_hash[:16],
            limit=limit,
            threshold=similarity_threshold,
        )

        try:
            query = self.semantic_table.search(embedding).metric("cosine").limit(limit)
            query = query.where(where_clause)
            results = query.to_arrow()

            if len(results) == 0:
                search_ms = (time.perf_counter() - search_start) * 1000
                logger.debug(
                    "semantic_search_no_candidates", latency_ms=round(search_ms, 1)
                )
                return []

            distances = results["_distance"]
            similarities = pc.subtract(1.0, distances)

            mask = pc.greater_equal(similarities, similarity_threshold)
            filtered_results = results.filter(mask)

            if len(filtered_results) == 0:
                search_ms = (time.perf_counter() - search_start) * 1000
                logger.debug(
                    "semantic_search_filtered_out",
                    candidates=len(results),
                    latency_ms=round(search_ms, 1),
                )
                return []

            entries = []
            filtered_distances = filtered_results["_distance"]
            for i in range(len(filtered_results)):
                similarity = 1.0 - filtered_distances[i].as_py()
                entry = self._arrow_row_to_cache_entry(filtered_results, i, similarity)
                if entry:
                    entries.append(entry)

            entries.sort(key=lambda x: x.similarity or 0, reverse=True)

            search_ms = (time.perf_counter() - search_start) * 1000
            logger.debug(
                "semantic_search_success",
                candidates=len(results),
                filtered=len(entries),
                top_similarity=round(entries[0].similarity, 3) if entries else 0,
                latency_ms=round(search_ms, 1),
            )
            return entries

        except Exception as e:
            search_ms = (time.perf_counter() - search_start) * 1000
            logger.error(
                "semantic_search_failed",
                error=str(e),
                error_type=type(e).__name__,
                latency_ms=round(search_ms, 1),
                exc_info=True,
            )
            if self.metrics:
                if not hasattr(self.metrics, "storage_search_errors"):
                    self.metrics.storage_search_errors = 0
                self.metrics.storage_search_errors += 1
            return []

    def _arrow_row_to_cache_entry(
        self, arrow_table: pa.Table, index: int, similarity: float
    ) -> CacheEntry:
        """Convert Arrow row to CacheEntry."""
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

            return CacheEntry(
                query_text=arrow_table["query_text"][index].as_py(),
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
        """Convert semantic table to Arrow table."""
        return self.semantic_table.to_arrow()

    def delete_by_id(self, entry_id: str) -> bool:
        """
        Delete a single entry by its unique ID.

        This method provides efficient single-entry deletion by ID across both
        exact and semantic tables. It handles both memory and persistent storage
        modes appropriately.

        Args:
            entry_id: SHA256 hash ID of the entry to delete

        Returns:
            True if entry was found and deleted, False otherwise

        Performance:
            - Memory mode: O(n) table scan with filter
            - Persistent mode: O(1) indexed delete by primary key
        """
        delete_start = time.perf_counter()
        logger.debug("delete_by_id_start", entry_id=entry_id[:16])

        deleted = False

        if self.config.db_uri == "memory://":
            # Memory mode: filter out the target entry and recreate tables
            import pyarrow.compute as pc

            for table, table_name, schema in [
                (self.exact_table, self._exact_table_name, self.exact_schema),
                (self.semantic_table, self._semantic_table_name, self.semantic_schema),
            ]:
                arrow_table = table.to_arrow()
                if len(arrow_table) == 0:
                    continue

                # Create mask: keep all entries except the one to delete
                mask = pc.not_equal(arrow_table["id"], entry_id)
                filtered = arrow_table.filter(mask)

                # Check if anything was actually deleted
                if len(filtered) < len(arrow_table):
                    deleted = True
                    new_table = self.db.create_table(
                        table_name,
                        data=filtered if len(filtered) > 0 else None,
                        schema=schema if len(filtered) == 0 else None,
                        mode="overwrite",
                    )

                    # Update table references
                    if table_name == self._exact_table_name:
                        self.exact_table = new_table
                    else:
                        self.semantic_table = new_table
                        self.table = new_table

        else:
            # Persistent mode: use SQL-based deletion for efficiency
            filter_expr = f"id = '{entry_id}'"

            # Try deleting from exact table
            try:
                before_exact = self.exact_table.count_rows()
                self.exact_table.delete(filter_expr)
                after_exact = self.exact_table.count_rows()

                if before_exact > after_exact:
                    deleted = True
                    logger.debug("deleted_from_exact_table", entry_id=entry_id[:16])
            except Exception as e:
                logger.debug("exact_table_delete_skipped", error=str(e))

            # Try deleting from semantic table
            try:
                before_semantic = self.semantic_table.count_rows()
                self.semantic_table.delete(filter_expr)
                after_semantic = self.semantic_table.count_rows()

                if before_semantic > after_semantic:
                    deleted = True
                    logger.debug("deleted_from_semantic_table", entry_id=entry_id[:16])
            except Exception as e:
                logger.debug("semantic_table_delete_skipped", error=str(e))

        delete_ms = (time.perf_counter() - delete_start) * 1000

        logger.debug(
            "delete_by_id_complete",
            entry_id=entry_id[:16],
            deleted=deleted,
            latency_ms=round(delete_ms, 1),
        )

        return deleted

    def delete_by_filter(self, filter_expr: str):
        """Delete entries matching filter from both tables."""
        delete_start = time.perf_counter()
        logger.debug("delete_by_filter_start", filter_expr=filter_expr)

        if self.config.db_uri == "memory://":
            raise NotImplementedError("Use delete_by_condition for memory://")
        else:
            deleted_count = 0
            try:
                before = self.exact_table.count_rows()
                self.exact_table.delete(filter_expr)
                after = self.exact_table.count_rows()
                deleted_count += before - after
                logger.debug("exact_table_deleted", count=before - after)
            except Exception as e:
                logger.debug("exact_table_delete_skipped", error=str(e))

            try:
                before = self.semantic_table.count_rows()
                self.semantic_table.delete(filter_expr)
                after = self.semantic_table.count_rows()
                deleted_count += before - after
                logger.debug("semantic_table_deleted", count=before - after)
            except Exception as e:
                logger.debug("semantic_table_delete_skipped", error=str(e))

            try:
                self.exact_table.compact_files()
                self.semantic_table.compact_files()
                logger.debug("tables_compacted")
            except AttributeError:
                pass

            delete_ms = (time.perf_counter() - delete_start) * 1000
            logger.info(
                "delete_by_filter_complete",
                deleted=deleted_count,
                latency_ms=round(delete_ms, 1),
            )

    def delete_by_condition(self, condition_func):
        """Delete by custom condition (for memory mode)."""
        delete_start = time.perf_counter()
        logger.debug("delete_by_condition_start")

        if self.config.db_uri == "memory://":
            exact_arrow = self.exact_table.to_arrow()
            mask_exact = condition_func(exact_arrow)
            filtered_exact = exact_arrow.filter(mask_exact)

            self.exact_table = self.db.create_table(
                self._exact_table_name,
                data=filtered_exact if len(filtered_exact) > 0 else None,
                schema=self.exact_schema if len(filtered_exact) == 0 else None,
                mode="overwrite",
            )

            semantic_arrow = self.semantic_table.to_arrow()
            mask_semantic = condition_func(semantic_arrow)
            filtered_semantic = semantic_arrow.filter(mask_semantic)

            self.semantic_table = self.db.create_table(
                self._semantic_table_name,
                data=filtered_semantic if len(filtered_semantic) > 0 else None,
                schema=self.semantic_schema if len(filtered_semantic) == 0 else None,
                mode="overwrite",
            )
            self.table = self.semantic_table

            delete_ms = (time.perf_counter() - delete_start) * 1000
            logger.info("delete_by_condition_complete", latency_ms=round(delete_ms, 1))
        else:
            raise NotImplementedError("Use delete_by_filter for persistent storage")

    def clear(self):
        """Clear all entries from both tables."""
        clear_start = time.perf_counter()
        before = self.count()

        logger.debug("clear_start", entries=before)

        self.exact_table = self.db.create_table(
            self._exact_table_name,
            schema=self.exact_schema,
            mode="overwrite",
        )
        self.semantic_table = self.db.create_table(
            self._semantic_table_name,
            schema=self.semantic_schema,
            mode="overwrite",
        )
        self.table = self.semantic_table
        self._index_created = False

        clear_ms = (time.perf_counter() - clear_start) * 1000
        logger.info("storage_cleared", deleted=before, latency_ms=round(clear_ms, 1))

    def has_index(self) -> bool:
        """Check if index exists on semantic table."""
        return self._index_created

    def create_index(self, num_partitions: int, num_sub_vectors: int):
        """Create IVF-PQ index on semantic table."""
        index_start = time.perf_counter()

        logger.info(
            "creating_index",
            partitions=num_partitions,
            sub_vectors=num_sub_vectors,
            entries=self.semantic_table.count_rows(),
        )

        self.semantic_table.create_index(
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
        )
        self._index_created = True

        index_ms = (time.perf_counter() - index_start) * 1000
        logger.info("index_created", latency_ms=round(index_ms, 1))

    def maybe_auto_create_index(self, threshold: int, num_partitions: int):
        """Create index if threshold reached on semantic table."""
        if self._index_created:
            return

        count = self.semantic_table.count_rows()
        if count >= threshold:
            logger.info("auto_creating_index", count=count, threshold=threshold)
            num_sub_vectors = max(1, self.embedding_dim // 4)
            self.create_index(num_partitions, num_sub_vectors)

    def get_storage_stats(self) -> dict:
        """Get storage-specific statistics."""
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
