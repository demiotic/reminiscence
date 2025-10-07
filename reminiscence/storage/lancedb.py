"""LanceDB storage backend - Safe multi-format serialization."""

import json
import hashlib
import base64
from typing import List, Dict, Any, Tuple
import lancedb
import pyarrow as pa


try:
    import orjson

    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False

from .base import StorageBackend
from ..types import CacheEntry
from ..utils.logging import get_logger
from ..utils.fingerprint import create_fingerprint

logger = get_logger(__name__)


class LanceDBBackend(StorageBackend):
    """LanceDB implementation with safe multi-format serialization."""

    def __init__(self, config, embedding_dim: int):
        self.config = config
        self.embedding_dim = embedding_dim
        self.db = lancedb.connect(config.db_uri)
        self.schema = self._create_schema()
        self.table = self._init_table()
        self._index_created = False

    def _create_schema(self) -> pa.Schema:
        """Create PyArrow schema with both JSON and hash."""
        return pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("query_text", pa.string()),
                pa.field("context", pa.string()),
                pa.field("context_hash", pa.string()),
                pa.field("embedding", pa.list_(pa.float32(), self.embedding_dim)),
                pa.field("result", pa.string()),
                pa.field("result_type", pa.string()),
                pa.field("timestamp", pa.float64()),
                pa.field("metadata", pa.string()),
            ]
        )

    def _init_table(self):
        """Initialize or open LanceDB table."""
        try:
            table = self.db.open_table(self.config.table_name)
            logger.debug("table_opened", name=self.config.table_name)
            return table
        except Exception:
            table = self.db.create_table(
                self.config.table_name, schema=self.schema, mode="overwrite"
            )
            logger.debug("table_created", name=self.config.table_name)
            return table

    def _serialize_result(self, result: Any) -> Tuple[str, str]:
        """
        Serialize result with optimal format selection.

        Priority:
        1. Arrow IPC for tabular data (Arrow, Pandas, Polars)
        2. orjson for JSON-serializable objects (if available)
        3. Fallback to stdlib json

        Raises:
            TypeError: If result is not serializable

        Returns:
            (serialized_data, type_indicator)
        """

        # PyArrow Table - use IPC (most efficient)
        if hasattr(result, "__class__") and result.__class__.__name__ == "Table":
            try:
                sink = pa.BufferOutputStream()
                with pa.ipc.new_stream(sink, result.schema) as writer:
                    writer.write_table(result)
                buffer = sink.getvalue()
                return base64.b64encode(buffer.to_pybytes()).decode(
                    "utf-8"
                ), "arrow_ipc"
            except Exception as e:
                logger.warning("arrow_serialization_failed", error=str(e))

        # Pandas DataFrame - convert to Arrow IPC
        if hasattr(result, "to_dict") and hasattr(result, "columns"):
            try:
                arrow_table = pa.Table.from_pandas(result)
                sink = pa.BufferOutputStream()
                with pa.ipc.new_stream(sink, arrow_table.schema) as writer:
                    writer.write_table(arrow_table)
                buffer = sink.getvalue()
                return base64.b64encode(buffer.to_pybytes()).decode(
                    "utf-8"
                ), "pandas_arrow"
            except Exception as e:
                logger.warning("pandas_serialization_failed", error=str(e))

        # Polars DataFrame - convert to Arrow IPC
        if hasattr(result, "to_arrow"):
            try:
                arrow_table = result.to_arrow()
                sink = pa.BufferOutputStream()
                with pa.ipc.new_stream(sink, arrow_table.schema) as writer:
                    writer.write_table(arrow_table)
                buffer = sink.getvalue()
                return base64.b64encode(buffer.to_pybytes()).decode(
                    "utf-8"
                ), "polars_arrow"
            except Exception as e:
                logger.warning("polars_serialization_failed", error=str(e))

        # Everything else - JSON only (no repr fallback)
        try:
            if HAS_ORJSON:
                serialized = orjson.dumps(result).decode("utf-8")
                return serialized, "orjson"
            else:
                return json.dumps(result), "json"
        except (TypeError, ValueError) as e:
            # Raise error instead of falling back to repr
            logger.error(
                "json_serialization_failed", error=str(e), type=type(result).__name__
            )
            raise TypeError(
                f"Type {type(result).__name__} is not serializable. "
                f"Supported types: dict, list, str, int, float, bool, None, "
                f"pandas.DataFrame, polars.DataFrame, pyarrow.Table"
            )

    def _deserialize_result(self, data: str, result_type: str) -> Any:
        """
        Deserialize result from string based on type indicator.

        Args:
            data: Serialized data
            result_type: Type indicator from serialization

        Returns:
            Deserialized object
        """

        if result_type == "arrow_ipc":
            buffer = base64.b64decode(data.encode("utf-8"))
            reader = pa.ipc.open_stream(buffer)
            return reader.read_all()

        elif result_type == "pandas_arrow":
            buffer = base64.b64decode(data.encode("utf-8"))
            reader = pa.ipc.open_stream(buffer)
            arrow_table = reader.read_all()
            return arrow_table.to_pandas()

        elif result_type == "polars_arrow":
            buffer = base64.b64decode(data.encode("utf-8"))
            reader = pa.ipc.open_stream(buffer)
            arrow_table = reader.read_all()
            try:
                import polars as pl

                return pl.from_arrow(arrow_table)
            except ImportError:
                logger.warning("polars_not_available", returning="arrow_table")
                return arrow_table

        elif result_type == "orjson":
            if HAS_ORJSON:
                return orjson.loads(data.encode("utf-8"))
            else:
                return json.loads(data)

        elif result_type == "json":
            return json.loads(data)

        elif result_type == "repr":
            # Legacy support for old repr entries
            obj = json.loads(data)
            return obj.get("__repr__")

        else:
            logger.error("unknown_result_type", type=result_type)
            return None

    def count(self) -> int:
        """Get number of entries."""
        return self.table.count_rows()

    def add(self, entries: List[CacheEntry]):
        """Add cache entries with context hash for fast lookup."""
        data = []
        for entry in entries:
            try:
                context_json = json.dumps(entry.context, sort_keys=True)
                context_hash = create_fingerprint(entry.context)
                serialized_result, result_type = self._serialize_result(entry.result)

                data.append(
                    {
                        "id": self._generate_id(entry),
                        "query_text": entry.query_text,
                        "context": context_json,
                        "context_hash": context_hash,
                        "embedding": entry.embedding,
                        "result": serialized_result,
                        "result_type": result_type,
                        "timestamp": entry.timestamp,
                        "metadata": json.dumps(entry.metadata)
                        if entry.metadata
                        else "{}",
                    }
                )
            except TypeError as e:
                # Skip unserializable entries
                logger.error(
                    "entry_skipped_unserializable", error=str(e), query=entry.query_text
                )
                continue
            except Exception as e:
                logger.error(
                    "serialization_failed", error=str(e), entry_query=entry.query_text
                )
                continue

        if data:
            self.table.add(data)

    def search(
        self,
        embedding: List[float],
        context: Dict[str, Any],
        limit: int = 50,
        similarity_threshold: float = 0.85,
    ) -> List[CacheEntry]:
        """
        Hybrid search: hash-based context filter + semantic similarity.

        Uses context_hash for O(1) exact matching, then vector similarity.
        """

        # Generate hash for fast filtering
        context_hash = (
            create_fingerprint(context) if context else create_fingerprint({})
        )

        # Build SQL filter using hash (O(1) lookup)
        where_clause = f"context_hash = '{context_hash}'"

        # Vector search with hash filter
        query = self.table.search(embedding).metric("cosine").limit(limit)
        query = query.where(where_clause)

        try:
            results = query.to_arrow()
        except Exception as e:
            logger.error("search_failed", error=str(e), exc_info=True)
            return []

        # Convert to CacheEntry with deserialization
        entries = []
        for i in range(len(results)):
            try:
                # Calculate similarity from distance
                distance = results["_distance"][i].as_py()
                similarity = 1.0 - distance

                # Filter by threshold
                if similarity < similarity_threshold:
                    continue

                # Deserialize
                context_dict = json.loads(results["context"][i].as_py())
                result_data = results["result"][i].as_py()
                result_type = results["result_type"][i].as_py()
                result_obj = self._deserialize_result(result_data, result_type)

                metadata_str = results["metadata"][i].as_py()
                metadata_obj = (
                    json.loads(metadata_str)
                    if metadata_str and metadata_str != "{}"
                    else None
                )

                entry = CacheEntry(
                    query_text=results["query_text"][i].as_py(),
                    context=context_dict,
                    embedding=results["embedding"][i],
                    result=result_obj,
                    timestamp=results["timestamp"][i].as_py(),
                    similarity=similarity,
                    metadata=metadata_obj,
                )
                entries.append(entry)

            except Exception as e:
                logger.error("deserialization_failed", index=i, error=str(e))
                continue

        # Sort by similarity descending
        entries.sort(key=lambda x: x.similarity or 0, reverse=True)

        return entries

    def to_arrow(self):
        """Convert to Arrow table."""
        return self.table.to_arrow()

    def delete_by_filter(self, filter_expr: str):
        """Delete entries matching filter."""
        if self.config.db_uri == "memory://":
            raise NotImplementedError("Use delete_by_condition for memory://")
        else:
            self.table.delete(filter_expr)
            try:
                self.table.compact_files()
            except AttributeError:
                pass

    def delete_by_condition(self, condition_func):
        """Delete by custom condition (for memory mode)."""
        if self.config.db_uri == "memory://":
            arrow_table = self.to_arrow()
            mask = condition_func(arrow_table)
            filtered = arrow_table.filter(mask)

            self.table = self.db.create_table(
                self.config.table_name,
                data=filtered if len(filtered) > 0 else None,
                schema=self.schema if len(filtered) == 0 else None,
                mode="overwrite",
            )
        else:
            raise NotImplementedError("Use delete_by_filter for persistent storage")

    def clear(self):
        """Clear all entries from cache."""
        self.table = self.db.create_table(
            self.config.table_name,
            schema=self.schema,
            mode="overwrite",
        )
        self._index_created = False

    def has_index(self) -> bool:
        """Check if index exists."""
        return self._index_created

    def create_index(self, num_partitions: int, num_sub_vectors: int):
        """Create IVF-PQ index."""
        logger.info(
            "creating_index",
            partitions=num_partitions,
            sub_vectors=num_sub_vectors,
            entries=self.count(),
        )

        self.table.create_index(
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
        )
        self._index_created = True
        logger.info("index_created")

    def maybe_auto_create_index(self, threshold: int, num_partitions: int):
        """Create index if threshold reached."""
        if self._index_created:
            return

        count = self.count()
        if count >= threshold:
            logger.info("auto_creating_index", count=count, threshold=threshold)
            num_sub_vectors = max(1, self.embedding_dim // 4)
            self.create_index(num_partitions, num_sub_vectors)

    def _generate_id(self, entry: CacheEntry) -> str:
        """Generate unique ID for entry using full SHA256 hash."""
        data = f"{entry.query_text}:{json.dumps(entry.context, sort_keys=True)}:{entry.timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()
