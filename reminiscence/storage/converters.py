"""Arrow table conversion utilities."""

import json
import time
from typing import Optional
import pyarrow as pa

from ..types import CacheEntry
from ..utils.logging import get_logger

logger = get_logger(__name__)


def arrow_row_to_cache_entry(
    arrow_table: pa.Table, index: int, similarity: float, serializer
) -> Optional[CacheEntry]:
    """Convert Arrow row to CacheEntry.

    Args:
        arrow_table: Arrow table containing the row
        index: Row index to convert
        similarity: Similarity score for this entry
        serializer: ResultSerializer instance for deserialization

    Returns:
        CacheEntry or None if conversion fails
    """
    try:
        context_dict = json.loads(arrow_table["context"][index].as_py())
        result_data = arrow_table["result"][index].as_py()
        result_type = arrow_table["result_type"][index].as_py()

        deserialize_start = time.perf_counter()
        result_obj = serializer.deserialize(result_data, result_type)
        deserialize_ms = (time.perf_counter() - deserialize_start) * 1000

        if deserialize_ms > 10:
            logger.debug(
                "deserialization_slow",
                latency_ms=round(deserialize_ms, 1),
                result_type=result_type,
            )

        metadata_str = arrow_table["metadata"][index].as_py()
        metadata_obj = (
            json.loads(metadata_str) if metadata_str and metadata_str != "{}" else None
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
