"""Indexing operations for LanceDB backend."""

from __future__ import annotations

import time

from ...utils.logging import get_logger

logger = get_logger(__name__)


class IndexingMixin:
    """Mixin providing index operations for LanceDB backend."""

    def has_index(self) -> bool:
        """Check if index exists on semantic table.

        Returns:
            True if index exists, False otherwise.
        """
        return self._index_created

    def create_index(self, num_partitions: int, num_sub_vectors: int) -> None:
        """Create IVF-PQ index on semantic table.

        Args:
            num_partitions: Number of IVF partitions.
            num_sub_vectors: Number of PQ sub-vectors for compression.
        """
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

    def maybe_auto_create_index(self, threshold: int, num_partitions: int) -> None:
        """Create index if threshold reached on semantic table.

        Args:
            threshold: Minimum number of entries to trigger index creation.
            num_partitions: Number of IVF partitions for the index.
        """
        if self._index_created:
            return

        count = self.semantic_table.count_rows()
        if count >= threshold:
            logger.info("auto_creating_index", count=count, threshold=threshold)
            num_sub_vectors = max(1, self.embedding_dim // 4)
            self.create_index(num_partitions, num_sub_vectors)
