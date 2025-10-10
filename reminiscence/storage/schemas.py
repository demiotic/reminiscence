"""Arrow schema definitions for LanceDB tables."""

import pyarrow as pa


def create_exact_schema() -> pa.Schema:
    """Schema without embeddings for exact matching."""
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("query_text", pa.string()),
            pa.field("query_hash", pa.string()),
            pa.field("context", pa.string()),
            pa.field("context_hash", pa.string()),
            pa.field("result", pa.string()),
            pa.field("result_type", pa.string()),
            pa.field("timestamp", pa.float64()),
            pa.field("metadata", pa.string()),
        ]
    )


def create_semantic_schema(embedding_dim: int) -> pa.Schema:
    """Schema with embeddings for semantic search."""
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("query_text", pa.string()),
            pa.field("context", pa.string()),
            pa.field("context_hash", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), embedding_dim)),
            pa.field("result", pa.string()),
            pa.field("result_type", pa.string()),
            pa.field("timestamp", pa.float64()),
            pa.field("metadata", pa.string()),
        ]
    )
