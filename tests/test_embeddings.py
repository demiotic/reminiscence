"""Tests for embeddings module."""

import pytest

from memora.embeddings import create_embedder, SentenceTransformerEmbedder
from memora import CacheConfig


class TestEmbedderFactory:
    """Test embedder factory."""

    def test_create_embedder_default(self):
        """Should create default embedder."""
        config = CacheConfig()
        embedder = create_embedder(config)

        assert isinstance(embedder, SentenceTransformerEmbedder)
        assert embedder.embedding_dim > 0


class TestSentenceTransformerEmbedder:
    """Test SentenceTransformer implementation."""

    def test_embed_basic(self):
        """Should generate embeddings."""
        config = CacheConfig()
        embedder = SentenceTransformerEmbedder(config)

        embedding = embedder.embed("test text")

        assert isinstance(embedding, list)
        assert len(embedding) == embedder.embedding_dim
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_deterministic(self):
        """Same text should produce same embedding."""
        config = CacheConfig()
        embedder = SentenceTransformerEmbedder(config)

        emb1 = embedder.embed("test")
        emb2 = embedder.embed("test")

        assert emb1 == emb2

    def test_embed_normalized(self):
        """Embeddings should be L2-normalized."""
        config = CacheConfig()
        embedder = SentenceTransformerEmbedder(config)

        embedding = embedder.embed("test")

        # Compute L2 norm
        import math

        norm = math.sqrt(sum(x * x for x in embedding))

        # Should be ~1.0 (normalized)
        assert abs(norm - 1.0) < 0.01

    def test_embed_failure(self):
        """Should raise on embed failure."""
        config = CacheConfig()
        embedder = SentenceTransformerEmbedder(config)

        # Mock model to fail
        _ = embedder._model

        def failing_encode(*args, **kwargs):
            raise RuntimeError("Model crashed")

        embedder._model.encode = failing_encode

        with pytest.raises(RuntimeError):
            embedder.embed("test")
