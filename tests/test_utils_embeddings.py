import pytest
import pyarrow as pa
from memora.utils import (
    cosine_similarity,
)


class TestEmbeddings:
    """Tests para utils/embeddings.py."""

    def test_cosine_similarity_identical(self):
        """Vectores idénticos deben tener similitud ~1.0."""
        vec = pa.array([[1.0, 0.0, 0.0]], type=pa.list_(pa.float32(), 3))

        sim = cosine_similarity(vec[0], vec[0])

        assert abs(sim - 1.0) < 0.001

    def test_cosine_similarity_orthogonal(self):
        """Vectores ortogonales deben tener similitud ~0.0."""
        vec1 = pa.array([[1.0, 0.0, 0.0]], type=pa.list_(pa.float32(), 3))
        vec2 = pa.array([[0.0, 1.0, 0.0]], type=pa.list_(pa.float32(), 3))

        sim = cosine_similarity(vec1[0], vec2[0])

        assert abs(sim) < 0.001

    def test_cosine_similarity_similar(self):
        """Test similitud entre vectores parecidos."""
        # Vectores L2-normalizados
        vec1 = pa.array([[1.0, 0.0, 0.0]], type=pa.list_(pa.float32(), 3))
        vec2 = pa.array(
            [[0.9486833, 0.31622777, 0.0]], type=pa.list_(pa.float32(), 3)
        )  # [0.95, 0.32] normalizado

        sim = cosine_similarity(vec1[0], vec2[0])

        # Debería dar ~0.95
        assert sim > 0.9
        assert sim < 1.0
