"""Tests de calidad de embeddings."""

import pytest
import numpy as np
from sentence_transformers import SentenceTransformer
from memora.utils import cosine_similarity


@pytest.fixture
def model():
    """Modelo para tests de embeddings."""
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def test_embedding_dimensions(model):
    """Test que embeddings tengan dimensiones correctas."""
    embedding = model.encode("Test", normalize_embeddings=True)
    assert embedding.shape == (384,)


def test_embedding_normalized(model):
    """Test que embeddings estén normalizados."""
    embedding = model.encode("Test", normalize_embeddings=True)
    norm = np.linalg.norm(embedding)
    assert abs(norm - 1.0) < 0.001  # Norm ≈ 1


def test_similar_queries_high_similarity(model):
    """Test que queries similares tengan alta similitud."""
    emb1 = model.encode("Explícame Python", normalize_embeddings=True)
    emb2 = model.encode("Qué es Python", normalize_embeddings=True)

    similarity = cosine_similarity(emb1, emb2)
    assert similarity > 0.85  # Alta similitud


def test_different_queries_low_similarity(model):
    """Test que queries diferentes tengan baja similitud."""
    emb1 = model.encode("Explícame Python", normalize_embeddings=True)
    emb2 = model.encode("Receta de paella", normalize_embeddings=True)

    similarity = cosine_similarity(emb1, emb2)
    assert similarity < 0.50  # Baja similitud


def test_cross_lingual_similarity(model):
    """Test que modelo funcione cross-lingual."""
    emb_es = model.encode("Explícame Python", normalize_embeddings=True)
    emb_en = model.encode("What is Python", normalize_embeddings=True)

    similarity = cosine_similarity(emb_es, emb_en)
    assert similarity > 0.80  # Alta similitud cross-lingual


def test_cosine_similarity_function():
    """Test función cosine_similarity."""
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    sim = cosine_similarity(v1, v2)
    assert abs(sim - 1.0) < 0.001  # Vectores idénticos = 1.0

    v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    sim2 = cosine_similarity(v1, v3)
    assert abs(sim2 - 0.0) < 0.001  # Vectores ortogonales = 0.0
