"""Operaciones con embeddings."""

import numpy as np


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calcula similitud coseno entre dos vectores normalizados.

    Para vectores L2-normalizados, esto es equivalente a dot product.

    Args:
        vec1: Vector normalizado
        vec2: Vector normalizado

    Returns:
        Similitud coseno en rango [0, 1]

    Example:
        >>> v1 = np.array([1.0, 0.0, 0.0])
        >>> v2 = np.array([0.9, 0.1, 0.0])
        >>> cosine_similarity(v1, v2)
        0.9
    """
    return float(np.dot(vec1, vec2))
