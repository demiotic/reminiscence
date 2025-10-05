"""Utilities for embeddings."""


def cosine_similarity(vec1, vec2) -> float:
    """
    Calculate cosine similarity between two L2-normalized vectors.

    For normalized vectors, cosine_similarity = dot_product.

    Args:
        vec1: List of floats or PyArrow scalar/array
        vec2: List of floats or PyArrow scalar/array

    Returns:
        Cosine similarity in range [-1, 1] (typically [0, 1] if normalized)
    """
    # Convert to Python lists if they are PyArrow objects
    if hasattr(vec1, "as_py"):
        vec1 = vec1.as_py()
    if hasattr(vec2, "as_py"):
        vec2 = vec2.as_py()

    # Dot product
    return float(sum(a * b for a, b in zip(vec1, vec2)))


def cosine_similarity_batch(query_vec, candidate_vecs) -> list[float]:
    """
    Calculate cosine similarity between a query and multiple candidates.

    Args:
        query_vec: Query vector (list of floats)
        candidate_vecs: List of candidate vectors

    Returns:
        List of similarities
    """
    return [cosine_similarity(query_vec, cand) for cand in candidate_vecs]


def euclidean_distance(vec1, vec2) -> float:
    """
    Calculate Euclidean distance between two vectors.

    Args:
        vec1: List of floats or PyArrow scalar/array
        vec2: List of floats or PyArrow scalar/array

    Returns:
        Euclidean distance
    """
    if hasattr(vec1, "as_py"):
        vec1 = vec1.as_py()
    if hasattr(vec2, "as_py"):
        vec2 = vec2.as_py()

    return float(sum((a - b) ** 2 for a, b in zip(vec1, vec2)) ** 0.5)


def normalize_embedding(vec) -> list[float]:
    """
    Normalize an embedding to L2 norm = 1.

    Args:
        vec: List of floats or PyArrow scalar/array

    Returns:
        Normalized vector
    """
    if hasattr(vec, "as_py"):
        vec = vec.as_py()

    norm = sum(x**2 for x in vec) ** 0.5
    if norm == 0:
        return vec

    return [x / norm for x in vec]
