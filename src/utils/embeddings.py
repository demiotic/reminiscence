"""Utilidades para embeddings."""


def cosine_similarity(vec1, vec2) -> float:
    """
    Calcula similitud coseno entre dos vectores L2-normalizados.

    Para vectores normalizados, cosine_similarity = dot_product.

    Args:
        vec1: Lista de floats o PyArrow scalar/array
        vec2: Lista de floats o PyArrow scalar/array

    Returns:
        Similitud coseno en rango [-1, 1] (típicamente [0, 1] si normalizados)
    """
    # Convertir a listas Python si son objetos PyArrow
    if hasattr(vec1, "as_py"):
        vec1 = vec1.as_py()
    if hasattr(vec2, "as_py"):
        vec2 = vec2.as_py()

    # Producto punto
    return float(sum(a * b for a, b in zip(vec1, vec2)))


def cosine_similarity_batch(query_vec, candidate_vecs) -> list[float]:
    """
    Calcula similitud coseno entre un query y múltiples candidatos.

    Args:
        query_vec: Vector query (lista de floats)
        candidate_vecs: Lista de vectores candidatos

    Returns:
        Lista de similitudes
    """
    return [cosine_similarity(query_vec, cand) for cand in candidate_vecs]


def euclidean_distance(vec1, vec2) -> float:
    """
    Calcula distancia euclidiana entre dos vectores.

    Args:
        vec1: Lista de floats o PyArrow scalar/array
        vec2: Lista de floats o PyArrow scalar/array

    Returns:
        Distancia euclidiana
    """
    if hasattr(vec1, "as_py"):
        vec1 = vec1.as_py()
    if hasattr(vec2, "as_py"):
        vec2 = vec2.as_py()

    return float(sum((a - b) ** 2 for a, b in zip(vec1, vec2)) ** 0.5)


def normalize_embedding(vec) -> list[float]:
    """
    Normaliza un embedding a norma L2 = 1.

    Args:
        vec: Lista de floats o PyArrow scalar/array

    Returns:
        Vector normalizado
    """
    if hasattr(vec, "as_py"):
        vec = vec.as_py()

    norm = sum(x**2 for x in vec) ** 0.5
    if norm == 0:
        return vec

    return [x / norm for x in vec]
