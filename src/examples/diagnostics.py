import numpy as np
from sentence_transformers import SentenceTransformer


def test_embedding_quality():
    """Diagnóstico: verifica calidad de embeddings multilingües."""
    print("━━━ DIAGNÓSTICO: Embedding Quality ━━━\n")

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    tests = [
        ("Español similar", "Explícame Python", "Qué es Python", 0.75),
        ("Español diferente", "Explícame Python", "Genera código Python", 0.50),
        ("Cross-lingual", "Explícame Python", "What is Python", 0.70),
        ("Inglés similar", "Explain Python", "What is Python", 0.80),
    ]

    for test_name, q1, q2, expected in tests:
        emb1 = model.encode(q1, convert_to_numpy=True, normalize_embeddings=True)
        emb2 = model.encode(q2, convert_to_numpy=True, normalize_embeddings=True)
        sim = float(np.dot(emb1, emb2))

        status = "✓" if sim >= expected else "✗"
        print(f"  {status} [{test_name}] '{q1}' vs '{q2}'")
        print(f"     Similitud: {sim:.4f} (expected: >{expected})")

    print(f"\n  ✅ Threshold recomendado: 0.75-0.80\n")


if __name__ == "__main__":
    test_embedding_quality()
