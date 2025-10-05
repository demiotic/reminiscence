"""Ejemplo básico de uso de Memora."""

import time
from memora import SemanticCache, CacheConfig


def main():
    print("=== Memora - Ejemplo Básico ===\n")

    # Setup del caché
    config = CacheConfig.for_development()
    cache = SemanticCache(config)

    def fake_llm(query: str, context: dict) -> str:
        """Simula llamada al LLM."""
        time.sleep(0.1)
        return f"Respuesta sobre '{query[:40]}...'"

    ctx = {"tools": ["search"]}

    # Primera query (MISS)
    print("1. Primera query:")
    start = time.time()
    r1 = cache.get_or_compute("Explícame Python y sus ventajas", ctx, fake_llm)
    t1 = (time.time() - start) * 1000
    print(f"   {r1}")
    print(f"   ⏱️  {t1:.1f}ms\n")

    # Query similar (HIT esperado)
    print("2. Query similar:")
    start = time.time()
    r2 = cache.get_or_compute("Qué es Python y por qué usarlo", ctx, fake_llm)
    t2 = (time.time() - start) * 1000
    print(f"   {r2}")
    print(f"   ⏱️  {t2:.1f}ms")
    print(f"   🎯 Cache HIT: {r1 == r2} (speedup: {t1 / t2:.1f}x)\n")

    # Contexto diferente (MISS)
    print("3. Contexto diferente:")
    ctx2 = {"tools": ["code_execute"]}
    start = time.time()
    r3 = cache.get_or_compute("Explícame Python", ctx2, fake_llm)
    t3 = (time.time() - start) * 1000
    print(f"   {r3}")
    print(f"   ⏱️  {t3:.1f}ms\n")

    # Stats
    print("━━━ Métricas ━━━")
    for k, v in cache.get_stats().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
