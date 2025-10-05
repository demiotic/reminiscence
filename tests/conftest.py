from memora import CacheConfig, SemanticCache
import pytest


@pytest.fixture
def cache_config():
    """Config básico para tests."""
    return CacheConfig.for_development()


@pytest.fixture
def cache(cache_config):
    """Cache instance para tests."""
    return SemanticCache(cache_config)


@pytest.fixture
def sample_context():
    """Contexto de ejemplo."""
    return {"tools": ["search"]}


@pytest.fixture
def fake_llm():
    """LLM mock que retorna respuestas predecibles."""

    def _llm(query: str, context: dict) -> str:
        return f"Response for: {query}"

    return _llm
