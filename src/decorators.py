"""Decoradores para Memora - Cacheo transparente de funciones."""

from functools import wraps
import inspect
import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class MemoraDecorator:
    """
    Decorador que envuelve funciones para usar Memora automáticamente.

    Permite cachear resultados de funciones de forma transparente,
    extrayendo automáticamente contexto de los argumentos.

    Example:
        >>> from memora import Memora, CacheConfig, MemoraDecorator
        >>>
        >>> memora = Memora(CacheConfig.for_development())
        >>> decorator = MemoraDecorator(memora)
        >>>
        >>> @decorator.cached(context={"agent": "sql"})
        >>> def query_db(query: str, db: str, timeout: int = 30):
        ...     # Lógica del agente
        ...     return execute_query(query, db)
        >>>
        >>> # Context final: {"agent": "sql", "db": "prod", "timeout": 30}
        >>> result = query_db("SELECT * FROM sales", db="prod")
    """

    def __init__(self, memora_instance):
        """
        Args:
            memora_instance: Instancia de Memora a usar para cacheo
        """
        self.memora = memora_instance

    def cached(
        self,
        context: Optional[Dict[str, Any]] = None,
        extract_from_args: bool = True,
        query_param: str = "query",
    ):
        """
        Decorador que cachea resultados de funciones.

        Args:
            context: Contexto estático (ej: {"agent": "sql", "version": "v1"})
            extract_from_args: Si True, extrae contexto de los argumentos de la función
            query_param: Nombre del parámetro que contiene el query (default: "query")

        Example:
            >>> @decorator.cached(
            ...     context={"agent": "analyzer", "version": "v2"},
            ...     query_param="prompt"
            ... )
            >>> def analyze(prompt: str, model: str = "gpt-4", temperature: float = 0):
            ...     return llm_call(prompt, model, temperature)
            >>>
            >>> # Context final: {
            >>> #   "agent": "analyzer",
            >>> #   "version": "v2",
            >>> #   "model": "gpt-4",
            >>> #   "temperature": 0
            >>> # }
        """
        base_context = context or {}

        def decorator_func(func: Callable) -> Callable:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            # Validar que existe query_param
            if query_param not in params:
                raise ValueError(
                    f"Parámetro '{query_param}' no encontrado en {func.__name__}. "
                    f"Parámetros disponibles: {params}"
                )

            # Detectar si es async
            is_async = inspect.iscoroutinefunction(func)

            if is_async:

                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    return await self._execute_with_cache_async(
                        func,
                        sig,
                        query_param,
                        base_context,
                        extract_from_args,
                        args,
                        kwargs,
                    )

                return async_wrapper
            else:

                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    return self._execute_with_cache_sync(
                        func,
                        sig,
                        query_param,
                        base_context,
                        extract_from_args,
                        args,
                        kwargs,
                    )

                return sync_wrapper

        return decorator_func

    def _execute_with_cache_sync(
        self,
        func,
        sig,
        query_param,
        base_context,
        extract_from_args,
        args,
        kwargs,
    ):
        """Lógica de ejecución con cache para funciones síncronas."""
        # Bind args a parámetros
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # Extraer query
        query = bound.arguments[query_param]

        # Construir contexto
        runtime_context = {}

        if extract_from_args:
            # Extraer todos los args excepto query_param
            runtime_context = {
                k: v for k, v in bound.arguments.items() if k != query_param
            }

        # Merge con contexto estático (base_context tiene prioridad)
        runtime_context.update(base_context)

        logger.debug(
            f"@cached | func={func.__name__} | "
            f"query='{str(query)[:50]}...' | ctx={runtime_context}"
        )

        # Lookup en caché
        result = self.memora.lookup(query, runtime_context)

        if result.is_hit:
            logger.info(
                f"Cache HIT | func={func.__name__} | similarity={result.similarity:.3f}"
            )
            # result.result YA está deserializado por core.py
            return result.result

        # Cache MISS - ejecutar función
        logger.info(f"Cache MISS | func={func.__name__} | executing...")

        output = func(*args, **kwargs)

        # Guardar en caché
        self.memora.store(
            query=query,
            context=runtime_context,
            result=output,
            metadata={"function": func.__name__},
        )

        return output

    async def _execute_with_cache_async(
        self,
        func,
        sig,
        query_param,
        base_context,
        extract_from_args,
        args,
        kwargs,
    ):
        """Lógica de ejecución con cache para funciones asíncronas."""
        # Bind args a parámetros
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # Extraer query
        query = bound.arguments[query_param]

        # Construir contexto
        runtime_context = {}

        if extract_from_args:
            # Extraer todos los args excepto query_param
            runtime_context = {
                k: v for k, v in bound.arguments.items() if k != query_param
            }

        # Merge con contexto estático (base_context tiene prioridad)
        runtime_context.update(base_context)

        logger.debug(
            f"@cached | func={func.__name__} | "
            f"query='{str(query)[:50]}...' | ctx={runtime_context}"
        )

        # Lookup en caché
        result = self.memora.lookup(query, runtime_context)

        if result.is_hit:
            logger.info(
                f"Cache HIT | func={func.__name__} | similarity={result.similarity:.3f}"
            )
            # result.result YA está deserializado por core.py
            return result.result

        # Cache MISS - ejecutar función
        logger.info(f"Cache MISS | func={func.__name__} | executing...")

        output = await func(*args, **kwargs)

        # Guardar en caché
        self.memora.store(
            query=query,
            context=runtime_context,
            result=output,
            metadata={"function": func.__name__},
        )

        return output


def create_cached_decorator(memora_instance):
    """
    Factory function para crear decorador rápidamente.

    Shortcut conveniente para no tener que instanciar MemoraDecorator.

    Args:
        memora_instance: Instancia de Memora

    Returns:
        Método cached listo para usar como decorador

    Example:
        >>> from memora import Memora, CacheConfig, create_cached_decorator
        >>>
        >>> memora = Memora(CacheConfig.for_development())
        >>> cached = create_cached_decorator(memora)
        >>>
        >>> @cached(context={"agent": "sql"})
        >>> def query_db(query: str, db: str):
        ...     return execute_query(query, db)
    """
    return MemoraDecorator(memora_instance).cached
