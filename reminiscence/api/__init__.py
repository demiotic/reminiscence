"""gRPC API for Reminiscence semantic cache.

This module provides both server and client implementations for remote
access to Reminiscence over gRPC.

Server (reminiscence.api.server):
    - ReminiscenceServicer: gRPC service implementation
    - create_server: Factory function to create gRPC server
    - serve: Start server in blocking mode

Client (reminiscence.api.client):
    - ReminiscenceClient: Python client for gRPC service

Example (Server):
    >>> from reminiscence import Reminiscence
    >>> from reminiscence.api.server import serve
    >>>
    >>> cache = Reminiscence()
    >>> serve(cache=cache, port=8080)  # Blocks until Ctrl+C

Example (Client):
    >>> from reminiscence.api.client import ReminiscenceClient
    >>> from reminiscence.types import MultiModalInput
    >>>
    >>> with ReminiscenceClient("localhost:8080") as client:
    ...     query = MultiModalInput(text="What is ML?")
    ...     result = client.lookup(query, {"model": "gpt-4"})
    ...     if not result.is_hit:
    ...         client.store(query, {"model": "gpt-4"}, "Machine Learning is...")
"""

from __future__ import annotations

__all__ = ["ReminiscenceClient"]

# Only import client by default (server requires grpc deps)
try:
    from .client import ReminiscenceClient
except ImportError:
    # gRPC dependencies not installed
    pass
