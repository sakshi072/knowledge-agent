"""
Singleton-at-lifespan pattern:
the graph is built ONCE at process startup (see main.py's lifespan), not
per-request - rebuilding it per-request would re-load MCP tools and
re-bind the model on every single call, which is both wasteful and would
silently defeat checkpointing (a fresh InMemorySaver every request means
no memory at all).
"""

from typing import Optional

_compiled_graph: Optional[object] = None

def get_graph():
    """Returns None if the graph hasn't been initialized yet (startup
    failed, or this is called before lifespan runs) - callers must check
    for None and raise ServiceUnavailableException, not assume it's ready."""
    return _compiled_graph

def set_graph(graph) -> None:
    global _compiled_graph
    _compiled_graph = graph