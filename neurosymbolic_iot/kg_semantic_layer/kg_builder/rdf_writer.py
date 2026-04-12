from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from rdflib import Graph

log = logging.getLogger(__name__)


def serialize_graph(g: Graph, out_path: Path, fmt: str = "turtle") -> None:
    """Serialize an rdflib Graph to a file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(out_path), format=fmt)
    log.info("Serialized %d triples to %s (format=%s)", len(g), out_path, fmt)


def load_graph_from_file(path: Path, fmt: str = "turtle") -> Graph:
    """Load a serialized graph from disk."""
    g = Graph()
    g.parse(str(path), format=fmt)
    log.info("Loaded %d triples from %s", len(g), path)
    return g


def push_to_graphdb(
    g: Graph,
    endpoint: str,
    repository: str,
    *,
    named_graph: Optional[str] = None,
    auth: Optional[Dict[str, str]] = None,
) -> bool:
    """
    Push graph triples to a GraphDB instance via SPARQL UPDATE.

    This is optional — fails gracefully with a warning if the endpoint
    is unreachable. Returns True on success, False otherwise.
    """
    try:
        from SPARQLWrapper import POST, SPARQLWrapper
    except ImportError:
        log.warning("SPARQLWrapper not installed; skipping GraphDB push.")
        return False

    update_url = f"{endpoint.rstrip('/')}/repositories/{repository}/statements"
    sparql = SPARQLWrapper(update_url)
    sparql.setMethod(POST)

    if auth:
        sparql.setCredentials(auth.get("username", ""), auth.get("password", ""))

    ttl_data = g.serialize(format="nt")

    graph_clause = f"GRAPH <{named_graph}>" if named_graph else ""
    query = f"INSERT DATA {{ {graph_clause} {{ {ttl_data} }} }}"

    sparql.setQuery(query)
    sparql.setReturnFormat("json")

    try:
        sparql.query()
        log.info(
            "Pushed %d triples to %s (repo=%s, graph=%s)",
            len(g), endpoint, repository, named_graph or "default",
        )
        return True
    except Exception as exc:
        log.warning("GraphDB push failed: %s", exc)
        return False
