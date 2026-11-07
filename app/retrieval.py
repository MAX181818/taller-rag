"""Recuperación vectorial, léxica y fusión por rango recíproco."""
from typing import Literal

import requests
from pymilvus.exceptions import MilvusException

from app import config, milvus_store, solr_store
from app.embeddings import embed, fingerprint


class RetrieverUnavailable(Exception):
    """El motor solicitado no pudo atender la búsqueda."""

    def __init__(self, backend: str):
        self.backend = backend
        super().__init__(f"El motor {backend} no está disponible.")


def rrf(lists: list[list[dict]], k_rrf: int = 60) -> list[dict]:
    """Combina por chunk_id y desempata por mejor rango y clave."""
    records = {}
    scores = {}
    best_ranks = {}
    for rows in lists:
        for rank, row in enumerate(rows, start=1):
            chunk_id = row["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k_rrf + rank)
            if chunk_id not in records or rank < best_ranks[chunk_id]:
                records[chunk_id] = row
                best_ranks[chunk_id] = rank
    ordered = sorted(records, key=lambda key: (-scores[key], best_ranks[key], key))
    return [dict(records[key], score=scores[key]) for key in ordered]


def retrieve(query: str, backend: Literal["milvus", "solr", "hybrid"] = "milvus",
             k: int = 5, chunker: str = config.CHUNKER,
             encoder: str = config.ENCODER, filters: dict | None = None,
             one_per_doc: bool = False) -> list[dict]:
    """Devuelve los mejores fragmentos con rangos contiguos desde uno."""
    if backend not in ("milvus", "solr", "hybrid"):
        raise ValueError(f"Backend no soportado: {backend}")
    if k <= 0:
        return []
    candidates = min(k * 4, 100) if one_per_doc else k
    if backend == "hybrid":
        candidates = max(20, candidates) if one_per_doc else 20

    lists = []
    if backend in ("milvus", "hybrid"):
        qvec = embed([query], "query", encoder)[0]
        stamp = fingerprint(encoder)
        try:
            client = milvus_store.get_client()
            rows = milvus_store.search(client, config.collection_name(encoder, chunker),
                                       qvec, candidates, filters=filters, fingerprint=stamp)
        except (requests.RequestException, ConnectionError, MilvusException) as error:
            raise RetrieverUnavailable(backend="milvus") from error
        lists.append(rows)
    if backend in ("solr", "hybrid"):
        try:
            rows = solr_store.search(query, candidates, chunker, filters=filters)
        except (requests.RequestException, ConnectionError, MilvusException) as error:
            raise RetrieverUnavailable(backend="solr") from error
        lists.append(rows)

    rows = rrf(lists, k_rrf=60) if backend == "hybrid" else lists[0]
    result = []
    seen = set()
    for row in rows:
        document_id = row["document_id"]
        if one_per_doc and document_id in seen:
            continue
        seen.add(document_id)
        result.append(dict(row, rank=len(result) + 1, backend=backend))
        if len(result) == k:
            break
    return result
