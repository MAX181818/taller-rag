"""Contratos del recuperador con motores y codificador simulados."""
from copy import deepcopy

import pytest
import requests
from pymilvus.exceptions import MilvusException

from app import retrieval
from app.embeddings import EncoderMismatch


def record(chunk_id, document_id=None, score=1.0):
    return dict(chunk_id=chunk_id, document_id=document_id or chunk_id,
                provenance_ids=[document_id or chunk_id, "origen"], score=score,
                text="Texto", title="Título", url="https://example.org",
                collection_name="archivo", source="prueba", language="es",
                created_ts=1, start_char=0, end_char=5, n_tokens=1, chunker="tok256")


@pytest.fixture(autouse=True)
def fake_dependencies(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No se permite acceso de red")

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(retrieval.milvus_store, "get_client", lambda: "cliente")
    monkeypatch.setattr(retrieval.milvus_store, "search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval.solr_store, "search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval, "embed", lambda *args, **kwargs: [[0.1, 0.2]])
    monkeypatch.setattr(retrieval, "fingerprint", lambda encoder: {"model": encoder})


def test_rrf_merges_by_chunk_id():
    lists = [[record("a"), record("b")], [record("b"), record("c")]]
    original = deepcopy(lists)
    rows = retrieval.rrf(lists)
    assert [row["chunk_id"] for row in rows] == ["b", "a", "c"]
    assert [row["score"] for row in rows] == pytest.approx(
        [1 / 61 + 1 / 62, 1 / 61, 1 / 62])
    assert lists == original


def test_rrf_ties_use_best_rank_then_chunk_id():
    rows = retrieval.rrf([[record("z"), record("a")],
                          [record("a"), record("z")]], k_rrf=10)
    assert [row["chunk_id"] for row in rows] == ["a", "z"]
    assert rows[0]["score"] == pytest.approx(1 / 11 + 1 / 12)
    assert retrieval.rrf([[], []]) == []
    rows = retrieval.rrf([[record("z"), record("a"), record("b")],
                          [record("b"), record("a"), record("z")]], k_rrf=0)
    assert [row["chunk_id"] for row in rows] == ["b", "z", "a"]


@pytest.mark.parametrize("backend", ["milvus", "solr", "hybrid"])
@pytest.mark.parametrize("k", [2, 30])
def test_one_per_doc_keeps_best_chunk(monkeypatch, backend, k):
    rows = [record("a", "doc1", 9), record("b", "doc1", 8),
            record("c", "doc2", 7), record("d", "doc3", 6)]
    calls = []

    def solr(query, limit, chunker, filters=None):
        calls.append(limit)
        return rows[:limit]

    def milvus(client, name, qvec, limit, **kwargs):
        calls.append(limit)
        return rows[:limit]

    monkeypatch.setattr(retrieval.solr_store, "search", solr)
    monkeypatch.setattr(retrieval.milvus_store, "search", milvus)
    result = retrieval.retrieve("consulta", backend, k=k, one_per_doc=True)
    assert [row["chunk_id"] for row in result] == ["a", "c", "d"][:k]
    expected = min(k * 4, 100)
    if backend == "hybrid":
        expected = max(20, expected)
    assert calls == [expected] * (2 if backend == "hybrid" else 1)
    assert [row["rank"] for row in result] == list(range(1, len(result) + 1))


@pytest.mark.parametrize("backend", ["solr", "milvus"])
@pytest.mark.parametrize("error", [ConnectionError("caído"),
                                   requests.RequestException("servidor"),
                                   MilvusException(message="servidor")])
@pytest.mark.parametrize("hybrid", [False, True])
def test_backend_error_raises_retriever_unavailable(monkeypatch, backend, error, hybrid):
    def fail(*args, **kwargs):
        raise error

    store = retrieval.solr_store if backend == "solr" else retrieval.milvus_store
    monkeypatch.setattr(store, "search", fail)
    with pytest.raises(retrieval.RetrieverUnavailable) as caught:
        retrieval.retrieve("consulta", "hybrid" if hybrid else backend)
    assert caught.value.backend == backend
    assert backend in str(caught.value)
    assert "no está disponible" in str(caught.value)
    assert caught.value.__cause__ is error


def test_milvus_client_connection_error_is_wrapped(monkeypatch):
    def fail():
        raise ConnectionError("caído")

    monkeypatch.setattr(retrieval.milvus_store, "get_client", fail)
    with pytest.raises(retrieval.RetrieverUnavailable) as caught:
        retrieval.retrieve("consulta")
    assert caught.value.backend == "milvus"


@pytest.mark.parametrize("backend", ["milvus", "solr", "hybrid"])
def test_ranks_are_1_based_and_contiguous(monkeypatch, backend):
    rows = [record("a", score=5), record("b", score=3), record("c", score=2)]
    original = deepcopy(rows)
    calls = []
    filters = {"language": "es"}

    def solr(query, k, chunker, filters=None):
        calls.append((query, k, chunker, filters))
        return rows

    def milvus(client, name, qvec, k, filters=None, fingerprint=None):
        calls.append(("consulta", k, "tok512", filters))
        return rows

    monkeypatch.setattr(retrieval.solr_store, "search", solr)
    monkeypatch.setattr(retrieval.milvus_store, "search", milvus)
    result = retrieval.retrieve("consulta", backend, k=2, chunker="tok512", filters=filters)
    assert [row["rank"] for row in result] == [1, 2]
    assert [row["backend"] for row in result] == [backend, backend]
    assert [row["chunk_id"] for row in result] == ["a", "b"]
    for row, source in zip(result, rows):
        assert {key: row[key] for key in source if key != "score"} == {
            key: value for key, value in source.items() if key != "score"}
    assert result[0]["score"] == pytest.approx(2 / 61 if backend == "hybrid" else 5)
    assert calls == [("consulta", 20 if backend == "hybrid" else 2, "tok512", filters)] * (
        2 if backend == "hybrid" else 1)
    assert rows == original


def test_milvus_uses_query_prefix_kind_and_fingerprint(monkeypatch):
    calls = []
    stamp = {"model": "minilm", "revision": "prueba"}

    def embed(texts, kind, encoder):
        calls.append((texts, kind, encoder))
        return [[0.5, 0.7]]

    def fingerprint(encoder):
        assert encoder == "minilm"
        return stamp

    def search(client, name, qvec, k, filters=None, fingerprint=None):
        assert (client, name, qvec, k, filters, fingerprint) == (
            "cliente", "c_minilm_tok512", [0.5, 0.7], 5, {"language": "es"}, stamp)
        return []

    monkeypatch.setattr(retrieval, "embed", embed)
    monkeypatch.setattr(retrieval, "fingerprint", fingerprint)
    monkeypatch.setattr(retrieval.milvus_store, "search", search)
    assert retrieval.retrieve("consulta", chunker="tok512", encoder="minilm",
                              filters={"language": "es"}) == []
    assert calls == [(["consulta"], "query", "minilm")]


@pytest.mark.parametrize("backend", ["milvus", "hybrid"])
def test_encoder_mismatch_propagates(monkeypatch, backend):
    error = EncoderMismatch("Huella distinta")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(retrieval.milvus_store, "search", fail)
    with pytest.raises(EncoderMismatch) as caught:
        retrieval.retrieve("consulta", backend)
    assert caught.value is error


def test_solr_does_not_encode(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Solr no requiere vectores")

    monkeypatch.setattr(retrieval, "embed", forbidden)
    monkeypatch.setattr(retrieval, "fingerprint", forbidden)
    monkeypatch.setattr(retrieval.milvus_store, "get_client", forbidden)
    assert retrieval.retrieve("consulta", "solr") == []


def test_invalid_backend_and_nonpositive_k():
    with pytest.raises(ValueError, match="Backend"):
        retrieval.retrieve("consulta", "otro")
    for backend in ("milvus", "solr", "hybrid"):
        assert retrieval.retrieve("consulta", backend, k=0) == []
        assert retrieval.retrieve("consulta", backend, k=-1) == []
