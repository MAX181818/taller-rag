"""Contratos HTTP con dependencias sustituidas y sin servicios."""
import logging
from pathlib import Path
from threading import Event
from time import perf_counter
from types import SimpleNamespace

import pytest
import requests
from fastapi.testclient import TestClient

from app import api, config, generation, milvus_store, rag, retrieval, solr_store
from app.embeddings import EncoderMismatch


@pytest.fixture(autouse=True)
def no_services(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Acceso externo inesperado")
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(milvus_store, "get_client", forbidden)
    monkeypatch.setattr(milvus_store, "MilvusClient", forbidden)
    monkeypatch.setattr(api, "_health_pending", {}, raising=False)
    monkeypatch.setattr(rag.runlog, "log_run", forbidden)
    monkeypatch.setattr(retrieval, "retrieve", forbidden)
    monkeypatch.setattr(rag, "answer", forbidden)


@pytest.fixture
def client():
    with TestClient(api.app) as instance:
        yield instance


def healthy(monkeypatch):
    monkeypatch.setattr(milvus_store, "MilvusClient", lambda **kwargs: SimpleNamespace(
        list_collections=lambda **kwargs: [], close=lambda: None))
    monkeypatch.setattr(solr_store, "ping", lambda: True)
    monkeypatch.setattr(api.requests, "get", lambda url, timeout: SimpleNamespace(
        raise_for_status=lambda: None))


def test_health_degraded_when_solr_down(client, monkeypatch):
    healthy(monkeypatch)
    monkeypatch.setattr(solr_store, "ping", lambda: False)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "degraded", "deps": {
        "milvus": True, "solr": False, "ollama": True}}


def test_health_ok_and_short_transport_timeouts(client, monkeypatch):
    calls = []
    healthy(monkeypatch)
    def collections(timeout):
        assert 0 < timeout <= 3
        calls.append("milvus")
        return []
    def tags(url, timeout):
        assert url == config.OLLAMA_URL.rstrip("/") + "/api/tags"
        assert 0 < timeout <= 3
        calls.append("ollama")
        return SimpleNamespace(raise_for_status=lambda: None)
    def client_factory(uri, timeout):
        assert uri == config.MILVUS_URI
        assert 0 < timeout <= 3
        return SimpleNamespace(list_collections=collections, close=lambda: calls.append("closed"))
    monkeypatch.setattr(milvus_store, "MilvusClient", client_factory)
    monkeypatch.setattr(api.requests, "get", tags)
    assert client.get("/health").json() == {"status": "ok", "deps": {
        "milvus": True, "solr": True, "ollama": True}}
    assert set(calls) == {"milvus", "ollama", "closed"}


def test_health_bounds_blocked_checks(client, monkeypatch):
    release = Event()
    monkeypatch.setattr(api, "HEALTH_TIMEOUT", 0.03, raising=False)
    monkeypatch.setattr(milvus_store, "MilvusClient", lambda **kw: release.wait())
    monkeypatch.setattr(solr_store, "ping", lambda: release.wait())
    monkeypatch.setattr(api.requests, "get", lambda *a, **kw: release.wait())
    start = perf_counter()
    try:
        response = client.get("/health")
        assert perf_counter() - start < 1
        assert response.status_code == 200
        assert response.json() == {"status": "degraded", "deps": {
            "milvus": False, "solr": False, "ollama": False}}
    finally:
        release.set()


@pytest.mark.parametrize("dependency", ["milvus", "solr", "ollama"])
def test_health_swallows_exceptions(client, monkeypatch, dependency):
    healthy(monkeypatch)
    def fail(*args, **kwargs):
        raise RuntimeError("caído")
    target, name = {"milvus": (milvus_store, "MilvusClient"),
                    "solr": (solr_store, "ping"), "ollama": (api.requests, "get")}[dependency]
    monkeypatch.setattr(target, name, fail)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["deps"][dependency] is False


def ask_error(client, monkeypatch, error):
    def fail(**kwargs):
        raise error
    monkeypatch.setattr(rag, "answer", fail)
    return client.post("/ask", json={"query": "pregunta", "model": "modelo"})


def test_ask_503_when_retriever_down(client, monkeypatch):
    response = ask_error(client, monkeypatch, retrieval.RetrieverUnavailable("solr"))
    assert response.status_code == 503
    assert response.json()["detail"].startswith("El recuperador solr no está disponible:")


def test_ask_503_when_encoder_mismatch(client, monkeypatch):
    response = ask_error(client, monkeypatch, EncoderMismatch("huella distinta"))
    assert response.status_code == 503
    assert response.json() == {"detail": "La colección fue indexada con otro encoder: huella distinta"}


def test_ask_400_when_model_missing(client, monkeypatch):
    response = ask_error(client, monkeypatch, generation.ModelNotAvailable("modelo"))
    assert response.status_code == 400
    assert "docker compose exec ollama ollama pull modelo" in response.json()["detail"]


def test_ask_503_when_ollama_down(client, monkeypatch):
    response = ask_error(client, monkeypatch, generation.GenerationError("timeout"))
    assert response.status_code == 503
    assert response.json() == {"detail": "No se pudo generar la respuesta con modelo: timeout"}


def test_ask_400_when_prompt_missing(client, monkeypatch):
    path = Path(api.__file__).resolve().parents[1] / "prompts" / f"{config.PROMPT_VERSION}.txt"
    response = ask_error(client, monkeypatch, FileNotFoundError(2, "ausente", str(path)))
    assert response.status_code == 400
    assert response.json() == {"detail": f"La versión de prompt {config.PROMPT_VERSION} no existe"}


@pytest.mark.parametrize("route", ["/ask", "/retrieve"])
def test_missing_non_prompt_file_is_not_reported_as_prompt(client, monkeypatch, route):
    def fail(**kwargs):
        raise FileNotFoundError(2, "ausente", "encoder/config.json")
    monkeypatch.setattr(rag, "answer", fail)
    monkeypatch.setattr(retrieval, "retrieve", fail)
    response = client.post(route, json={"query": "pregunta"})
    assert response.status_code == 500
    assert response.json() == {"detail": "No se pudo completar la solicitud: falta un archivo requerido"}


def test_health_reuses_pending_probes_and_refreshes(client, monkeypatch):
    release = Event()
    started = {name: Event() for name in ("milvus", "solr", "ollama")}
    counts = dict.fromkeys(started, 0)
    def block(name):
        counts[name] += 1
        started[name].set()
        release.wait()
        raise ConnectionError("caído")
    monkeypatch.setattr(api, "HEALTH_TIMEOUT", 0.03)
    monkeypatch.setattr(milvus_store, "MilvusClient", lambda **kw: block("milvus"))
    monkeypatch.setattr(solr_store, "ping", lambda: block("solr"))
    monkeypatch.setattr(api.requests, "get", lambda *a, **kw: block("ollama"))
    try:
        for _ in range(3):
            assert client.get("/health").json()["status"] == "degraded"
        assert all(event.is_set() for event in started.values())
        assert counts == {"milvus": 1, "solr": 1, "ollama": 1}
    finally:
        release.set()
        for thread, _ in getattr(api, "_health_pending", {}).values():
            thread.join(timeout=1)
    healthy(monkeypatch)
    assert client.get("/health").json() == {"status": "ok", "deps": {
        "milvus": True, "solr": True, "ollama": True}}


def test_ask_ok_shape(client, monkeypatch):
    record = dict(run_id="run", query="pregunta", answer="respuesta [1]", abstained=False,
                  citations=[1], invalid_citations=[], model="modelo", model_info={},
                  backend="hybrid", chunker="tok256", encoder="minilm", k=2,
                  prompt_version="rag_v3", evidence=[{"text": "evidencia"}],
                  context_sha256="abc", timings={"total_ms": 2}, tokens={}, params={})
    received = {}
    def answer(**kwargs):
        received.update(kwargs)
        return record
    monkeypatch.setattr(rag, "answer", answer)
    payload = dict(query="  pregunta  ", backend="hybrid", k=2, chunker="tok256",
                   encoder="minilm", filters={"language": "es"}, one_per_doc=True,
                   model="modelo", temperature=0.4, prompt_version="rag_v3")
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    assert response.json() == record
    assert received == dict(payload, query="pregunta")


@pytest.mark.parametrize("route", ["/ask", "/retrieve"])
@pytest.mark.parametrize("payload", [{"query": "pregunta", "k": 0}, {"query": "ab"},
                                     {"query": "  ab  "}, {"query": " " * 8}])
def test_validation_rejects_k_0_and_short_query(client, route, payload):
    assert client.post(route, json=payload).status_code == 422


@pytest.mark.parametrize("route", ["/ask", "/retrieve"])
def test_validation_rejects_unknown_filter_key(client, route):
    assert client.post(route, json={"query": "pregunta", "filters": {"source": "web"}}).status_code == 422


@pytest.mark.parametrize("field,value", [
    ("query", "x" * 2001), ("k", 21), ("backend", "otro"), ("encoder", "otro"),
    ("chunker", "../x"), ("chunker", "x" * 33), ("filters", {"language": ""}),
    ("filters", {"language": "x" * 65}), ("filters", {"language": 12}),
    ("model", ""), ("model", "x" * 101), ("temperature", -0.1),
    ("temperature", 1.1), ("prompt_version", "../x"), ("prompt_version", "x" * 33)])
def test_validation_limits(client, field, value):
    assert client.post("/ask", json={"query": "pregunta", field: value}).status_code == 422


def test_retrieve_returns_chunks_and_latency(client, monkeypatch):
    received = {}
    def retrieve(**kwargs):
        received.update(kwargs)
        return [{"chunk_id": "c1", "text": "texto", "score": 0.5}]
    monkeypatch.setattr(retrieval, "retrieve", retrieve)
    response = client.post("/retrieve", json={"query": " pregunta "})
    assert response.status_code == 200
    assert response.json()["chunks"] == [{"chunk_id": "c1", "text": "texto", "score": 0.5}]
    assert isinstance(response.json()["latency_ms"], float)
    assert response.json()["latency_ms"] >= 0
    assert received == dict(query="pregunta", backend="milvus", k=config.DEFAULT_K,
                            chunker=config.CHUNKER, encoder=config.ENCODER,
                            filters=None, one_per_doc=False)


@pytest.mark.parametrize("error,prefix", [
    (retrieval.RetrieverUnavailable("milvus"), "El recuperador milvus no está disponible:"),
    (EncoderMismatch("huella"), "La colección fue indexada con otro encoder:")])
def test_retrieve_translates_errors(client, monkeypatch, error, prefix):
    def fail(**kwargs):
        raise error
    monkeypatch.setattr(retrieval, "retrieve", fail)
    response = client.post("/retrieve", json={"query": "pregunta"})
    assert response.status_code == 503
    assert response.json()["detail"].startswith(prefix)


def test_models_503_when_ollama_down(client, monkeypatch):
    def fail():
        raise generation.GenerationError("caído")
    monkeypatch.setattr(generation, "list_models", fail)
    response = client.get("/models")
    assert response.status_code == 503
    assert response.json()["detail"].startswith("No se pudieron consultar los modelos de Ollama:")


def test_models_ok(client, monkeypatch):
    monkeypatch.setattr(generation, "list_models", lambda: ["modelo:latest"])
    assert client.get("/models").json() == {"models": ["modelo:latest"]}


@pytest.mark.parametrize("route,status", [("/retrieve", 200), ("/ask", 503), ("/ask", 422)])
def test_request_logging_is_bounded(client, monkeypatch, caplog, route, status):
    monkeypatch.setattr(retrieval, "retrieve", lambda **kw: [])
    def fail(**kwargs):
        raise generation.GenerationError("caído")
    monkeypatch.setattr(rag, "answer", fail)
    question = "a" * 81 + "SECRETO"
    with caplog.at_level(logging.INFO, logger="app.api"):
        response = client.post(route, json=dict(query=question, backend="solr",
                                                model="modelo\ninyectado",
                                                k=0 if status == 422 else 3))
    assert response.status_code == status
    records = [r.getMessage() for r in caplog.records if r.name == "app.api"]
    assert len(records) == 1
    line = records[0]
    assert len(line.splitlines()) == 1
    assert "SECRETO" not in line
    assert "backend=" in line and "k=" in line and "modelo=" in line
    assert "latencia_ms=" in line and f"estado={status}" in line
