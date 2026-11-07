"""Generación, orquestación y registro con transporte y búsqueda simulados."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import pytest
import requests

from app import config, generation, rag, runlog
from app.prompts import ABSTENTION
from app.retrieval import RetrieverUnavailable


class Response:
    def __init__(self, payload=None, status=200):
        self.payload = payload or {}
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def json(self):
        return self.payload


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No se permite acceso de red")

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)


@pytest.fixture
def rag_env(monkeypatch, tmp_path):
    path = tmp_path / "runs.jsonl"
    real_log = runlog.log_run
    monkeypatch.setattr(runlog, "log_run", lambda record: real_log(record, path=path))
    rag._model_info.cache_clear()
    monkeypatch.setattr(generation, "model_info", lambda model: {"family": "prueba"})

    def forbidden(*args, **kwargs):
        raise AssertionError("La dependencia debe simularse explícitamente")

    monkeypatch.setattr(rag, "retrieve", forbidden)
    monkeypatch.setattr(generation, "generate", forbidden)
    yield path
    rag._model_info.cache_clear()


def evidence(text="Evidencia"):
    return dict(text=text, source="noticias", collection_name="archivo", title="Nota", rank=1)


def generated(answer="Respuesta [1]. Cita fuera de rango [7]."):
    return dict(answer=answer, prompt_tokens=12, completion_tokens=6,
                load_s=0.1, prompt_eval_s=0.2, eval_s=0.3, total_s=0.6, tokens_per_s=20.0)


def test_generate_maps_404_to_model_not_available(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **kw: Response(status=404))
    with pytest.raises(generation.ModelNotAvailable) as caught:
        generation.generate("ausente", "Pregunta")
    assert caught.value.model == "ausente"
    assert "ausente" in str(caught.value) and "no está disponible" in str(caught.value)


@pytest.mark.parametrize("error", [requests.ConnectionError("caído"), requests.Timeout("agotado")])
def test_generate_connection_error_is_generation_error(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(requests, "post", fail)
    with pytest.raises(generation.GenerationError, match="Ollama") as caught:
        generation.generate("modelo", "Pregunta")
    assert caught.value.__cause__ is error


def test_generate_http_error_is_generation_error(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **kw: Response(status=503))
    with pytest.raises(generation.GenerationError, match="503"):
        generation.generate("modelo", "Pregunta")


@pytest.mark.parametrize("duration,speed", [(2_000_000_000, 50.0), (0, 0.0)])
def test_tokens_per_s(monkeypatch, duration, speed):
    def post(url, *, json, timeout):
        assert url == config.OLLAMA_URL.rstrip("/") + "/api/generate"
        assert json == dict(model="modelo", prompt="Pregunta", stream=False, keep_alive="10m",
                            options=dict(temperature=0.0, seed=42, num_ctx=4096, num_predict=384))
        assert timeout == 300
        return Response(dict(response="Hecho [1]", eval_count=100, eval_duration=duration,
                             prompt_eval_count=50, load_duration=100_000_000,
                             prompt_eval_duration=300_000_000, total_duration=3_000_000_000))

    monkeypatch.setattr(requests, "post", post)
    result = generation.generate("modelo", "Pregunta")
    assert result == dict(answer="Hecho [1]", prompt_tokens=50, completion_tokens=100,
                          load_s=0.1, prompt_eval_s=0.3, eval_s=duration / 1e9,
                          total_s=3.0, tokens_per_s=speed)


def test_generate_forwards_options(monkeypatch):
    def post(url, *, json, timeout):
        assert json["options"] == dict(temperature=0.7, seed=9, num_ctx=2048, num_predict=40)
        assert json["keep_alive"] == "2m" and timeout == 7
        return Response({"response": "Sí"})

    monkeypatch.setattr(requests, "post", post)
    assert generation.generate("modelo", "p", temperature=0.7, seed=9, num_ctx=2048,
                               num_predict=40, keep_alive="2m", timeout=7)["answer"] == "Sí"


def test_models_and_metadata(monkeypatch):
    def get(url, *, timeout):
        assert url == config.OLLAMA_URL.rstrip("/") + "/api/tags" and timeout > 0
        return Response({"models": [{"name": "modelo:latest", "digest": "abc123"}]})

    def post(url, *, json, timeout):
        assert url == config.OLLAMA_URL.rstrip("/") + "/api/show"
        assert json == {"model": "modelo"} and timeout > 0
        return Response({"details": {"parameter_size": "1B", "quantization_level": "Q4_K_M", "family": "qwen2"}})

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(requests, "post", post)
    assert generation.list_models() == ["modelo:latest"]
    assert generation.model_info("modelo") == dict(parameter_size="1B", quantization_level="Q4_K_M",
                                                   digest="abc123", family="qwen2")


def test_unload_sends_empty_prompt_and_zero_keep_alive(monkeypatch):
    calls = []

    def post(url, *, json, timeout):
        calls.append(json)
        return Response()

    monkeypatch.setattr(requests, "post", post)
    generation.unload("modelo")
    assert len(calls) == 1
    assert calls[0]["model"] == "modelo" and calls[0]["prompt"] == ""
    assert calls[0]["keep_alive"] == 0 and calls[0]["stream"] is False


@pytest.mark.parametrize("chunks", [[], [{"text": " "}], [evidence("x" * 7000)]])
def test_answer_without_evidence_skips_llm(monkeypatch, rag_env, chunks):
    result = rag.answer("consulta", evidence=chunks)
    assert result["answer"] == ABSTENTION and result["abstained"] is True
    assert result["evidence"] == [] and result["citations"] == []
    assert result["invalid_citations"] == []
    assert result["timings"]["generation_ms"] == result["timings"]["retrieval_ms"] == 0
    assert result["context_sha256"] == hashlib.sha256(b"").hexdigest()


def test_answer_uses_given_evidence_without_retrieving(monkeypatch, rag_env):
    chunks = [evidence()]

    def generate(model, prompt, **kwargs):
        assert model == "prueba"
        assert "<documentos>" in prompt and "</documentos>" in prompt
        assert "\nEvidencia" in prompt and "consulta" in prompt
        assert kwargs == dict(temperature=0.4, seed=42, num_ctx=4096, num_predict=384)
        return generated()

    monkeypatch.setattr(generation, "generate", generate)
    result = rag.answer("consulta", model="prueba", evidence=chunks, temperature=0.4,
                        backend="solr", k=3, one_per_doc=True, filters={"source": "noticias"})
    assert result["answer"] == "Respuesta [1]. Cita fuera de rango [7]."
    assert result["abstained"] is False
    assert result["citations"] == [1, 7] and result["invalid_citations"] == [7]
    assert result["evidence"] == [dict(chunks[0], n=1)] and "n" not in chunks[0]
    assert result["timings"]["retrieval_ms"] == 0
    assert result["tokens"] == dict(prompt=12, completion=6, tokens_per_s=20.0, load_s=0.1)
    assert result["params"] == dict(temperature=0.4, seed=42, num_ctx=4096, num_predict=384,
                                    max_context_chars=config.MAX_CONTEXT_CHARS, one_per_doc=True,
                                    filters={"source": "noticias"})
    assert result["model_info"] == {"family": "prueba"}
    assert re.fullmatch(r"[0-9a-f]{32}", result["run_id"])
    assert re.fullmatch(rf"{config.PROMPT_VERSION}@[0-9a-f]{{12}}", result["prompt_version"])
    assert all(value >= 0 for value in result["timings"].values())
    context = "[1] (fuente: noticias | colección: archivo | fecha: s/f | título: Nota)\nEvidencia"
    assert result["context_sha256"] == hashlib.sha256(context.encode("utf-8")).hexdigest()
    assert (result["query"], result["backend"], result["k"], result["chunker"], result["encoder"]) == (
        "consulta", "solr", 3, config.CHUNKER, config.ENCODER)


def test_answer_retrieves_with_filters_and_recognizes_abstention(monkeypatch, rag_env):
    def retrieve(query, **kwargs):
        assert query == "consulta"
        assert kwargs == dict(backend="hybrid", k=2, chunker="tok128", encoder="minilm",
                              filters={"source": "bluesky"}, one_per_doc=True)
        return [evidence()]

    monkeypatch.setattr(rag, "retrieve", retrieve)
    monkeypatch.setattr(generation, "generate", lambda *a, **kw: generated("Not enough evidence."))
    result = rag.answer("consulta", backend="hybrid", k=2, chunker="tok128", encoder="minilm",
                        filters={"source": "bluesky"}, one_per_doc=True)
    assert result["abstained"] is True and result["answer"] == "Not enough evidence."


def test_answer_logs_run(rag_env):
    result = rag.answer("consulta con ñ", evidence=[])
    records = [json.loads(line) for line in rag_env.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    record = records[0]
    timestamp = datetime.fromisoformat(record.pop("timestamp"))
    assert timestamp.utcoffset().total_seconds() == 0
    assert record == result
    assert "consulta con ñ" in rag_env.read_text(encoding="utf-8")


@pytest.mark.parametrize("kind", ["retrieval", "missing_model", "generation"])
def test_answer_logs_errors_before_propagating(monkeypatch, rag_env, kind):
    error = {"retrieval": RetrieverUnavailable("solr"),
             "missing_model": generation.ModelNotAvailable("ausente"),
             "generation": generation.GenerationError("Ollama no responde")}[kind]

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(rag, "retrieve", fail if kind == "retrieval" else lambda *a, **kw: [evidence()])
    monkeypatch.setattr(generation, "generate", fail)
    with pytest.raises(type(error)) as caught:
        rag.answer("consulta", backend="solr")
    assert caught.value is error
    records = [json.loads(line) for line in rag_env.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert str(error) in records[0]["error"] and records[0]["query"] == "consulta"


def test_model_info_is_cached_per_model(monkeypatch, rag_env):
    calls = []

    def info(model):
        calls.append(model)
        return {"family": model}

    monkeypatch.setattr(generation, "model_info", info)
    monkeypatch.setattr(generation, "generate", lambda *a, **kw: generated())
    for model in ("a", "a", "b"):
        assert rag.answer("consulta", model=model, evidence=[evidence()])["model_info"] == {"family": model}
    assert calls == ["a", "b"]


def test_model_info_failure_does_not_block_answer(monkeypatch, rag_env):
    def fail(model):
        raise generation.GenerationError("Metadatos no disponibles")

    monkeypatch.setattr(generation, "model_info", fail)
    monkeypatch.setattr(generation, "generate", lambda *a, **kw: generated())
    result = rag.answer("consulta", evidence=[evidence()])
    assert result["model_info"] == {} and result["answer"].startswith("Respuesta")


def test_runlog_appends_utf8_without_mutating_record(tmp_path):
    path = tmp_path / "sub" / "runs.jsonl"
    record = {"query": "¿Qué ocurrió?", "answer": "Información"}
    runlog.log_run(record, path)
    runlog.log_run({"query": "Otra"}, path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 and rows[0]["answer"] == "Información"
    assert "timestamp" not in record
    assert datetime.fromisoformat(rows[0]["timestamp"]).tzinfo is not None
    assert "Información" in path.read_text(encoding="utf-8")


def test_runlog_default_path_is_relative_to_repository(monkeypatch):
    writes = []

    class Sink:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def write(self, text):
            writes.append(text)

    def open_file(path, mode, *, encoding):
        assert path.parent == Path(runlog.__file__).resolve().parents[1] / "reports/runs"
        assert path.name == "api_" + datetime.now(timezone.utc).strftime("%Y%m%d") + ".jsonl"
        assert mode == "a" and encoding == "utf-8"
        return Sink()

    monkeypatch.setattr(Path, "open", open_file)
    monkeypatch.setattr(Path, "mkdir", lambda *a, **kw: None)
    runlog.log_run({"answer": "Prueba"})
    assert len(writes) == 1 and json.loads(writes[0])["answer"] == "Prueba"
