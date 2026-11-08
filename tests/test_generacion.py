from copy import deepcopy
import hashlib
from pathlib import Path
import pytest
from app import prompts
from app.citations import invalid_citations, is_abstention, parse_citations
from datetime import datetime, timezone
import json
import re
import requests
from app import config, generation, rag, runlog
from app.prompts import ABSTENTION
from app.retrieval import RetrieverUnavailable
from datetime import datetime
import socket
import subprocess
import sys
from types import SimpleNamespace
from scripts import run_generation as runner
from scripts import sysmon


# --- prompts ---

def chunk(text="Texto", **fields):
    return dict(text=text, source="noticias", collection_name="archivo",
                title="Título", created_ts=1704153600, **fields)


def test_context_respects_budget_with_whole_chunks():
    chunks = [chunk(char * 2500) for char in "abc"]
    original = deepcopy(chunks)
    context, used = prompts.build_context(chunks, 6000)
    assert len(context) <= 6000
    assert len(used) == 2
    assert "a" * 2500 in context and "b" * 2500 in context
    assert "c" * 2500 not in context and "[3]" not in context
    assert chunks == original


def test_citation_numbers_match_used_evidence():
    chunks = [chunk("x" * 700, rank=1), chunk("Segundo", rank=2),
              chunk("Tercero", rank=3)]
    context, used = prompts.build_context(chunks, 250)
    assert [row["n"] for row in used] == [1, 2]
    assert [row["text"] for row in used] == ["Segundo", "Tercero"]
    assert context == (
        "[1] (fuente: noticias | colección: archivo | fecha: 2024-01-02 | título: Título)\nSegundo\n\n"
        "[2] (fuente: noticias | colección: archivo | fecha: 2024-01-02 | título: Título)\nTercero")


def test_context_uses_rank_and_counts_separators():
    chunks = [chunk("B", rank=2), chunk("A", rank=1)]
    context, used = prompts.build_context(chunks, 165)
    assert [row["text"] for row in used] == ["A"]
    assert context.endswith("\nA")
    full, used = prompts.build_context(chunks, 166)
    assert len(full) == 166 and [row["text"] for row in used] == ["A", "B"]


@pytest.mark.parametrize("budget", [0, -1, 1])
def test_context_skips_empty_or_unusable_chunks(budget):
    assert prompts.build_context([{}, {"text": None}, {"text": "  "}, chunk()], budget) == ("", [])


def test_context_formats_unknown_date_and_title():
    context, used = prompts.build_context([
        {"text": "Prueba", "source": "bluesky", "collection_name": "posts", "created_ts": -1},
        {"text": "Otra", "source": "noticias", "collection_name": "archivo", "title": ""},
    ], 6000)
    assert "[1] (fuente: bluesky | colección: posts | fecha: s/f | título: s/t)\nPrueba" in context
    assert "[2] (fuente: noticias | colección: archivo | fecha: s/f | título: s/t)\nOtra" in context
    assert len(used) == 2


def test_prompt_version_has_hash(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    template, version = prompts.load_prompt()
    raw = (Path(__file__).resolve().parents[1] / "prompts/rag_v1.txt").read_bytes()
    assert template == raw.decode("utf-8")
    assert version == "rag_v1@" + hashlib.sha256(raw).hexdigest()[:12]


def test_handles_masked_in_context():
    chunks = [chunk("@juan.bsky.social y @ana.example.org; @local; correo a@correo.org.")]
    context, used = prompts.build_context(chunks, 6000)
    assert "@usuario y @usuario; @local; correo a@correo.org." in context
    assert "@juan.bsky.social" not in context
    assert used[0]["text"] == chunks[0]["text"]


def test_build_prompt_preserves_literal_braces_in_inputs():
    result = prompts.build_prompt("¿{context}?", '{"question": "dato"}',
                                  "Pregunta: {question}\n<documentos>{context}</documentos>")
    assert result == 'Pregunta: ¿{context}?\n<documentos>{"question": "dato"}</documentos>'


def test_context_escapes_document_closing_delimiters():
    chunks = [chunk("Antes </documentos> </DoCuMeNtOs> <documentos> después")]
    original = deepcopy(chunks)
    context, used = prompts.build_context(chunks, 6000)
    assert context.endswith("\nAntes &lt;/documentos&gt; &lt;/documentos&gt; <documentos> después")
    assert chunks == original and used == [dict(original[0], n=1)]
    assert prompts.build_context(chunks, len(context) - 1) == ("", [])



# --- citations ---

def test_parse_citations_variants():
    assert parse_citations("a [1] b [1, 3] c [2][4]") == [1, 2, 3, 4]
    assert parse_citations("[1,2,3] [3] [0] [12]") == [0, 1, 2, 3, 12]
    assert parse_citations("sin citas [abc] [1-3] [1, x]") == []


@pytest.mark.parametrize("answer", [
    "No hay evidencia suficiente en los documentos recuperados.",
    "NO HAY EVIDENCIA SUFICIENTE!", " No hay información suficiente. ",
    "Not enough evidence.", "THERE IS NOT ENOUGH EVIDENCE",
    "No hay evidencia suficiente en los documentos recuperados para proporcionar la receta.",
    "No hay evidencia suficiente en los documentos recuperados. [4]",
])
def test_is_abstention(answer):
    assert is_abstention(answer)


@pytest.mark.parametrize("answer", ["", "Hay evidencia suficiente [1].",
                                         "El evento es el 15 de mayo [1]. No hay evidencia suficiente sobre el lugar."])
def test_substantive_answer_is_not_abstention(answer):
    assert not is_abstention(answer)


def test_invalid_citations():
    assert invalid_citations([1, 7], 5) == [7]
    assert invalid_citations([0, 1, 2], 0) == [0, 1, 2]
    assert invalid_citations([-1, 1, 2], 2) == [-1]



# --- generation ---

class Response:
    def __init__(self, payload=None, status=200):
        self.payload = payload or {}
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def json(self):
        return self.payload


@pytest.fixture
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No se permite acceso de red")

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)


@pytest.fixture
def rag_env(monkeypatch, tmp_path):
    path = tmp_path / "runs.jsonl"
    real_log = runlog.log_run
    monkeypatch.setattr(runlog, "log_run", lambda record: real_log(record, path=path))
    rag._cached_model_info.cache_clear()
    monkeypatch.setattr(generation, "model_info", lambda model: {"family": "prueba"})

    def forbidden(*args, **kwargs):
        raise AssertionError("La dependencia debe simularse explícitamente")

    monkeypatch.setattr(rag, "retrieve", forbidden)
    monkeypatch.setattr(generation, "generate", forbidden)
    yield path
    rag._cached_model_info.cache_clear()


def evidence(text="Evidencia"):
    return dict(text=text, source="noticias", collection_name="archivo", title="Nota", rank=1)


def generated(answer="Respuesta [1]. Cita fuera de rango [7]."):
    return dict(answer=answer, prompt_tokens=12, completion_tokens=6,
                load_s=0.1, prompt_eval_s=0.2, eval_s=0.3, total_s=0.6, tokens_per_s=20.0)


@pytest.mark.usefixtures('no_network')
def test_generate_maps_404_to_model_not_available(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **kw: Response(status=404))
    with pytest.raises(generation.ModelNotAvailable) as caught:
        generation.generate("ausente", "Pregunta")
    assert caught.value.model == "ausente"
    assert "ausente" in str(caught.value) and "no está disponible" in str(caught.value)


@pytest.mark.usefixtures('no_network')
def test_list_models_404_is_generation_error(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **kw: Response(status=404))
    monkeypatch.setattr(requests, "post", lambda *a, **kw: Response(status=404))
    for request in (generation.list_models, lambda: generation.generate("", "Pregunta")):
        with pytest.raises(generation.GenerationError, match="404") as caught:
            request()
        assert type(caught.value) is generation.GenerationError


@pytest.mark.usefixtures('no_network')
@pytest.mark.parametrize("error", [requests.ConnectionError("caído"), requests.Timeout("agotado")])
def test_generate_connection_error_is_generation_error(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(requests, "post", fail)
    with pytest.raises(generation.GenerationError, match="Ollama") as caught:
        generation.generate("modelo", "Pregunta")
    assert caught.value.__cause__ is error


@pytest.mark.usefixtures('no_network')
def test_generate_http_error_is_generation_error(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **kw: Response(status=503))
    with pytest.raises(generation.GenerationError, match="503"):
        generation.generate("modelo", "Pregunta")


@pytest.mark.usefixtures('no_network')
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


@pytest.mark.usefixtures('no_network')
def test_generate_forwards_options(monkeypatch):
    def post(url, *, json, timeout):
        assert json["options"] == dict(temperature=0.7, seed=9, num_ctx=2048, num_predict=40)
        assert json["keep_alive"] == "2m" and timeout == 7
        return Response({"response": "Sí"})

    monkeypatch.setattr(requests, "post", post)
    assert generation.generate("modelo", "p", temperature=0.7, seed=9, num_ctx=2048,
                               num_predict=40, keep_alive="2m", timeout=7)["answer"] == "Sí"


@pytest.mark.usefixtures('no_network')
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


@pytest.mark.usefixtures('no_network')
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


@pytest.mark.usefixtures('no_network')
@pytest.mark.parametrize("chunks", [[], [{"text": " "}], [evidence("x" * 7000)]])
def test_answer_without_evidence_skips_llm(monkeypatch, rag_env, chunks):
    result = rag.answer("consulta", evidence=chunks)
    assert result["answer"] == ABSTENTION and result["abstained"] is True
    assert result["evidence"] == [] and result["citations"] == []
    assert result["invalid_citations"] == []
    assert result["timings"]["generation_ms"] == result["timings"]["retrieval_ms"] == 0
    assert result["context_sha256"] == hashlib.sha256(b"").hexdigest()


@pytest.mark.usefixtures('no_network')
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


@pytest.mark.usefixtures('no_network')
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


@pytest.mark.usefixtures('no_network')
def test_answer_logs_run(rag_env):
    result = rag.answer("consulta con ñ", evidence=[])
    records = [json.loads(line) for line in rag_env.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    record = records[0]
    timestamp = datetime.fromisoformat(record.pop("timestamp"))
    assert timestamp.utcoffset().total_seconds() == 0
    assert record == result
    assert "consulta con ñ" in rag_env.read_text(encoding="utf-8")


@pytest.mark.usefixtures('no_network')
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


@pytest.mark.usefixtures('no_network')
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


@pytest.mark.usefixtures('no_network')
def test_model_info_failure_does_not_block_answer(monkeypatch, rag_env):
    def fail(model):
        raise generation.GenerationError("Metadatos no disponibles")

    monkeypatch.setattr(generation, "model_info", fail)
    monkeypatch.setattr(generation, "generate", lambda *a, **kw: generated())
    result = rag.answer("consulta", evidence=[evidence()])
    assert result["model_info"] == {} and result["answer"].startswith("Respuesta")


@pytest.mark.usefixtures('no_network')
def test_model_info_retries_after_failure_and_caches_success(monkeypatch, rag_env):
    calls = []

    def info(model):
        calls.append(model)
        if len(calls) == 1:
            raise generation.GenerationError("Metadatos no disponibles")
        return {"family": "recuperada"}

    monkeypatch.setattr(generation, "model_info", info)
    monkeypatch.setattr(generation, "generate", lambda *a, **kw: generated())
    results = [rag.answer("consulta", model="prueba", evidence=[evidence()])["model_info"]
               for _ in range(3)]
    assert results == [{}, {"family": "recuperada"}, {"family": "recuperada"}]
    assert calls == ["prueba", "prueba"]


@pytest.mark.usefixtures('no_network')
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


@pytest.mark.usefixtures('no_network')
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



# --- run_generation ---

@pytest.fixture
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Dependencia externa sin simular")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.fixture
def experiment(monkeypatch, tmp_path):
    gold = tmp_path / "gold.jsonl"
    questions = [
        dict(qid="Q01", split="test", question="Primera", type="directa"),
        dict(qid="Q02", split="test", question="Segunda", type="sintesis"),
        dict(qid="Q03", split="dev", question="Desarrollo", type="directa"),
    ]
    gold.write_text("".join(json.dumps(q) + "\n" for q in questions), encoding="utf-8")
    state = SimpleNamespace(retrieved=[], answered=[], events=[], hardware_calls=0,
                            version_calls=0, fail_once=False, mutate=False)

    def retrieve(query, **kwargs):
        state.retrieved.append((query, deepcopy(kwargs)))
        state.events.append(("retrieve", query, kwargs["backend"]))
        return [dict(chunk_id=query + kwargs["backend"], text=query, rank=1)]

    def answer(query, **kwargs):
        state.answered.append((query, deepcopy(kwargs)))
        state.events.append(("answer", kwargs["model"], kwargs["backend"], query))
        if state.fail_once:
            state.fail_once = False
            raise RuntimeError("generacion fallida")
        digest = hashlib.sha256(json.dumps(kwargs["evidence"], sort_keys=True).encode()).hexdigest()
        if state.mutate:
            kwargs["evidence"][0]["text"] = "alterado"
        return dict(answer="Respuesta [1]", model=kwargs["model"], backend=kwargs["backend"],
                    context_sha256=digest, timings={"retrieval_ms": 0}, tokens={"prompt": 3})

    hardware = dict(cpu="CPU de prueba", ram_gb=16, gpu=None, vram_gb=None)
    versions = dict(ollama="0.0", milvus=None, solr=None, python="3.11",
                    packages={name: "1" for name in runner.PACKAGES})

    def collect_hardware():
        state.hardware_calls += 1
        return hardware

    def collect_versions():
        state.version_calls += 1
        return versions

    monkeypatch.setattr(runner, "retrieve", retrieve)
    monkeypatch.setattr(runner, "answer", answer)
    monkeypatch.setattr(runner, "unload", lambda model: state.events.append(("unload", model)))
    monkeypatch.setattr(runner, "list_models", lambda: ["ajeno", "m1", "m2"])
    monkeypatch.setattr(runner, "model_info", lambda model: {"digest": model})
    monkeypatch.setattr(runner, "collect_hardware", collect_hardware)
    monkeypatch.setattr(runner, "collect_versions", collect_versions)
    state.argv = ["--gold", str(gold), "--out-dir", str(tmp_path / "raw"),
                  "--models", "m1", "m2", "--backends", "milvus", "solr",
                  "--runs", "2", "--tag", "ensayo", "--one-per-doc"]
    state.out = tmp_path / "raw"
    state.execute = lambda *extra: runner.run(runner.build_parser().parse_args(state.argv + list(extra)))
    return state


@pytest.mark.usefixtures('offline')
def test_frozen_evidence_order_and_cold(experiment):
    result = experiment.execute()
    records = read_jsonl(Path(result["file"]))
    frozen = read_jsonl(experiment.out / "ensayo_evidence.jsonl")
    assert result["calls"] == 16 and result["errors"] == 0
    assert len(frozen) == 4
    assert len(experiment.retrieved) == 6
    assert [q for q, _ in experiment.retrieved] == ["Primera", "Primera", "Segunda"] * 2
    assert all(kw == dict(backend=kw["backend"], k=5, chunker="tok256", encoder="e5small",
                          filters=None, one_per_doc=True) for _, kw in experiment.retrieved)
    assert [event for event in experiment.events if event[0] == "answer"] == [
        ("answer", model, backend, query) for model in ("m1", "m2") for _ in range(2)
        for backend in ("milvus", "solr") for query in ("Primera", "Segunda")]
    for model in ("m1", "m2"):
        rows = [r for r in records if r["model"] == model]
        assert [r["cold"] for r in rows] == [True] + [False] * 7
        index = experiment.events.index(("answer", model, "milvus", "Primera"))
        other = "m2" if model == "m1" else "m1"
        assert experiment.events[index - 3:index] == [
            ("unload", "ajeno"), ("unload", other), ("unload", model)]
    for evidence in frozen:
        rows = [r for r in records if (r["qid"], r["backend"]) ==
                (evidence["qid"], evidence["backend"])]
        assert len({r["context_sha256"] for r in rows}) == 1
        assert all(r["retrieval_ms_original"] == evidence["retrieval_ms"] for r in rows)
    assert all(kw["evidence"] == next(row["chunks"] for row in frozen
               if row["backend"] == kw["backend"] and row["chunks"][0]["text"] == query)
               for query, kw in experiment.answered)
    assert all(kw["prompt_version"] == "rag_v3" and kw["temperature"] == 0
               for _, kw in experiment.answered)


@pytest.mark.usefixtures('offline')
def test_error_is_flushed_and_does_not_stop(experiment, monkeypatch):
    experiment.fail_once = True
    fake_answer = runner.answer

    def inspect_previous(*args, **kwargs):
        files = [p for p in experiment.out.glob("*.jsonl") if "evidence" not in p.name]
        if experiment.answered:
            assert len(read_jsonl(files[0])) == len(experiment.answered)
        return fake_answer(*args, **kwargs)

    monkeypatch.setattr(runner, "answer", inspect_previous)
    result = experiment.execute()
    records = read_jsonl(Path(result["file"]))
    assert result["calls"] == 16 and result["errors"] == 1
    assert records[0]["error"] == "generacion fallida"
    assert records[0]["cold"] is True and records[1]["cold"] is False
    assert all("error" not in row for row in records[1:])


@pytest.mark.usefixtures('offline')
def test_resume_retries_errors_and_reuses_evidence(experiment):
    experiment.fail_once = True
    first = experiment.execute()
    frozen_bytes = (experiment.out / "ensayo_evidence.jsonl").read_bytes()
    experiment.retrieved.clear()
    experiment.answered.clear()
    second = experiment.execute("--resume")
    assert second["calls"] == 1 and second["errors"] == 0
    assert len(experiment.answered) == 1 and not experiment.retrieved
    rows = read_jsonl(Path(first["file"]))
    assert len(rows) == 17 and rows[-1]["cold"] is True
    assert frozen_bytes == (experiment.out / "ensayo_evidence.jsonl").read_bytes()
    assert experiment.execute("--resume")["calls"] == 0


@pytest.mark.usefixtures('offline')
def test_resume_reads_all_dates_and_only_exact_tag(experiment):
    experiment.execute()
    logs = [p for p in experiment.out.glob("*.jsonl") if "evidence" not in p.name]
    rows = read_jsonl(logs[0])
    logs[0].unlink()
    for name, records in [("ensayo_20260101_0101.jsonl", rows[:8]),
                          ("ensayo_20260102_0101.jsonl", rows[8:15]),
                          ("ensayo_otro_20260101_0101.jsonl", rows[15:])]:
        (experiment.out / name).write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    experiment.answered.clear()
    assert experiment.execute("--resume")["calls"] == 1
    assert experiment.answered[0][1]["model"] == "m2"


@pytest.mark.usefixtures('offline')
def test_resume_retries_exception_with_empty_message(experiment, monkeypatch):
    real_answer = runner.answer

    def fail(*args, **kwargs):
        raise RuntimeError()

    monkeypatch.setattr(runner, "answer", fail)
    assert experiment.execute()["errors"] == 16
    monkeypatch.setattr(runner, "answer", real_answer)
    assert experiment.execute("--resume")["calls"] == 16


@pytest.mark.usefixtures('offline')
def test_resume_after_interrupted_record(experiment):
    result = experiment.execute()
    path = Path(result["file"])
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + '\n{"model":', encoding="utf-8")
    assert experiment.execute("--resume")["calls"] == 1
    assert len(read_jsonl(path)) == 16
    assert experiment.execute("--resume")["calls"] == 0


@pytest.mark.usefixtures('offline')
def test_partial_evidence_is_reused_and_completed(experiment):
    experiment.execute()
    path = experiment.out / "ensayo_evidence.jsonl"
    rows = read_jsonl(path)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows[:-1]) + '{"qid":', encoding="utf-8")
    experiment.retrieved.clear()
    assert experiment.execute("--resume")["calls"] == 0
    assert len(experiment.retrieved) == 2
    assert len(read_jsonl(path)) == 4


@pytest.mark.usefixtures('offline')
@pytest.mark.parametrize("kind", ["calls", "evidence"])
def test_resume_after_interrupted_utf8_character(experiment, kind):
    result = experiment.execute()
    path = (experiment.out / "ensayo_evidence.jsonl") if kind == "evidence" else Path(result["file"])
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(lines[:-1]) + b'{"text":"\xc3')
    assert experiment.execute("--resume")["calls"] == (0 if kind == "evidence" else 1)
    assert len(read_jsonl(path)) == (4 if kind == "evidence" else 16)
    assert experiment.execute("--resume")["calls"] == 0


@pytest.mark.usefixtures('offline')
def test_evidence_remains_frozen_if_consumer_mutates(experiment):
    experiment.mutate = True
    result = experiment.execute()
    records = read_jsonl(Path(result["file"]))
    for qid in ("Q01", "Q02"):
        for backend in ("milvus", "solr"):
            assert len({r["context_sha256"] for r in records
                        if r["qid"] == qid and r["backend"] == backend}) == 1


@pytest.mark.usefixtures('offline')
def test_qids_config_and_call_metadata(experiment):
    result = experiment.execute("--qids", "Q02", "--k", "3", "--temperature", "0.2")
    rows = read_jsonl(Path(result["file"]))
    assert result["calls"] == 8 and {r["qid"] for r in rows} == {"Q02"}
    effective = json.loads((experiment.out / "ensayo_config.json").read_text(encoding="utf-8"))
    assert {"gold", "split", "backends", "models", "runs", "k", "chunker", "encoder",
            "prompt", "tag", "one_per_doc", "qids", "out_dir", "resume", "temperature"} <= effective.keys()
    assert effective["prompt_version"].startswith("rag_v3@")
    assert effective["MAX_CONTEXT_CHARS"] == runner.config.MAX_CONTEXT_CHARS
    assert effective["inference_params"] == dict(temperature=0.2, seed=42, num_ctx=4096,
                                                  num_predict=384, keep_alive="10m")
    assert effective["model_info"] == {"m1": {"digest": "m1"}, "m2": {"digest": "m2"}}
    assert effective["qids_used"] == ["Q02"] and effective["seed"] == 42
    assert datetime.fromisoformat(effective["date"]).utcoffset().total_seconds() == 0
    assert experiment.hardware_calls == experiment.version_calls == 1
    for row in rows:
        assert row["split"] == "test" and row["type"] == "sintesis" and row["tag"] == "ensayo"
        assert set(row["hardware"]) == {"cpu", "ram_gb", "gpu", "vram_gb"}
        assert row["versions"] == effective["versions"] and "tokens" in row
        assert row["retrieval_ms_original"] >= 0
        assert datetime.fromisoformat(row["timestamp"]).utcoffset().total_seconds() == 0


@pytest.mark.usefixtures('offline')
def test_main_prints_summary_and_progress(experiment, capsys, caplog):
    with caplog.at_level("INFO"):
        assert runner.main(experiment.argv) == 0
    summary = json.loads(capsys.readouterr().out)
    assert set(summary) == {"tag", "calls", "errors", "seconds", "file"}
    assert summary["seconds"] >= 0 and summary["tag"] == "ensayo"
    assert any("10/16" in r.message and "%" in r.message and "ETA" in r.message for r in caplog.records)


@pytest.mark.usefixtures('offline')
@pytest.mark.parametrize("extra", [("--runs", "0"), ("--k", "0"),
                                   ("--qids", "Q99"), ("--tag", "../otro")])
def test_invalid_arguments_fail_before_services(experiment, extra):
    with pytest.raises(ValueError):
        experiment.execute(*extra)
    assert experiment.hardware_calls == 0 and not experiment.retrieved


@pytest.mark.usefixtures('offline')
def test_hardware_with_fake_proc_and_gpu(monkeypatch):
    original = Path.read_text

    def read(path, *args, **kwargs):
        if path.as_posix() == "/proc/cpuinfo":
            return "processor : 0\nmodel name : CPU ejemplo\n"
        if path.as_posix() == "/proc/meminfo":
            return "MemTotal: 16777216 kB\n"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)

    def command(args, **kwargs):
        assert args[:6] == ["docker", "compose", "exec", "-T", "ollama", "nvidia-smi"]
        assert kwargs["timeout"] > 0
        return SimpleNamespace(returncode=0, stdout="GPU ejemplo, 8192\n")

    monkeypatch.setattr(subprocess, "run", command)
    assert runner.collect_hardware() == dict(cpu="CPU ejemplo", ram_gb=16, gpu="GPU ejemplo", vram_gb=8)


@pytest.mark.usefixtures('offline')
def test_hardware_and_versions_unavailable_are_null(monkeypatch):
    original = Path.read_text

    def unavailable(*args, **kwargs):
        raise OSError("no disponible")

    def read(path, *args, **kwargs):
        if path.as_posix().startswith("/proc/"):
            return unavailable()
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(subprocess, "run", unavailable)
    monkeypatch.setattr(requests, "get", unavailable)
    monkeypatch.setitem(sys.modules, "pymilvus", SimpleNamespace(MilvusClient=unavailable))
    assert runner.collect_hardware() == dict(cpu=None, ram_gb=None, gpu=None, vram_gb=None)
    versions = runner.collect_versions()
    assert versions["ollama"] is versions["milvus"] is versions["solr"] is None
    assert versions["python"] and set(versions["packages"]) == set(runner.PACKAGES)


@pytest.mark.usefixtures('offline')
def test_versions_with_fake_http_and_milvus(monkeypatch):
    urls, closed = [], []

    def get(url, **kwargs):
        urls.append(url)
        assert kwargs["timeout"] > 0
        payload = {"version": "0.40"} if url.endswith("/api/version") else {"lucene": {"solr-spec-version": "9.10"}}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)

    class Client:
        def __init__(self, *, uri, timeout):
            assert uri == runner.config.MILVUS_URI and timeout > 0

        def get_server_version(self, **kwargs):
            return "2.6"

        def close(self):
            closed.append(True)

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setitem(sys.modules, "pymilvus", SimpleNamespace(MilvusClient=Client))
    versions = runner.collect_versions()
    assert [versions[k] for k in ("ollama", "solr", "milvus")] == ["0.40", "9.10", "2.6"]
    assert urls[1].endswith("/solr/admin/info/system") and closed == []


@pytest.mark.usefixtures('offline')
@pytest.mark.parametrize("text,expected", [
    ("rag-api-1,12.50%,256MiB / 2GiB\nrag-ollama-1,99%,1GiB / 4GiB\nrag-solr-1,2%,1MB / 2GB\n",
     [("rag-api-1", 12.5, 256, 2048), ("rag-ollama-1", 99, 1024, 4096)]),
    ("api,0%,1GB / 2GB\n", [("api", 0, 1e9 / 2**20, 2e9 / 2**20)]),
    ("", []), ("texto incorrecto\n", []),
])
def test_parse_docker_stats(text, expected):
    rows = sysmon.parse_docker_stats(text)
    assert [(r["container"], r["cpu_pct"], r["mem_used_mb"], r["mem_limit_mb"]) for r in rows] == expected


@pytest.mark.usefixtures('offline')
@pytest.mark.parametrize("text,expected", [
    ("1024, 8192, 25\n", dict(gpu_mem_used_mb=1024, gpu_mem_total_mb=8192, gpu_util_pct=25)),
    ("", dict(gpu_mem_used_mb="", gpu_mem_total_mb="", gpu_util_pct="")),
    ("[N/A], [N/A], [N/A]\n", dict(gpu_mem_used_mb="", gpu_mem_total_mb="", gpu_util_pct="")),
])
def test_parse_nvidia_smi(text, expected):
    assert sysmon.parse_nvidia_smi(text) == expected


@pytest.mark.usefixtures('offline')
def test_monitor_failed_commands_and_duration(monkeypatch, tmp_path):
    import csv

    count = []

    def fail(args, **kwargs):
        count.append(args)
        raise OSError("docker ausente")

    clock = [0.0]
    monkeypatch.setattr(subprocess, "run", fail)
    monkeypatch.setattr(sysmon.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(sysmon.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    out = tmp_path / "monitor.csv"
    assert sysmon.main(["--out", str(out), "--interval", "2", "--duration", "3"]) == 0
    with out.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 2 and len(count) == 4 and clock[0] == 3
    assert all(row["timestamp"] and all(value == "" for key, value in row.items()
                                      if key != "timestamp") for row in rows)


@pytest.mark.usefixtures('offline')
def test_monitor_interrupt_and_successful_sample(monkeypatch, tmp_path):
    import csv

    commands = []

    def command(args, **kwargs):
        commands.append(args)
        text = "api,3%,512MiB / 2GiB\n" if args[1] == "stats" else "100, 8000, 20\n"
        return SimpleNamespace(returncode=0, stdout=text)

    def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "run", command)
    monkeypatch.setattr(sysmon.time, "sleep", interrupt)
    out = tmp_path / "monitor.csv"
    assert sysmon.main(["--out", str(out)]) == 0
    with out.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 1 and rows[0]["gpu_util_pct"] == "20.0"
    assert rows[0]["container"] == "api" and rows[0]["mem_used_mb"] == "512.0"
    assert commands[0] == ["docker", "stats", "--no-stream", "--format", "{{.Name}},{{.CPUPerc}},{{.MemUsage}}"]
    assert commands[1][:6] == ["docker", "compose", "exec", "-T", "ollama", "nvidia-smi"]


@pytest.mark.usefixtures('offline')
def test_monitor_nonzero_commands_ignore_stdout(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=1, stdout="api,3%,512MiB / 2GiB\n"))
    rows = sysmon.sample()
    assert len(rows) == 1 and rows[0]["timestamp"]
    assert all(value == "" for key, value in rows[0].items() if key != "timestamp")


@pytest.mark.usefixtures('offline')
def test_frozen_evidence_with_other_params_is_rejected(experiment):
    experiment.execute()
    with pytest.raises(ValueError, match="otros parámetros"):
        experiment.execute("--k", "3")


@pytest.mark.usefixtures('offline')
def test_unload_failure_does_not_abort(experiment, monkeypatch):
    def unload(model):
        raise RuntimeError("modelo no descargado")
    monkeypatch.setattr(runner, "unload", unload)
    result = experiment.execute()
    assert result["calls"] == 16 and result["errors"] == 0


@pytest.mark.usefixtures('offline')
def test_resume_preserves_original_config(experiment):
    experiment.execute()
    original = experiment.out / "ensayo_config.json"
    before = original.read_bytes()
    experiment.execute("--resume")
    assert original.read_bytes() == before
    resumed = list(experiment.out.glob("ensayo_config_resume_????????_????.json"))
    assert len(resumed) == 1
    assert json.loads(resumed[0].read_text(encoding="utf-8"))["resume"] is True
