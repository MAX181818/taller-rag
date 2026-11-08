"""API de recuperación y respuestas con errores explícitos."""
import logging
from concurrent.futures import Future
from pathlib import Path
import sys
from threading import Lock, Thread
from time import perf_counter
from typing import Annotated, Literal

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StringConstraints

from app import config, generation, milvus_store, rag, retrieval, solr_store
from app.embeddings import EncoderMismatch


logging.basicConfig(level=config.LOG_LEVEL, stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)
HEALTH_TIMEOUT = 3.0
_health_lock = Lock()
_health_pending = {}
app = FastAPI(title="RAG Solr + Milvus — Taller PLN")

Query = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=2000)]
FilterValue = Annotated[str, StringConstraints(min_length=1, max_length=64)]
Version = Annotated[str, StringConstraints(pattern=r"^[a-z0-9_]{1,32}$")]


class RetrieveRequest(BaseModel):
    query: Query
    backend: Literal["milvus", "solr", "hybrid"] = "milvus"
    k: int = Field(default=config.DEFAULT_K, ge=1, le=20)
    chunker: Version = config.CHUNKER
    encoder: Literal["e5small", "minilm"] = config.ENCODER
    filters: dict[Literal["collection_name", "language"], FilterValue] | None = None
    one_per_doc: bool = False


class AskRequest(RetrieveRequest):
    model: str = Field(default=config.DEFAULT_MODEL, min_length=1, max_length=100)
    temperature: float = Field(default=0.0, ge=0, le=1)
    prompt_version: Version = config.PROMPT_VERSION


@app.middleware("http")
async def log_request(request: Request, call_next):
    if request.url.path not in {"/ask", "/retrieve"}:
        return await call_next(request)
    start = perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        params = getattr(request.state, "params", None)
        logger.info("ruta=%s backend=%s k=%s modelo=%r latencia_ms=%.3f estado=%s",
                    request.url.path, params.backend if params else "-",
                    params.k if params else "-", getattr(params, "model", "-"),
                    (perf_counter() - start) * 1000, status)


@app.exception_handler(retrieval.RetrieverUnavailable)
async def retriever_error(request: Request, error: retrieval.RetrieverUnavailable):
    return JSONResponse(status_code=503, content={"detail":
        f"El recuperador {error.backend} no está disponible: {error}"})


@app.exception_handler(EncoderMismatch)
async def encoder_error(request: Request, error: EncoderMismatch):
    return JSONResponse(status_code=503, content={"detail":
        f"La colección fue indexada con otro encoder: {error}"})


@app.exception_handler(generation.ModelNotAvailable)
async def model_error(request: Request, error: generation.ModelNotAvailable):
    model = error.model
    return JSONResponse(status_code=400, content={"detail":
        f"El modelo {model} no está descargado. Ejecute: docker compose exec ollama ollama pull {model}"})


@app.exception_handler(generation.GenerationError)
async def generation_error(request: Request, error: generation.GenerationError):
    model = getattr(getattr(request.state, "params", None), "model", config.DEFAULT_MODEL)
    return JSONResponse(status_code=503, content={"detail":
        f"No se pudo generar la respuesta con {model}: {error}"})


@app.exception_handler(FileNotFoundError)
async def prompt_error(request: Request, error: FileNotFoundError):
    params = getattr(request.state, "params", None)
    if isinstance(params, AskRequest):
        version = params.prompt_version
        path = Path(__file__).resolve().parents[1] / "prompts" / f"{version}.txt"
        if error.filename and Path(error.filename).resolve() == path:
            return JSONResponse(status_code=400, content={"detail":
                f"La versión de prompt {version} no existe"})
    return JSONResponse(status_code=500, content={"detail":
        "No se pudo completar la solicitud: falta un archivo requerido"})


def _milvus_alive():
    # Sin close(): pymilvus comparte el canal gRPC por URI y cerrarlo corta búsquedas en curso.
    milvus_store.get_client().list_collections(timeout=HEALTH_TIMEOUT)
    return True


def _ollama_alive():
    response = requests.get(config.OLLAMA_URL.rstrip("/") + "/api/tags", timeout=HEALTH_TIMEOUT)
    response.raise_for_status()
    return True


def _probe(check, result):
    try:
        result.set_result(bool(check()))
    except Exception:
        result.set_result(False)


@app.get("/health")
def health():
    checks = {"milvus": _milvus_alive, "solr": solr_store.ping, "ollama": _ollama_alive}
    deadline = perf_counter() + HEALTH_TIMEOUT
    # Reutiliza chequeos bloqueados para mantener como máximo tres hilos pendientes.
    with _health_lock:
        for name, check in checks.items():
            previous = _health_pending.get(name)
            if previous is None or not previous[0].is_alive():
                result = Future()
                thread = Thread(target=_probe, args=(check, result), daemon=True)
                thread.start()
                _health_pending[name] = (thread, result)
        pending = dict(_health_pending)
    deps = {}
    for name, (thread, result) in pending.items():
        thread.join(timeout=max(0, deadline - perf_counter()))
        try:
            deps[name] = result.result(timeout=0)
        except TimeoutError:
            deps[name] = False
    return {"status": "ok" if all(deps.values()) else "degraded", "deps": deps}


@app.get("/models")
def models():
    try:
        return {"models": generation.list_models()}
    except generation.GenerationError as error:
        return JSONResponse(status_code=503, content={"detail":
            f"No se pudieron consultar los modelos de Ollama: {error}"})


@app.post("/retrieve")
def search(params: RetrieveRequest, request: Request):
    request.state.params = params
    start = perf_counter()
    chunks = retrieval.retrieve(**params.model_dump())
    return {"chunks": chunks, "latency_ms": (perf_counter() - start) * 1000}


@app.post("/ask")
def ask(params: AskRequest, request: Request):
    request.state.params = params
    return rag.answer(**params.model_dump())
