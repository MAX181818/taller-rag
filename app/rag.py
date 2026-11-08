"""Orquestación de recuperación, contexto, generación y registro."""
from functools import lru_cache
import hashlib
from time import perf_counter
from uuid import uuid4

from app import config, generation, runlog
from app.citations import invalid_citations, is_abstention, parse_citations
from app.prompts import ABSTENTION, build_context, build_prompt, load_prompt
from app.retrieval import retrieve


@lru_cache(maxsize=None)
def _cached_model_info(model) -> dict:
    return generation.model_info(model)


def _model_info(model) -> dict:
    try:
        return _cached_model_info(model)
    except Exception:
        return {}


def answer(query, *, backend="milvus", model=config.DEFAULT_MODEL, k=config.DEFAULT_K,
           chunker=config.CHUNKER, encoder=config.ENCODER, filters=None,
           temperature=0.0, prompt_version=config.PROMPT_VERSION, one_per_doc=False,
           evidence: list[dict] | None = None) -> dict:
    start = perf_counter()
    record = {
        "run_id": uuid4().hex, "query": query, "answer": ABSTENTION, "abstained": True,
        "citations": [], "invalid_citations": [], "model": model, "model_info": {},
        "backend": backend, "chunker": chunker, "encoder": encoder, "k": k,
        "prompt_version": prompt_version, "evidence": [],
        "context_sha256": hashlib.sha256(b"").hexdigest(),
        "timings": {"retrieval_ms": 0, "generation_ms": 0, "total_ms": 0},
        "tokens": {"prompt": 0, "completion": 0, "tokens_per_s": 0.0, "load_s": 0.0},
        "params": {"temperature": temperature, "seed": 42, "num_ctx": 4096,
                   "num_predict": 384, "max_context_chars": config.MAX_CONTEXT_CHARS,
                   "one_per_doc": one_per_doc, "filters": filters},
    }
    try:
        template, record["prompt_version"] = load_prompt(prompt_version)
        if evidence is None:
            retrieval_start = perf_counter()
            try:
                evidence = retrieve(query, backend=backend, k=k, chunker=chunker,
                                    encoder=encoder, filters=filters, one_per_doc=one_per_doc)
            finally:
                record["timings"]["retrieval_ms"] = (perf_counter() - retrieval_start) * 1000
        context, record["evidence"] = build_context(evidence, config.MAX_CONTEXT_CHARS)
        record["context_sha256"] = hashlib.sha256(context.encode("utf-8")).hexdigest()
        if record["evidence"]:
            record["model_info"] = dict(_model_info(model))
            prompt = build_prompt(query, context, template)
            generation_start = perf_counter()
            try:
                result = generation.generate(model, prompt, temperature=temperature,
                                             seed=42, num_ctx=4096, num_predict=384)
            finally:
                record["timings"]["generation_ms"] = (perf_counter() - generation_start) * 1000
            record["answer"] = result["answer"]
            record["tokens"] = {"prompt": result["prompt_tokens"], "completion": result["completion_tokens"],
                                "tokens_per_s": result["tokens_per_s"], "load_s": result["load_s"]}
        record["abstained"] = is_abstention(record["answer"])
        record["citations"] = parse_citations(record["answer"])
        record["invalid_citations"] = invalid_citations(record["citations"], len(record["evidence"]))
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["timings"]["total_ms"] = (perf_counter() - start) * 1000
        runlog.log_run(record)
    return record
