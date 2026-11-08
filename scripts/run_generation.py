"""Ejecuta experimentos de generación con evidencia congelada."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from importlib import metadata
import json
import logging
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
from time import perf_counter
from urllib.parse import urlsplit, urlunsplit

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config
from app.prompts import load_prompt

LOGGER = logging.getLogger(__name__)
PACKAGES = ("pymilvus", "sentence-transformers", "torch", "fastapi")
DEFAULT_MODELS = ("qwen2.5:1.5b", "llama3.2:1b", "gemma2:2b", "llama3.2:3b")


def retrieve(*args, **kwargs):
    from app.retrieval import retrieve as implementation
    return implementation(*args, **kwargs)


def answer(*args, **kwargs):
    from app.rag import answer as implementation
    return implementation(*args, **kwargs)


def unload(model):
    from app.generation import unload as implementation
    return implementation(model)


def model_info(model):
    from app.generation import model_info as implementation
    return implementation(model)


def list_models():
    from app.generation import list_models as implementation
    return implementation()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def collect_hardware():
    """Obtiene los datos disponibles una vez, sin exigir GPU ni Docker."""
    result = dict(cpu=None, ram_gb=None, gpu=None, vram_gb=None)
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        match = re.search(r"^(?:model name|Hardware)\s*:\s*(.+)$", cpuinfo, re.MULTILINE)
        if match:
            result["cpu"] = match[1].strip()
    except OSError:
        pass
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        match = re.search(r"^MemTotal:\s*(\d+)\s+kB", meminfo, re.MULTILINE)
        if match:
            result["ram_gb"] = int(match[1]) / 1024**2
    except OSError:
        pass
    try:
        command = subprocess.run(
            ["docker", "compose", "exec", "-T", "ollama", "nvidia-smi",
             "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False)
        if command.returncode == 0 and command.stdout.strip():
            rows = [line.rsplit(",", 1) for line in command.stdout.strip().splitlines()]
            result["gpu"] = "; ".join(row[0].strip() for row in rows)
            result["vram_gb"] = sum(float(row[1].strip()) for row in rows) / 1024
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        result.update(gpu=None, vram_gb=None)
    return result


def _http_json(url):
    import requests

    response = requests.get(url, timeout=2)
    response.raise_for_status()
    return response.json()


def _milvus_version():
    from pymilvus import MilvusClient

    client = MilvusClient(uri=config.MILVUS_URI, timeout=2)
    return client.get_server_version(timeout=2)


def collect_versions():
    """Los servicios o paquetes ausentes se registran como null."""
    result = dict(ollama=None, milvus=None, solr=None, python=platform.python_version(), packages={})
    try:
        result["ollama"] = _http_json(config.OLLAMA_URL.rstrip("/") + "/api/version").get("version")
    except Exception:
        pass
    try:
        parsed = urlsplit(config.SOLR_URL)
        base = parsed.path.split("/solr", 1)[0] + "/solr/admin/info/system"
        url = urlunsplit((parsed.scheme, parsed.netloc, base, "", ""))
        result["solr"] = _http_json(url).get("lucene", {}).get("solr-spec-version")
    except Exception:
        pass
    try:
        result["milvus"] = _milvus_version()
    except Exception:
        pass
    for package in PACKAGES:
        try:
            result["packages"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result["packages"][package] = None
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=Path("eval/gold.jsonl"))
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--backends", nargs="+", choices=("milvus", "solr", "hybrid"),
                        default=["milvus", "solr"])
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--chunker", default="tok256")
    parser.add_argument("--encoder", default="e5small")
    parser.add_argument("--prompt", default="rag_v3")
    parser.add_argument("--tag", default="main")
    parser.add_argument("--one-per-doc", action="store_true")
    parser.add_argument("--qids")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/runs/raw"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser


def _questions(args):
    if args.runs <= 0 or args.k <= 0:
        raise ValueError("--runs y --k deben ser positivos.")
    if not re.fullmatch(r"[\w.-]+", args.tag) or args.tag in (".", ".."):
        raise ValueError("--tag debe ser un nombre simple.")
    for values in (args.models, args.backends):
        if len(set(values)) != len(values):
            raise ValueError("No se admiten modelos ni motores duplicados.")
    with args.gold.open(encoding="utf-8-sig") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    rows = [row for row in rows if row["split"] == args.split]
    if args.qids is not None:
        selected = {qid.strip() for qid in args.qids.split(",") if qid.strip()}
        if not selected or selected - {row["qid"] for row in rows}:
            raise ValueError("Hay qids vacíos o ausentes en el split solicitado.")
        rows = [row for row in rows if row["qid"] in selected]
    if not rows:
        raise ValueError("No hay preguntas para el experimento.")
    if len({row["qid"] for row in rows}) != len(rows):
        raise ValueError("El gold contiene qids duplicados.")
    return rows


def _read_jsonl(path, *, tolerate_partial=False):
    if not path.exists():
        return []
    # En bytes: una interrupción puede cortar un carácter UTF-8 a la mitad.
    lines = path.read_bytes().splitlines()
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if tolerate_partial and index == len(lines) - 1:
                LOGGER.warning("Última línea incompleta en %s; se volverá a intentar.", path)
            else:
                raise
    return rows


def _append_stream(path):
    data = path.read_bytes() if path.exists() else b""
    lines = data.splitlines(keepends=True)
    if lines and lines[-1].strip():
        try:
            json.loads(lines[-1].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            data = b"".join(lines[:-1])
            path.write_bytes(data)
            LOGGER.warning("Se descartó la última línea incompleta de %s.", path)
    stream = path.open("a", encoding="utf-8", newline="\n")
    if data and not data.endswith(b"\n"):
        stream.write("\n")
        stream.flush()
    return stream


def _write_record(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    stream.flush()


def _freeze_evidence(args, questions):
    path = args.out_dir / f"{args.tag}_evidence.jsonl"
    retrieval_params = dict(k=args.k, chunker=args.chunker, encoder=args.encoder,
                            filters=None, one_per_doc=args.one_per_doc)
    cached = {(row["qid"], row["backend"]): row
              for row in _read_jsonl(path, tolerate_partial=True)}
    # Reusar evidencia congelada con otros parámetros mezclaría condiciones sin aviso.
    stale = {json.dumps(row.get("params"), sort_keys=True) for row in cached.values()
             if row.get("params") != retrieval_params}
    if stale:
        raise ValueError(f"La evidencia congelada en {path} se generó con otros parámetros "
                         f"({', '.join(sorted(stale))}); use otro --tag o borre ese archivo.")
    with _append_stream(path) as target:
        for backend in args.backends:
            missing = [q for q in questions if (q["qid"], backend) not in cached]
            if not missing:
                continue
            retrieve(missing[0]["question"], backend=backend, **retrieval_params)
            for question in missing:
                start = perf_counter()
                chunks = retrieve(question["question"], backend=backend, **retrieval_params)
                record = dict(qid=question["qid"], backend=backend, chunks=deepcopy(chunks),
                              retrieval_ms=(perf_counter() - start) * 1000,
                              params=retrieval_params)
                _write_record(target, record)
                cached[(question["qid"], backend)] = record
    return cached


def _completed(args):
    done = set()
    pattern = re.compile(re.escape(args.tag) + r"_\d{8}_\d{4}\.jsonl")
    for path in args.out_dir.iterdir():
        if not path.is_file() or not pattern.fullmatch(path.name):
            continue
        for row in _read_jsonl(path, tolerate_partial=True):
            if row.get("error") is None:
                key = tuple(row.get(name) for name in ("model", "backend", "qid", "run_idx"))
                done.add(key)
    return done


def _effective_config(args, questions, hardware, versions, prompt_version):
    details = {}
    for model in args.models:
        try:
            details[model] = model_info(model)
        except Exception as error:
            LOGGER.warning("No se pudieron obtener metadatos de %s: %s", model, error)
            details[model] = None
    return {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "prompt_version": prompt_version, "MAX_CONTEXT_CHARS": config.MAX_CONTEXT_CHARS,
        "inference_params": dict(temperature=args.temperature, seed=42, num_ctx=4096,
                                 num_predict=384, keep_alive="10m"),
        "model_info": details, "hardware": hardware, "versions": versions,
        "date": utc_now(), "seed": 42, "qids_used": [q["qid"] for q in questions],
    }


def run(args):
    """Devuelve el resumen; cada llamada queda persistida antes de continuar."""
    start = perf_counter()
    questions = _questions(args)
    _, prompt_version = load_prompt(args.prompt)
    random.seed(42)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    evidence = _freeze_evidence(args, questions)
    done = _completed(args) if args.resume else set()
    hardware, versions = collect_hardware(), collect_versions()
    effective = _effective_config(args, questions, hardware, versions, prompt_version)
    config_path = args.out_dir / f"{args.tag}_config.json"
    if args.resume and config_path.exists():
        config_path = args.out_dir / f"{args.tag}_config_resume_{datetime.now(timezone.utc):%Y%m%d_%H%M}.json"
    config_path.write_text(
        json.dumps(effective, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    path = args.out_dir / f"{args.tag}_{datetime.now(timezone.utc):%Y%m%d_%H%M}.jsonl"
    total = sum((model, backend, q["qid"], index) not in done
                for model in args.models for index in range(1, args.runs + 1)
                for backend in args.backends for q in questions)
    calls = errors = 0
    known_models = []
    if total:
        try:
            known_models = list_models()
        except Exception as error:
            LOGGER.warning("No se pudieron listar los modelos: %s", error)
    generation_start = perf_counter()
    with _append_stream(path) as target:
        for model in args.models:
            pending = [(index, backend, q) for index in range(1, args.runs + 1)
                       for backend in args.backends for q in questions
                       if (model, backend, q["qid"], index) not in done]
            if not pending:
                continue
            others = dict.fromkeys([*known_models, *args.models])
            for other in [*[name for name in others if name != model], model]:
                try:
                    unload(other)
                except Exception as error:
                    # Un modelo no descargado no debe abortar la corrida: sus llamadas quedan con error.
                    LOGGER.warning("No se pudo descargar %s de memoria: %s", other, error)
            for position, (index, backend, question) in enumerate(pending):
                frozen = evidence[(question["qid"], backend)]
                record = dict(model=model, backend=backend, query=question["question"],
                              k=args.k, chunker=args.chunker, encoder=args.encoder,
                              prompt_version=prompt_version)
                try:
                    record.update(answer(
                        question["question"], backend=backend, model=model, k=args.k,
                        chunker=args.chunker, encoder=args.encoder, filters=None,
                        temperature=args.temperature, prompt_version=args.prompt,
                        one_per_doc=args.one_per_doc, evidence=deepcopy(frozen["chunks"])))
                except Exception as error:
                    errors += 1
                    record["error"] = str(error)
                else:
                    if record.get("error") is not None:
                        errors += 1
                record.update(qid=question["qid"], split=question["split"], type=question.get("type"),
                              run_idx=index, cold=position == 0, tag=args.tag,
                              retrieval_ms_original=frozen["retrieval_ms"], hardware=hardware,
                              versions=versions, timestamp=utc_now())
                _write_record(target, record)
                calls += 1
                if calls % 10 == 0:
                    elapsed = perf_counter() - generation_start
                    LOGGER.info("%s: %d/%d (%.1f%%), ETA %.1fs", args.tag, calls, total,
                                calls / total * 100, elapsed / calls * (total - calls))
    return dict(tag=args.tag, calls=calls, errors=errors, seconds=perf_counter() - start, file=str(path))


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        summary = run(args)
    except (OSError, ValueError, KeyError) as error:
        LOGGER.error("No se pudo ejecutar el experimento: %s", error)
        return 1
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
