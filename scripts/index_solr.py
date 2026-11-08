"""Indexa fragmentos JSONL en Solr con reanudación y verificación final."""
import argparse
import json
import logging
from pathlib import Path
import sys
import time

import requests

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import solr_store


logger = logging.getLogger(__name__)


def _records(path):
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict) or not record.get("chunk_id") or not record.get("chunker"):
                    raise ValueError("Faltan chunk_id o chunker")
            except ValueError as error:
                raise ValueError(f"Línea {number}: {error}") from error
            yield record


def _retry(operation):
    for attempt in range(3):
        try:
            return operation()
        except requests.RequestException as error:
            status = error.response.status_code if error.response is not None else None
            retryable = isinstance(error, (requests.ConnectionError, requests.Timeout)) or (
                status is not None and 500 <= status < 600)
            if not retryable or attempt == 2:
                raise
            delay = 2 ** attempt
            logger.warning("Intento %s fallido; reintento en %s s", attempt + 1, delay)
            time.sleep(delay)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=500)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.batch <= 0:
        parser.error("--batch debe ser positivo")
    started = time.perf_counter()
    try:
        chunker = None
        expected = 0
        for record in _records(args.chunks):
            chunker = chunker or record["chunker"]
            if record["chunker"] != chunker:
                raise ValueError("Todas las líneas deben tener el mismo chunker")
            expected += 1
        if not expected:
            raise ValueError("El JSONL está vacío; no se puede determinar el chunker")
        if args.recreate:
            _retry(lambda: solr_store.delete_chunker(chunker))
        batch = []
        batches = 0
        def flush():
            nonlocal batches
            exists = args.resume and _retry(lambda: solr_store.contains_all(
                [item["chunk_id"] for item in batch], chunker))
            if not exists:
                _retry(lambda: solr_store.upsert(batch, commit=False))
            batches += 1
            logger.info("Lote %s: %s registros, %s", batches, len(batch),
                        "omitido" if exists else "indexado")
            if batches % 10 == 0:
                _retry(solr_store.commit)
        for record in _records(args.chunks):
            batch.append(record)
            if len(batch) == args.batch:
                flush()
                batch = []
        if batch:
            flush()
        _retry(solr_store.commit)
        actual = solr_store.count(chunker)
        print(json.dumps(dict(collection="solr:rag", chunker=chunker, expected=expected,
                              count=actual, seconds=time.perf_counter() - started)))
        return 0 if actual == expected else 1
    except requests.RequestException as error:
        detail = error.response.text if error.response is not None else str(error)
        logger.error("Error de Solr: %s", detail)
        return 1
    except (OSError, ValueError) as error:
        logger.error("No se pudo indexar: %s", error)
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
