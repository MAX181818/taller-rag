"""Indexa un JSONL homogéneo en Milvus y verifica el conteo exacto."""
import argparse
import json
import logging
from pathlib import Path
import time

import grpc
from pymilvus.exceptions import MilvusException

from app import config, embeddings, milvus_store


logger = logging.getLogger(__name__)
TRANSIENT_GRPC_CODES = {grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED,
                        grpc.StatusCode.RESOURCE_EXHAUSTED, grpc.StatusCode.ABORTED,
                        grpc.StatusCode.INTERNAL, grpc.StatusCode.UNKNOWN}


def _read_chunks(path):
    records = []
    ids = set()
    chunker = None
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                label = row["chunker"]
                if not isinstance(label, str) or not label:
                    raise ValueError("chunker vacío o inválido")
                if chunker is not None and label != chunker:
                    raise ValueError("Todas las líneas deben tener el mismo chunker")
                if row["chunk_id"] in ids:
                    raise ValueError(f"chunk_id repetido: {row['chunk_id']}")
                milvus_store.prepare_record(row)
                ids.add(row["chunk_id"])
                chunker = label
                records.append(row)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Línea {line_number}: {error}") from error
    if not records:
        raise ValueError("El JSONL no contiene fragmentos")
    return records, chunker


def _retry_batch(operation):
    for attempt in range(1, 4):
        try:
            return operation()
        except (ConnectionError, TimeoutError, MilvusException, grpc.RpcError) as error:
            if isinstance(error, MilvusException) and error.code not in {1, 2, 4, 14, 65535} | TRANSIENT_GRPC_CODES:
                raise
            if isinstance(error, grpc.RpcError) and error.code() not in TRANSIENT_GRPC_CODES:
                raise
            logger.warning("Fallo de conexión/servidor en intento %s/3: %s", attempt, error)
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt - 1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--encoder", choices=embeddings.ENCODERS, default=config.ENCODER)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args(argv)
    if args.batch <= 0:
        parser.error("--batch debe ser positivo")
    logging.basicConfig(level=config.LOG_LEVEL)
    started = time.perf_counter()
    try:
        records, chunker = _read_chunks(args.chunks)
        name = config.collection_name(args.encoder, chunker)
        fp = embeddings.fingerprint(args.encoder)
        client = milvus_store.get_client()
        if args.recreate:
            milvus_store.drop(client, name)
        milvus_store.ensure_collection(client, name, embeddings.dim(args.encoder), fp)
        for offset in range(0, len(records), args.batch):
            batch = records[offset:offset + args.batch]

            def insert_batch():
                if args.resume:
                    ids = {row["chunk_id"] for row in batch}
                    if milvus_store.existing_ids(client, name, list(ids)) == ids:
                        return
                vectors = embeddings.embed([row["text"] for row in batch], "passage", args.encoder)
                milvus_store.upsert(client, name, batch, vectors)

            _retry_batch(insert_batch)
            logger.info("Procesados %s/%s fragmentos en %s", offset + len(batch), len(records), name)
        total = milvus_store.count(client, name)
        if args.describe:
            schema = client.describe_collection(collection_name=name)
            report = dict(schema=schema, index=milvus_store.INDEX_SPEC,
                          description=schema["description"], fingerprint=fp, count=total)
            path = Path("reports/milvus") / f"schema_{name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(dict(collection=name, expected=len(records), count=total,
                              seconds=time.perf_counter() - started)))
        return 0 if total == len(records) else 1
    except (OSError, ValueError, KeyError, TypeError, MilvusException, grpc.RpcError, embeddings.EncoderMismatch) as error:
        logger.error("No se pudo indexar: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
