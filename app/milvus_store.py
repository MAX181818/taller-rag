"""Almacenamiento vectorial con esquema y procedencia explícitos."""
import json

from pymilvus import DataType, MilvusClient

from app import config
from app.embeddings import EncoderMismatch


VARCHAR_LIMITS = {
    "chunk_id": 64, "text": 16384, "title": 2048, "url": 4096,
    "document_id": 256, "provenance_ids": 65535, "collection_name": 64,
    "source": 64, "language": 16, "chunker": 32,
}
INTEGER_FIELDS = {"created_ts": DataType.INT64, "start_char": DataType.INT64,
                  "end_char": DataType.INT64, "n_tokens": DataType.INT32}
RECORD_FIELDS = list(VARCHAR_LIMITS) + list(INTEGER_FIELDS)
INDEX_SPEC = dict(field_name="vector", index_type="HNSW", metric_type="COSINE",
                  params=dict(M=16, efConstruction=200))


def get_client() -> MilvusClient:
    return MilvusClient(uri=config.MILVUS_URI)


def _check_fingerprint(client, name, fingerprint):
    description = client.describe_collection(collection_name=name).get("description", "")
    try:
        stored = json.loads(description)
    except (ValueError, TypeError):
        stored = None
    if stored != fingerprint:
        raise EncoderMismatch(f"La huella del codificador no coincide con la colección {name}")


def ensure_collection(client, name, dim, fingerprint: dict) -> None:
    if client.has_collection(collection_name=name):
        _check_fingerprint(client, name, fingerprint)
    else:
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False,
                                      description=json.dumps(fingerprint, sort_keys=True))
        for field, limit in VARCHAR_LIMITS.items():
            schema.add_field(field_name=field, datatype=DataType.VARCHAR, max_length=limit,
                             is_primary=field == "chunk_id")
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
        for field, datatype in INTEGER_FIELDS.items():
            schema.add_field(field_name=field, datatype=datatype)
        indexes = client.prepare_index_params()
        indexes.add_index(**INDEX_SPEC)
        client.create_collection(collection_name=name, schema=schema, index_params=indexes,
                                 consistency_level="Strong")
    client.load_collection(collection_name=name)


def prepare_record(record: dict) -> dict:
    """Valida bytes UTF-8 y recorta únicamente título y URL."""
    result = {}
    chunk_id = record.get("chunk_id", "desconocido")
    for field, limit in VARCHAR_LIMITS.items():
        value = record[field]
        if field == "provenance_ids":
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(f"provenance_ids debe ser una lista de textos: chunk_id={chunk_id}")
            value = json.dumps(value, ensure_ascii=False)
        if value is None:
            value = ""
        if not isinstance(value, str):
            raise ValueError(f"El campo {field} debe ser texto: chunk_id={chunk_id}")
        encoded = value.encode("utf-8")
        if len(encoded) > limit:
            if field in ("title", "url"):
                value = encoded[:limit].decode("utf-8", errors="ignore")
            else:
                raise ValueError(f"El campo {field} del chunk_id={chunk_id} ocupa {len(encoded)} bytes UTF-8; límite {limit}")
        result[field] = value
    for field in INTEGER_FIELDS:
        result[field] = record[field]
    return result


def upsert(client, name, records: list[dict], vectors) -> int:
    if len(records) != len(vectors):
        raise ValueError("El número de registros y vectores debe coincidir")
    rows = [dict(prepare_record(record), vector=list(vector)) for record, vector in zip(records, vectors)]
    if not rows:
        return 0
    return int(client.upsert(collection_name=name, data=rows)["upsert_count"])


def filter_expression(filters: dict | None) -> str:
    if not filters:
        return ""
    unsupported = set(filters) - {"collection_name", "language"}
    if unsupported:
        raise ValueError(f"Filtros no soportados: {sorted(unsupported)}")
    return " and ".join(f"{field} == {json.dumps(value, ensure_ascii=False)}"
                        for field, value in filters.items())


def search(client, name, qvec, k, filters: dict | None = None,
           fingerprint: dict | None = None) -> list[dict]:
    expression = filter_expression(filters)
    if fingerprint is not None:
        _check_fingerprint(client, name, fingerprint)
    hits = client.search(collection_name=name, data=[list(qvec)], anns_field="vector", limit=k,
                         filter=expression, output_fields=RECORD_FIELDS,
                         search_params=dict(metric_type="COSINE", params=dict(ef=64)),
                         consistency_level="Strong")
    result = []
    for hit in hits[0]:
        row = dict(hit["entity"])
        row["chunk_id"] = hit["id"]
        row["provenance_ids"] = json.loads(row["provenance_ids"])
        row["score"] = float(hit["distance"])
        result.append(row)
    return result


def count(client, name) -> int:
    rows = client.query(collection_name=name, filter="", output_fields=["count(*)"],
                        consistency_level="Strong")
    return int(rows[0]["count(*)"])


def drop(client, name):
    if client.has_collection(collection_name=name):
        client.drop_collection(collection_name=name)


def existing_ids(client, name, ids: list[str]) -> set[str]:
    if not ids:
        return set()
    rows = client.query(collection_name=name, ids=ids, output_fields=["chunk_id"],
                        consistency_level="Strong")
    return {row["chunk_id"] for row in rows}
