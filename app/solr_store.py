"""Acceso al core de búsqueda léxica y normalización de registros."""
import os
import re

import requests


SOLR_URL = os.environ.get("SOLR_URL", "http://solr:8983/solr/rag").rstrip("/")
_TIMEOUT = 30
_FIELDS = frozenset({"chunk_id", "document_id", "provenance_ids", "chunker",
                     "start_char", "end_char", "n_tokens", "text", "title", "url",
                     "collection_name", "source", "language", "created_ts"})
_SCHEMA = _FIELDS | {"text_es", "text_en"}
_SPECIAL = re.compile(r'([+\-!(){}\[\]^"~*?:\\/]|[&|](?=[&|])|(?<=[&|])[&|])')
_RESERVED = re.compile(r"\b(?:AND|OR|NOT)\b")


def escape_query(q: str) -> str:
    """Neutraliza los operadores y caracteres de la consulta del usuario."""
    return _SPECIAL.sub(r"\\\1", _RESERVED.sub(lambda match: match.group().lower(), q))


def _filter(field: str, value: str) -> str:
    escaped = _SPECIAL.sub(r"\\\1", str(value))
    if not escaped or any(char.isspace() for char in escaped) or _RESERVED.search(escaped):
        escaped = f'"{escaped}"'
    return f"{field}:{escaped}"


def _get(path, params):
    result = requests.get(f"{SOLR_URL}/{path}", params=params, timeout=_TIMEOUT)
    result.raise_for_status()
    return result.json()


def _update(payload, commit=False):
    result = requests.post(f"{SOLR_URL}/update", json=payload,
                           params={"commit": "true"} if commit else {}, timeout=_TIMEOUT)
    result.raise_for_status()


def upsert(records: list[dict], commit: bool = True) -> int:
    """Indexa solo campos del esquema, conservando la procedencia múltiple."""
    documents = [{key: value for key, value in record.items() if key in _SCHEMA}
                 for record in records]
    _update(documents, commit=commit)
    return len(documents)


def search(query: str, k: int, chunker: str, filters: dict | None = None) -> list[dict]:
    """Busca con eDisMax y devuelve registros con puntuación numérica."""
    fq = [_filter("chunker", chunker)]
    for field, value in (filters or {}).items():
        if field not in {"collection_name", "language"}:
            raise ValueError(f"Filtro no admitido: {field}")
        fq.append(_filter(field, value))
    result = _get("select", dict(defType="edismax", q=escape_query(query),
                                qf="text text_es^0.8 text_en^0.8 title^0.5",
                                fq=fq, rows=k, fl="*,score", wt="json"))
    documents = []
    for doc in result["response"]["docs"]:
        record = {key: value for key, value in doc.items() if key in _FIELDS}
        provenance = record.get("provenance_ids") or []
        record["provenance_ids"] = [provenance] if isinstance(provenance, str) else list(provenance)
        record["score"] = float(doc["score"])
        documents.append(record)
    return documents


def count(chunker: str) -> int:
    """Cuenta los fragmentos de una segmentación."""
    return int(_get("select", dict(q="*:*", fq=_filter("chunker", chunker), rows=0, wt="json"))
               ["response"]["numFound"])


def ping() -> bool:
    """Comprueba disponibilidad sin propagar errores de transporte."""
    try:
        return _get("admin/ping", {"wt": "json"}).get("status") == "OK"
    except (requests.RequestException, ValueError):
        return False


def delete_chunker(chunker: str) -> None:
    """Borra una segmentación y confirma la operación."""
    _update({"delete": {"query": _filter("chunker", chunker)}}, commit=True)


def commit() -> None:
    """Confirma las actualizaciones pendientes."""
    _update({"commit": {}})


def contains_all(chunk_ids: list[str], chunker: str) -> bool:
    """Consulta por términos para decidir si se puede omitir un lote."""
    unique_ids = set(chunk_ids)
    result = requests.post(f"{SOLR_URL}/select", timeout=_TIMEOUT,
                           data={"q": "{!terms f=chunk_id}" + ",".join(sorted(unique_ids)),
                                 "fq": _filter("chunker", chunker), "rows": 0, "wt": "json"})
    result.raise_for_status()
    return int(result.json()["response"]["numFound"]) == len(unique_ids)
