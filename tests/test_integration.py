"""Verificaciones reales del piloto; ejecutar dentro del contenedor de la API."""
import os
from pathlib import Path
import socket
from urllib.parse import urlsplit

import pytest
import requests


pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def servicios():
    uri = os.getenv("MILVUS_URI")
    if not uri:
        pytest.skip("MILVUS_URI no está configurada; requiere los servicios locales")
    address = urlsplit(uri if "://" in uri else "http://" + uri)
    try:
        with socket.create_connection((address.hostname, address.port or 19530), timeout=2):
            pass
    except OSError as error:
        pytest.skip(f"MILVUS_URI no responde: {error}")
    from pymilvus import MilvusClient
    from pymilvus.exceptions import MilvusException

    try:
        client = MilvusClient(uri=uri, timeout=3)
        client.list_collections(timeout=3)
    except MilvusException as error:
        pytest.skip(f"MILVUS_URI no responde por gRPC: {error}")
    # No cerrar: el cliente comparte el canal por URI con otras conexiones.
    return client


@pytest.fixture(scope="module")
def lineas():
    path = ROOT / "data/derived/chunks_tok256.jsonl"
    assert path.is_file(), f"Falta el piloto: {path}"
    with path.open(encoding="utf-8") as source:
        total = sum(1 for line in source if line.strip())
    assert total > 0
    return total


def test_conteo_milvus(servicios, lineas):
    rows = servicios.query(collection_name="c_e5small_tok256", filter="",
                           output_fields=["count(*)"], consistency_level="Strong", timeout=10)
    assert int(rows[0]["count(*)"]) == lineas


def test_conteo_solr(servicios, lineas):
    url = os.getenv("SOLR_URL", "http://solr:8983/solr/rag").rstrip("/")
    response = requests.get(url + "/select", params={"q": "*:*", "fq": "chunker:tok256",
                                                     "rows": 0, "wt": "json"}, timeout=10)
    response.raise_for_status()
    assert int(response.json()["response"]["numFound"]) == lineas


def test_health(servicios):
    url = os.getenv("API_URL", "http://api:8000").rstrip("/")
    response = requests.get(url + "/health", timeout=15)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ok"
    assert body["deps"] == {"milvus": True, "solr": True, "ollama": True}


@pytest.mark.parametrize("backend", ["solr", "milvus"])
def test_ask_con_evidencia(servicios, backend):
    url = os.getenv("API_URL", "http://api:8000").rstrip("/")
    response = requests.post(url + "/ask", json={
        "query": "¿Qué señalan los documentos sobre la transición energética en Colombia?",
        "backend": backend, "model": os.getenv("DEFAULT_MODEL", "qwen2.5:1.5b"),
        "encoder": "e5small", "chunker": "tok256", "k": 5,
        "temperature": 0, "prompt_version": "rag_v3",
    }, timeout=300)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["backend"] == backend
    assert body["answer"].strip()
    assert body["evidence"]
    assert all(e["text"].strip() and e["document_id"] and e["chunk_id"]
               for e in body["evidence"])
