"""Pruebas de las funciones puras de la interfaz (sin Streamlit corriendo)."""
import sys

from app import frontend


def test_importar_no_ejecuta_main(monkeypatch):
    llamadas = []
    monkeypatch.setattr(frontend, "main", lambda: llamadas.append(1))
    import importlib
    importlib.reload(frontend)  # __name__ != "__main__"
    assert llamadas == []
    assert "streamlit" in sys.modules


def test_motor_a_backend():
    assert frontend.motor_a_backend("Solr (BM25)") == "solr"
    assert frontend.motor_a_backend("Milvus (denso)") == "milvus"
    assert frontend.motor_a_backend("Híbrido RRF") == "hybrid"


def test_payload_sin_filtros():
    p = frontend.construir_payload("¿hola?", "solr", 5, "m", 0.0, "Todas", "Todos",
                                   False, "tok256", "e5small", "v1")
    assert p["filters"] is None
    assert p == {"query": "¿hola?", "backend": "solr", "k": 5, "model": "m",
                 "temperature": 0.0, "filters": None, "one_per_doc": False,
                 "chunker": "tok256", "encoder": "e5small", "prompt_version": "v1"}


def test_payload_con_filtros():
    p = frontend.construir_payload("q q q", "milvus", 3, "m", 0.5, "bluesky_posts", "es",
                                   True, "sent256", "minilm", "v1")
    assert p["filters"] == {"collection_name": "bluesky_posts", "language": "es"}
    assert p["one_per_doc"] is True
    p = frontend.construir_payload("q q q", "milvus", 3, "m", 0.5, "Todas", "en",
                                   False, "tok256", "e5small", "v1")
    assert p["filters"] == {"language": "en"}


def test_formatear_detalle():
    assert frontend.formatear_detalle("fallo") == "fallo"
    lista = [{"msg": "campo requerido", "loc": ["body"]}, {"msg": "valor inválido"}]
    assert frontend.formatear_detalle(lista) == "campo requerido; valor inválido"


def test_formatear_fecha():
    assert frontend.formatear_fecha(-1) == "s/f"
    assert frontend.formatear_fecha(None) == "s/f"
    assert frontend.formatear_fecha(1700000000) == "2023-11-14"


def test_titulo_evidencia():
    ev = {"n": 2, "title": "Un título", "source": "bsky", "created_ts": 1700000000, "score": 0.12345}
    assert frontend.formatear_evidencia_titulo(ev) == "[2] Un título · bsky · 2023-11-14 · 0.123"
    ev = {"n": 1, "title": "", "source": "web", "created_ts": -1, "score": 1}
    assert frontend.formatear_evidencia_titulo(ev) == "[1] (sin título) · web · s/f · 1.000"
