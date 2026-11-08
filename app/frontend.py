"""Interfaz Streamlit: compara Solr (BM25) y Milvus (denso) sobre el mismo RAG."""
import os
from datetime import datetime, timezone

import requests
import streamlit as st

from app.privacy import mask_handles

API_URL = os.getenv("API_URL", "http://api:8000")
TIMEOUT = 300
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "rag_v3")
MOTORES = {"Solr (BM25)": "solr", "Milvus (denso)": "milvus", "Híbrido RRF": "hybrid"}
COMANDO_PULL = "docker compose exec ollama ollama pull qwen2.5:1.5b"


def motor_a_backend(label):
    return MOTORES[label]


def texto_literal(text):
    """Prepara texto privado para mostrarlo sin interpretar su sintaxis."""
    return mask_handles(str(text))


def construir_payload(query, backend, k, model, temperature, coleccion, idioma,
                      one_per_doc, chunker, encoder, prompt_version):
    filtros = {}
    if coleccion != "Todas":
        filtros["collection_name"] = coleccion
    if idioma != "Todos":
        filtros["language"] = idioma
    return {"query": query, "backend": backend, "k": k, "model": model,
            "temperature": temperature, "filters": filtros or None,
            "one_per_doc": one_per_doc, "chunker": chunker, "encoder": encoder,
            "prompt_version": prompt_version}


def formatear_detalle(detail):
    if isinstance(detail, list):
        return "; ".join(str(d.get("msg", d)) if isinstance(d, dict) else str(d) for d in detail)
    return str(detail)


def formatear_fecha(created_ts):
    if created_ts is None or created_ts < 0:
        return "s/f"
    return datetime.fromtimestamp(created_ts, tz=timezone.utc).strftime("%Y-%m-%d")


def formatear_evidencia_titulo(ev):
    return mask_handles(f"[{ev.get('n')}] {ev.get('title') or '(sin título)'} · {ev.get('source') or 's/f'}"
                        f" · {formatear_fecha(ev.get('created_ts'))} · {float(ev.get('score') or 0):.3f}")


def _llamar(metodo, ruta, **kw):
    """Devuelve (json, error_en_texto); nunca lanza."""
    try:
        r = requests.request(metodo, API_URL + ruta, timeout=TIMEOUT, **kw)
    except requests.ConnectionError:
        return None, f"No se pudo conectar con la API en {API_URL}"
    except requests.Timeout:
        return None, "La API tardó demasiado en responder"
    except requests.RequestException as e:
        return None, f"Error de red: {e}"
    try:
        cuerpo = r.json()
    except ValueError:
        return None, f"Respuesta no válida de la API (HTTP {r.status_code})"
    if r.status_code >= 400:
        return None, formatear_detalle(cuerpo.get("detail", f"HTTP {r.status_code}")
                                       if isinstance(cuerpo, dict) else cuerpo)
    return cuerpo, None


def _mostrar_estado():
    datos, _ = _llamar("GET", "/health")
    if datos is None:
        st.error(f"La API no responde en {API_URL}")
        return
    st.caption(" · ".join(f"{'✅' if ok else '❌'} {n}" for n, ok in datos["deps"].items()))


def _mostrar_respuesta(titulo, out, error):
    st.subheader(titulo)
    if error:
        st.error(error)
        return
    st.code(texto_literal(out["answer"]), language=None, wrap_lines=True)
    if out["abstained"]:
        st.warning("Abstención: el modelo indicó que no hay evidencia suficiente.")
    if out["invalid_citations"]:
        st.warning(f"Citas inválidas en la respuesta: {out['invalid_citations']}")
    t, tok = out["timings"], out["tokens"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total (s)", f"{t['total_ms'] / 1000:.2f}")
    c2.metric("Recuperación (s)", f"{t['retrieval_ms'] / 1000:.2f}")
    c3.metric("Generación (s)", f"{t['generation_ms'] / 1000:.2f}")
    c4.metric("Tokens/s", f"{tok.get('tokens_per_s') or 0:.1f}")
    st.caption(f"Modelo: {out['model']} · prompt: {out['prompt_version']}")
    for ev in out["evidence"]:
        with st.expander(formatear_evidencia_titulo(ev)):
            st.code(texto_literal(ev["text"]), language=None, wrap_lines=True)
            if str(ev.get("url") or "").startswith("http"):
                st.code(texto_literal(ev["url"]), language=None, wrap_lines=True)
            st.caption(f"Colección: {ev.get('collection_name')} · idioma: {ev.get('language')}"
                       f" · documento: {ev.get('document_id')}"
                       f" · caracteres {ev.get('start_char')}–{ev.get('end_char')}"
                       f" · procedencias: {len(ev.get('provenance_ids') or [])}")


def main():
    st.set_page_config(page_title="Asistente RAG", layout="wide")
    st.title("Asistente RAG — Solr (BM25) vs Milvus (denso)")
    _mostrar_estado()
    with st.sidebar:
        motor = st.selectbox("Motor", list(MOTORES))
        comparar = st.checkbox("Comparar Solr vs Milvus")
        datos, _ = _llamar("GET", "/models")
        modelos = (datos or {}).get("models") or []
        if modelos:
            modelo = st.selectbox("Modelo", modelos)
        else:
            st.warning(f"No hay modelos disponibles. Descargue uno con: {COMANDO_PULL}")
            modelo = st.text_input("Modelo", "qwen2.5:1.5b")
        k = st.slider("k", 1, 20, 5)
        coleccion = st.selectbox("Colección", ["Todas", "bluesky_posts", "url_crawls"])
        idioma = st.selectbox("Idioma", ["Todos", "es", "en"])
        temperatura = st.slider("Temperatura", 0.0, 1.0, 0.0, 0.1)
        una_por_doc = st.checkbox("Un fragmento por documento")
        with st.expander("Opciones avanzadas"):
            chunker = st.selectbox("Chunker", ["tok256", "sent256", "tok128"])
            encoder = st.selectbox("Encoder", ["e5small", "minilm"])
    pregunta = st.text_area("Pregunta")
    if not st.button("Preguntar"):
        return
    if len(pregunta.strip()) < 3:
        st.error("Escriba una pregunta de al menos 3 caracteres.")
        return
    motores = [("Solr (BM25)", "solr"), ("Milvus (denso)", "milvus")] if comparar \
        else [(motor, motor_a_backend(motor))]
    columnas = st.columns(2) if comparar else [st.container()]
    with st.spinner("Consultando…"):
        for col, (titulo, backend) in zip(columnas, motores):
            payload = construir_payload(pregunta, backend, k, modelo, temperatura, coleccion,
                                        idioma, una_por_doc, chunker, encoder, PROMPT_VERSION)
            out, error = _llamar("POST", "/ask", json=payload)
            with col:
                _mostrar_respuesta(titulo, out, error)


if __name__ == "__main__":
    main()
