"""Vecinos semánticos, sensibilidad y solapamiento de codificadores."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import embeddings
from app.config import collection_name


QUERIES = (
    ("Transición energética y energías renovables en Colombia", "es"),
    ("Biodiversidad y acuerdos de la COP16 en Colombia", "es"),
    ("Elecciones presidenciales en Colombia", "es"),
    ("Conflicto armado y negociaciones de paz", "es"),
    ("Amazon rainforest conservation and deforestation", "en"),
    ("Venezuelan migration to Colombia", "en"),
    ("Earthquake damage and emergency response", "en"),
    ("Climate change and extreme weather", "en"),
)

PAIRS = tuple(
    dict(tipo=kind, izquierda=left, derecha=right)
    for kind, pairs in (
        ("parafrasis", (
            ("Colombia impulsa la energía solar.", "La energía del sol recibe impulso en Colombia."),
            ("La COP16 busca proteger la biodiversidad.", "Proteger la diversidad biológica es una meta de la COP16."),
            ("Los ciudadanos eligen presidente.", "La ciudadanía vota para elegir al presidente."),
            ("El gobierno negocia la paz.", "Las autoridades mantienen conversaciones de paz."),
            ("La Amazonía pierde bosque.", "La deforestación reduce la selva amazónica."),
            ("Migrantes venezolanos llegan a Colombia.", "Colombia recibe personas que migran desde Venezuela."),
        )),
        ("traduccion", (
            ("Colombia impulsa la energía solar.", "Colombia promotes solar energy."),
            ("La COP16 busca proteger la biodiversidad.", "COP16 seeks to protect biodiversity."),
            ("Los ciudadanos eligen presidente.", "Citizens elect a president."),
            ("El gobierno negocia la paz.", "The government negotiates peace."),
            ("Un terremoto dañó varias viviendas.", "An earthquake damaged several homes."),
            ("El cambio climático intensifica las lluvias.", "Climate change intensifies rainfall."),
        )),
        ("no_relacionados", (
            ("Colombia impulsa la energía solar.", "Una receta de pan lleva harina."),
            ("La COP16 busca proteger la biodiversidad.", "El equipo ganó el torneo de fútbol."),
            ("Los ciudadanos eligen presidente.", "The telescope observed a distant galaxy."),
            ("El gobierno negocia la paz.", "Los zapatos nuevos son azules."),
            ("Un terremoto dañó varias viviendas.", "The musician released a new album."),
            ("El cambio climático intensifica las lluvias.", "La biblioteca presta novelas históricas."),
        )),
    ) for left, right in pairs
)


def get_client():
    from app.milvus_store import get_client as connect
    return connect()


def search(*args, **kwargs):
    from app.milvus_store import search as retrieve
    return retrieve(*args, **kwargs)


def _embed_raw(texts, encoder):
    vectors = embeddings._model(encoder).encode(texts, normalize_embeddings=True)
    return [[float(value) for value in vector] for vector in vectors]


def cosine(left, right):
    if len(left) != len(right):
        raise ValueError("Los vectores deben tener igual dimensión")
    norm = math.sqrt(sum(x * x for x in left) * sum(x * x for x in right))
    if not norm:
        raise ValueError("El coseno requiere vectores no nulos")
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(left, right)) / norm))


def jaccard(left, right):
    left, right = set(left), set(right)
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["encoder"], row["prefijo"])
        groups.setdefault(key, {}).setdefault(row["tipo"], []).append(row["similitud"])
    summary = []
    for (encoder, prefix), types in groups.items():
        related = types["parafrasis"] + types["traduccion"]
        unrelated = types["no_relacionados"]
        gap = sum(related) / len(related) - sum(unrelated) / len(unrelated)
        for kind, values in types.items():
            summary.append(dict(encoder=encoder, prefijo=prefix, tipo=kind,
                                media=sum(values) / len(values), min=min(values),
                                max=max(values), brecha=gap))
    return summary


def _cell(value):
    masked = re.sub(r"@[\w]+", "@usuario", str(value))
    return masked.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def run(out, k=5):
    if k < 1:
        raise ValueError("k debe ser positivo")
    encoders = ("e5small", "minilm")
    fingerprints = {encoder: embeddings.fingerprint(encoder) for encoder in encoders}
    client = get_client()
    markdown = ["# Vecinos semánticos", ""]
    results = {}
    for encoder in encoders:
        name = collection_name(encoder, "tok256")
        vectors = embeddings.embed([query for query, _ in QUERIES], kind="query", encoder=encoder)
        for (query, language), vector in zip(QUERIES, vectors):
            hits = search(client, name, vector, max(k, 10), fingerprint=fingerprints[encoder])
            results[(query, encoder)] = {hit["document_id"] for hit in hits[:10]}
            markdown.extend([f"## {query} ({language}) — {encoder}", "",
                             "| rank | score | document_id | colección | idioma | título / texto |",
                             "| --- | --- | --- | --- | --- | --- |"])
            for rank, hit in enumerate(hits[:k], 1):
                title = hit.get("title") or re.sub(
                    r"@[\w]+", "@usuario", hit.get("text", ""))[:80]
                fields = [rank, hit["score"], hit["document_id"],
                          hit.get("collection_name", name), hit.get("language", ""), title]
                markdown.append("| " + " | ".join(_cell(value) for value in fields) + " |")
            markdown.append("")

    similarities = []
    texts = [pair[field] for pair in PAIRS for field in ("izquierda", "derecha")]
    for encoder in encoders:
        prefix = embeddings.ENCODERS[encoder][1]
        variants = [(prefix, embeddings.embed(texts, kind="query", encoder=encoder))]
        if encoder == "e5small":
            variants.append(("", _embed_raw(texts, encoder)))
        for prefix, vectors in variants:
            for index, pair in enumerate(PAIRS):
                similarities.append(dict(par=index, encoder=encoder, prefijo=prefix,
                                         tipo=pair["tipo"],
                                         similitud=cosine(vectors[2 * index], vectors[2 * index + 1])))
    sensitivity = dict(pares=PAIRS, similitudes=similarities,
                       resumen=summarize(similarities), fingerprints=fingerprints)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "vecinos.md").write_text("\n".join(markdown), encoding="utf-8")
    (out / "sensibilidad.json").write_text(
        json.dumps(sensitivity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out / "solapamiento.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["consulta", "idioma", "jaccard_top10", "n_e5", "n_minilm"])
        for query, language in QUERIES:
            left, right = (results[(query, encoder)] for encoder in encoders)
            writer.writerow([query, language, jaccard(left, right), len(left), len(right)])
    return dict(consultas=len(QUERIES), pares=len(PAIRS), encoders=list(encoders),
                out=str(out), k=k, resumen=sensitivity["resumen"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("reports/embeddings"))
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("--k debe ser positivo")
    print(json.dumps(run(args.out, args.k), ensure_ascii=False))


if __name__ == "__main__":
    main()
