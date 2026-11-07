"""Perfil reproducible del corpus: integridad, calidad y sesgos de recolección."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlsplit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts.make_sample import make_sample, normalize_created_at

EXPECTED_SHA = "f4a669f17eaedd357f72b70c2192e685b6d534a94f6ffaeadbced7290b181728"
DISTRIBUTIONS = ["collection_name", "source", "source_type", "language_base", "keyword_family"]
PRIVATE = {"author_id", "author_name", "author_handle"}


def file_sha(path):
    """Calcula SHA-256 mediante lectura binaria por bloques."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def counts(series):
    return {str(k): int(v) for k, v in series.fillna("(nulo)").value_counts(dropna=False).items()}


def summary(series):
    return {str(k): float(v) for k, v in series.describe(percentiles=[.5, .9, .95, .99]).items()}


def date_format(value):
    if not isinstance(value, str) or not value.strip():
        return "vacía o nula"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return "solo fecha"
    if "T" in value:
        return "ISO T con Z" if value.endswith("Z") else "ISO T con offset" if re.search(r"[+-]\d\d:\d\d$", value) else "ISO T sin zona"
    return "fecha con espacio" if " " in value else "otro"


def url_info(value):
    if not isinstance(value, str) or not value.strip():
        return "vacía", ""
    try:
        parsed = urlsplit(value.strip())
        return ("http" if parsed.scheme.lower() in ("http", "https") else "esquema no http",
                (parsed.hostname or "").lower())
    except ValueError:
        return "inválida", ""


def manifest_comparison(path, df, report, input_path):
    """Contrasta cifras del manifiesto sin publicar conteos de autores."""
    if not Path(path).exists():
        return {"status": "Descarga fallida con gdown: red no accesible (WinError 10051); no se leyó el manifiesto.",
                "figures": None, "comparison": []}
    manifest = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    nonempty = {c: int(df[c].map(is_nonempty).sum()) for c in df.columns if c not in PRIVATE}
    measured = {"rows": report["rows"], "columns": report["fields"],
                "output_sha256": report["sha256_before"], "output_bytes": Path(input_path).stat().st_size,
                "nonempty_counts": nonempty,
                "unique_texts": report["unique_text_hashes"], "duplicate_text_rows": report["dup_text_rows"],
                "distributions": {c: counts(df[c]) for c in manifest.get("distributions", {}) if c in df and c not in PRIVATE},
                "validation": {"ids_unique": report["document_id_unique"],
                               "required_fields_nonempty": all(nonempty[c] == len(df) for c in ("document_id", "text", "collection_name"))}}
    comparisons = []

    def compare(expected, actual, prefix=""):
        for key, value in expected.items():
            if key in PRIVATE:
                continue
            name = f"{prefix}.{key}" if prefix else key
            observed = actual.get(key) if isinstance(actual, dict) else None
            if isinstance(value, dict):
                compare(value, observed, name)
            elif isinstance(value, (int, float)) or name == "output_sha256":
                comparisons.append({"figure": name, "manifest": value, "measured": observed,
                                    "matches": value == observed if observed is not None else None})

    compare(manifest, measured)
    return {"status": "Manifiesto leído como JSON. Conteos de autores omitidos por privacidad; solo se publican sus porcentajes de nulos. Las validaciones de origen requieren el archivo fuente y no son verificables con el Parquet de salida.",
            "comparison": comparisons}


def is_nonempty(value):
    """Considera vacíos los nulos, cadenas sin contenido y listas vacías."""
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, np.ndarray, dict)):
        return len(value) > 0
    return not bool(pd.isna(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Parquet de entrada")
    parser.add_argument("--out", required=True, help="Directorio de informes")
    parser.add_argument("--manifest", default=r"D:\taller-rag-trabajo\corpus_home\manifiesto.md", help="Manifiesto JSON de referencia")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    before = file_sha(args.input)
    parquet = pq.ParquetFile(args.input)
    df = parquet.read().to_pandas()
    rows = len(df)
    text = df["text"].fillna("")
    digest = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    calculated = text.map(digest)
    mismatches = int(calculated.ne(df.text_sha256).sum())
    variants = {"UTF-8 literal": mismatches}
    if mismatches:
        for name, transform in [("strip", str.strip), ("NFC", lambda s: unicodedata.normalize("NFC", s)),
                                ("strip + NFC", lambda s: unicodedata.normalize("NFC", s.strip()))]:
            variants[name] = int(text.map(lambda s: digest(transform(s))).ne(df.text_sha256).sum())
    words, chars = text.map(lambda s: len(s.split())), text.str.len()
    report = {
        "sha256_before": before, "sha256_expected": EXPECTED_SHA,
        "rows": rows, "fields": len(parquet.schema_arrow), "parquet_leaf_columns": len(parquet.schema),
        "nested_fields": {f.name: {"type": str(f.type),
                                  "leaf_paths": [parquet.schema.column(i).path for i in range(len(parquet.schema))
                                                 if parquet.schema.column(i).path.startswith(f.name + ".")]}
                          for f in parquet.schema_arrow if f.name not in PRIVATE and f.type.num_fields},
        "schema": {f.name: str(f.type) for f in parquet.schema_arrow if f.name not in PRIVATE},
        "null_percent": {c: float(df[c].isna().mean() * 100) for c in df.columns},
        "document_id_unique": bool(df.document_id.is_unique),
        "document_id_nulls": int(df.document_id.isna().sum()),
        "dup_text_rows": int(df.text_sha256.duplicated().sum()),
        "unique_text_hashes": int(df.text_sha256.nunique()),
        "empty_text_rows": int(text.str.strip().eq("").sum()),
        "text_sha256_mismatches": mismatches, "text_sha256_variant_mismatches": variants,
        "word_count_mismatches": int(words.ne(df.word_count).sum()),
        "char_count_mismatches": int(chars.ne(df.char_count).sum()),
        "count_method": "Palabras: len(text.split()); caracteres: len(text); nulos tratados como cadena vacía solo al medir.",
        "distributions": {c: counts(df[c]) for c in DISTRIBUTIONS},
        "collection_query_distribution": counts(df.collection_query),
        "lengths": {}, "dates": {}, "urls": {}, "sample_comparison": {},
    }
    report["manifest"] = manifest_comparison(args.manifest, df, report, args.input)
    for collection, indices in df.groupby("collection_name", dropna=False).groups.items():
        report["lengths"][str(collection)] = {"words": summary(words.loc[indices]), "chars": summary(chars.loc[indices])}
    for name, values in [("palabras", words), ("caracteres", chars)]:
        fig, ax = plt.subplots(figsize=(10, 5))
        bins = np.unique(np.r_[0, np.geomspace(1, max(2, values.max()), 60)])
        for collection, indices in df.groupby("collection_name", dropna=False).groups.items():
            ax.hist(values.loc[indices], bins=bins, histtype="step", linewidth=1.5, label=str(collection))
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("log")
        ax.set(xlabel=f"Longitud en {name}", ylabel="Documentos (escala logarítmica)", title=f"Distribución de {name} por colección")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"histograma_{name}.png", dpi=140)
        plt.close(fig)
    for column in ["created_at", "synced_at"]:
        epochs = df[column].map(normalize_created_at)
        valid = epochs.ne(-1)
        dates = pd.to_datetime(epochs[valid], unit="s", utc=True)
        formats = df[column].map(date_format)
        report["dates"][column] = {
            "min_utc": dates.min().isoformat() if len(dates) else None,
            "max_utc": dates.max().isoformat() if len(dates) else None,
            "unparseable_rows": int((~valid).sum()), "unparseable_percent": float((~valid).mean() * 100),
            "by_year": counts(dates.dt.year),
            "by_month": counts(dates.dt.strftime("%Y-%m")),
            "by_format": {fmt: {"rows": int(mask.sum()), "unparseable_percent": float((~valid[mask]).mean() * 100)}
                          for fmt in sorted(formats.unique()) for mask in [formats.eq(fmt)]}}
    for column in ["url", "final_url"]:
        info = df[column].map(url_info)
        kinds = info.map(lambda x: x[0])
        domains = info.map(lambda x: x[1])
        report["urls"][column] = {"quality": counts(kinds),
                                   "top_20_domains": {}}
        report["urls"][column]["top_20_domains"] = {str(k): int(v) for k, v in domains[domains.ne("")].value_counts().head(20).items()}
    original, final = df.url.fillna(""), df.final_url.fillna("")
    both = original.str.strip().ne("") & final.str.strip().ne("")
    report["urls"]["comparison"] = {"both_present": int(both.sum()),
        "same_when_both_present": int((both & original.eq(final)).sum()),
        "different_when_both_present": int((both & original.ne(final)).sum()),
        "different_domain_when_both_present": int(sum(url_info(a)[1] != url_info(b)[1] for a, b in zip(original[both], final[both])))}
    eligible = df.loc[text.str.strip().ne(""), ["document_id", "text_sha256", "collection_name"]].sort_values("document_id").drop_duplicates("text_sha256")
    report["sampling_population"] = {"rows": len(eligible), "by_collection": counts(eligible.collection_name)}
    small, sample = make_sample(df, 700), make_sample(df, 10000)
    selected = df.set_index("document_id").loc[sample.document_id]
    report["samples"] = {"seed": 42, "nested": set(small.document_id) <= set(sample.document_id),
        "700": counts(small.collection_name), "10000": counts(sample.collection_name),
        "method": "Deduplicación por hash, ID mínimo, procedencias ordenadas; permutación fija seed 42 por estrato; cuotas proporcionales Webster monótonas."}
    lines = ["# Perfilado del corpus", "", f"Filas: {rows}. Duplicados por hash (filas posteriores a la primera): {report['dup_text_rows']}.",
             "", "## Integridad y esquema", "",
             f"Campos: {report['fields']}; columnas hoja Parquet: {report['parquet_leaf_columns']}. No hay índice pandas.",
             "Los campos anidados explican la diferencia: keyword_matches es una lista de cadenas; top_terms es una lista de estructuras con term y count.",
             json.dumps(report["nested_fields"], ensure_ascii=False),
             f"IDs únicos: {report['document_id_unique']}; IDs nulos: {report['document_id_nulls']}; textos vacíos: {report['empty_text_rows']}.",
             f"Discrepancias text_sha256: {mismatches}; variantes (discrepancias): {json.dumps(variants, ensure_ascii=False)}.",
             f"Discrepancias word_count: {report['word_count_mismatches']}; char_count: {report['char_count_mismatches']}.", report["count_method"],
             "", "| Columna | Tipo físico | % nulos |", "|---|---|---:|"]
    for column in df.columns:
        lines.append(f"| {column} | {report['schema'].get(column, 'dato personal: solo % nulos')} | {report['null_percent'][column]:.4f} |")
    lines += ["", "## Manifiesto vs medido", "", report["manifest"]["status"],
              "", "| Cifra | Manifiesto | Medido | Coincide |", "|---|---|---|---|"]
    for item in report["manifest"]["comparison"]:
        measured = item["measured"] if item["measured"] is not None else "No verificable con el archivo de salida"
        match = "No verificable" if item["matches"] is None else "sí" if item["matches"] else "no"
        lines.append(f"| {item['figure']} | {item['manifest']} | {measured} | {match} |")
    lines += [
              "", "## Distribuciones del corpus y muestra 10k", "",
              "Porcentajes sobre todas las filas del corpus y sobre los 10000 documentos canónicos seleccionados; Δ = muestra − corpus, en puntos porcentuales. Los nulos forman una categoría.", "",
              "| Variable | Categoría | Corpus n | Muestra n | Corpus % | Muestra % | Δ pp |", "|---|---|---:|---:|---:|---:|---:|"]
    for column in DISTRIBUTIONS:
        full, sampled = counts(df[column]), counts(selected[column])
        comparison = []
        for category in sorted(full.keys() | sampled.keys()):
            a, b = full.get(category, 0), sampled.get(category, 0)
            delta = b / len(sample) * 100 - a / rows * 100
            comparison.append({"category": category, "corpus_n": a, "sample_n": b, "corpus_percent": a / rows * 100,
                               "sample_percent": b / len(sample) * 100, "difference_pp": delta})
            lines.append(f"| {column} | {category.replace('|', '/')} | {a} | {b} | {a/rows*100:.4f} | {b/len(sample)*100:.4f} | {delta:+.4f} |")
        report["sample_comparison"][column] = comparison
    lines += ["", "## Tamaños de muestras por colección", "", "| Colección | 700 | 10000 |", "|---|---:|---:|"]
    for name in report["distributions"]["collection_name"]:
        lines.append(f"| {name} | {report['samples']['700'].get(name, 0)} | {report['samples']['10000'].get(name, 0)} |")
    lines.append(f"Universo elegible deduplicado: {len(eligible)} documentos; por colección: {report['sampling_population']['by_collection']}. La tolerancia de ±1 punto porcentual se evalúa sobre este universo.")
    lines += [f"Anidamiento: {report['samples']['nested']}. {report['samples']['method']}",
              "La estratificación es proporcional al universo deduplicado no vacío, que difiere del corpus bruto; otras variables pueden cambiar.",
              "", "## Longitudes por colección", "", "| Colección | Medida | Mínimo | Mediana | Media | P95 | P99 | Máximo |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, measures in report["lengths"].items():
        for measure, stat in measures.items():
            lines.append(f"| {name} | {'palabras' if measure == 'words' else 'caracteres'} | {stat['min']:.0f} | {stat['50%']:.1f} | {stat['mean']:.2f} | {stat['95%']:.1f} | {stat['99%']:.1f} | {stat['max']:.0f} |")
    lines += ["", "![Histograma de palabras](histograma_palabras.png)", "![Histograma de caracteres](histograma_caracteres.png)",
              "", "## Fechas y calidad de URL", "", "Los siguientes agregados incluyen rangos UTC, años, meses, formatos y porcentaje no parseable; para URL incluyen vacías, esquemas y dominios top-20.",
              "", "```json", json.dumps({"dates": report["dates"], "urls": report["urls"]}, ensure_ascii=False, indent=2), "```",
              "", "## Sesgos de recolección", "",
              "La recolección está guiada por palabras clave: keyword_family y collection_query describen consultas de captura; no son etiquetas semánticas validadas. No deben usarse como verdad de referencia temática."]
    for column, label in [("source", "plataforma/fuente"), ("source_type", "tipo de plataforma"), ("language_base", "idioma")]:
        top = df[column].fillna("(nulo)").value_counts()
        lines.append(f"Concentración de {label}: {top.index[0]} representa {top.iloc[0]/rows*100:.2f}% de las filas; la cobertura responde a las fuentes disponibles y no a una población representativa.")
    domains = df.url.map(url_info).map(lambda x: x[1])
    top_domains = domains[domains.ne("")].value_counts()
    if len(top_domains):
        lines.append(f"El dominio principal, {top_domains.index[0]}, concentra {top_domains.iloc[0]/rows*100:.2f}% del corpus; los 20 principales concentran {top_domains.head(20).sum()/rows*100:.2f}%. Esto concentra medios y plataformas.")
    month_counts = report["dates"]["created_at"]["by_month"]
    if month_counts:
        month = max(month_counts, key=month_counts.get)
        lines.append(f"Concentración temporal: {month} contiene {month_counts[month]} filas ({month_counts[month]/rows*100:.2f}% del corpus); los periodos disponibles condicionan los resultados.")
    lines += ["Los códigos de idioma son metadatos publicados, sin validación lingüística independiente. La deduplicación cambia el peso relativo de las colecciones y elimina repeticiones, no corrige estos sesgos.",
              "", "## Consultas de captura", "", "Distribución de collection_query (consultas, sin textos ni autores):", "", "```json", json.dumps(report["collection_query_distribution"], ensure_ascii=False, indent=2), "```"]
    after = file_sha(args.input)
    report.update(sha256_after=after, sha256_ok=before == after == EXPECTED_SHA)
    lines += ["", f"SHA-256 antes: {before}", f"SHA-256 después: {after}", f"sha256_ok: {report['sha256_ok']}. El archivo se abrió solo para lectura."]
    (out / "perfil.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "perfil.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["sha256_ok", "rows", "dup_text_rows", "text_sha256_mismatches", "word_count_mismatches", "char_count_mismatches", "samples"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()


