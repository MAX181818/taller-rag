"""Construye fragmentos deduplicados con procedencia y estadísticas."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from app.chunking import CHUNKERS, chunk_id
from scripts.make_sample import normalize_created_at


COLUMNS = ["document_id", "text", "title", "url", "collection_name", "source",
           "language_base", "created_at"]


def load_documents(input_path, sample_path, extra_path=None):
    """Lee únicamente campos necesarios y filas canónicas de la muestra."""
    sample = pd.read_csv(sample_path, dtype=str, keep_default_na=False)
    if sample.document_id.duplicated().any():
        raise ValueError("La muestra contiene document_id repetidos")
    ids = sample.document_id.tolist()
    frame = pd.read_parquet(input_path, columns=COLUMNS, filters=[("document_id", "in", ids)])
    if frame.document_id.duplicated().any():
        raise ValueError("El Parquet contiene document_id repetidos en la muestra")
    missing = sorted(set(ids) - set(frame.document_id))
    if missing:
        raise ValueError(f"Faltan documentos de la muestra: {missing}")
    provenance = dict(zip(ids, sample.provenance_ids))
    documents = frame.astype(object).where(frame.notna(), None).to_dict("records")
    for doc in documents:
        doc["provenance_ids"] = sorted(set(provenance[doc["document_id"]].split("|")) - {""})
    if extra_path:
        with Path(extra_path).open(encoding="utf-8") as source:
            documents.extend(json.loads(line) for line in source if line.strip())
    all_ids = [doc["document_id"] for doc in documents]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Los documentos adicionales repiten un document_id")
    return sorted(documents, key=lambda doc: doc["document_id"])


def _normalized(text):
    return " ".join(text.lower().split())


def _percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def build_chunks(documents, tok, chunker):
    """Fusiona textos equivalentes entre documentos y conserva apariciones."""
    segment = CHUNKERS[chunker]
    groups = defaultdict(list)
    counts = defaultdict(list)
    originals = {}
    raw_count = 0
    single_count = 0
    for doc in sorted(documents, key=lambda item: item["document_id"]):
        document_id = doc["document_id"]
        if document_id in originals:
            raise ValueError(f"document_id repetido: {document_id}")
        original = doc.get("text") or ""
        originals[document_id] = original
        spans = segment(original, tok)
        counts[doc.get("collection_name") or "und"].append(len(spans))
        single_count += len(spans) == 1
        for span in spans:
            text = original[span.start:span.end]
            occurrence = dict(document_id=document_id, start_char=span.start, end_char=span.end)
            record = dict(chunk_id=chunk_id(document_id, chunker, span.start, span.end),
                          document_id=document_id,
                          provenance_ids=sorted(set(doc.get("provenance_ids") or []) | {document_id}),
                          chunker=chunker, start_char=span.start, end_char=span.end,
                          text=text, title=doc.get("title"), url=doc.get("url"),
                          collection_name=doc.get("collection_name"), source=doc.get("source"),
                          language=(doc.get("language_base") or "").strip() or "und",
                          created_ts=normalize_created_at(doc.get("created_at")), occurrences=[occurrence])
            groups[_normalized(text)].append(record)
            raw_count += 1
    records = []
    for group in groups.values():
        if len({record["document_id"] for record in group}) > 1:
            representative = group[0]
            representative["occurrences"] = [occ for record in group for occ in record["occurrences"]]
            representative["provenance_ids"] = sorted({pid for record in group for pid in record["provenance_ids"]})
            records.append(representative)
        else:
            records.extend(group)
    records.sort(key=lambda record: (record["document_id"], record["start_char"], record["end_char"]))
    if records:
        encoded = tok([record["text"] for record in records], add_special_tokens=False,
                      return_attention_mask=False, truncation=False, verbose=False)
        for record, tokens in zip(records, encoded["input_ids"]):
            record["n_tokens"] = len(tokens)
    lengths = [record["n_tokens"] for record in records]
    stats = dict(
        n_documents=len(originals), n_chunks=len(records), chunks_before_dedup=raw_count,
        tokens_per_chunk=dict(p50=_percentile(lengths, 0.5), p95=_percentile(lengths, 0.95), max=max(lengths, default=0)),
        chunks_per_document_by_collection={name: dict(mean=sum(values) / len(values), p50=_percentile(values, 0.5), max=max(values))
                                           for name, values in sorted(counts.items())},
        single_chunk_documents_pct=100 * single_count / len(originals) if originals else 0.0,
        chunks_merged=raw_count - len(records),
        offset_mismatches=sum(record["text"] != originals[record["document_id"]][record["start_char"]:record["end_char"]]
                              for record in records),
    )
    return records, stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--chunker", choices=CHUNKERS, required=True)
    parser.add_argument("--extra")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    documents = load_documents(args.input, args.sample, args.extra)
    tok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small", local_files_only=True)
    records, stats = build_chunks(documents, tok, args.chunker)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    stats_path = Path("reports/chunking/stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    all_stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    all_stats[args.chunker] = stats
    stats_path.write_text(json.dumps(all_stats, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(chunker=args.chunker, **stats), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
