"""Evalúa recuperación y exporta resultados trazables por pregunta."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import re
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.metrics import (bootstrap_ci, doc_ranking, dup_rate_at_k, fragment_hit_at_k,
                         hit_at_k, mrr, ndcg_at_k, percentile, recall_at_k)
from app.privacy import mask_handles


KS = (1, 3, 5, 10)
SEED = 42
METRICS = ("hit", "recall", "mrr", "ndcg", "fragment_hit", "dup_rate")
PER_QUESTION_COLUMNS = [
    "condition", "backend", "chunker", "encoder", "one_per_doc", "filtered",
    "qid", "split", "type", "language", "collection_target", "k", "hit",
    "recall", "mrr", "ndcg", "fragment_hit", "dup_rate", "latency_ms"]
SUMMARY_COLUMNS = ["condition", "k", "metric", "n", "mean", "ci_low", "ci_high"]


def retrieve(*args, **kwargs):
    from app.retrieval import retrieve as implementation
    return implementation(*args, **kwargs)


def conditions_for(groups):
    conditions = []

    def add(group, backend, chunker="tok256", encoder="e5small",
            one_per_doc=False, filtered=False):
        name = f"{group}_{backend}_{chunker}_{encoder}"
        if one_per_doc:
            name += "_one_per_doc"
        if group == "filters":
            name += "_filtered" if filtered else "_unfiltered"
        conditions.append(dict(condition=name, group=group, backend=backend,
                               chunker=chunker, encoder=encoder,
                               one_per_doc=one_per_doc, filtered=filtered))

    for group in dict.fromkeys(groups):
        if group == "main":
            for backend in ("solr", "milvus", "hybrid"):
                add(group, backend)
        elif group == "ablations":
            for backend in ("milvus", "solr"):
                for chunker in ("sent256", "tok128"):
                    add(group, backend, chunker)
            add(group, "milvus", encoder="minilm")
            add(group, "milvus", chunker="tok128", encoder="minilm")
            add(group, "milvus", one_per_doc=True)
        elif group == "filters":
            for backend in ("milvus", "solr"):
                for filtered in (False, True):
                    add(group, backend, filtered=filtered)
        elif group == "10k":
            for backend in ("solr", "milvus"):
                add(group, backend, "tok256_10k")
    return conditions


def eligible(question, condition):
    return (condition["group"] != "filters"
            or question.get("collection_target") in {"bluesky_posts", "url_crawls"})


def retrieval_args(question, condition):
    return dict(backend=condition["backend"], k=10, chunker=condition["chunker"],
                encoder=condition["encoder"], one_per_doc=condition["one_per_doc"],
                filters={"collection_name": question["collection_target"]}
                if condition["filtered"] else None)


def question_scores(question, chunks, k):
    grades, spans = {}, defaultdict(list)
    for item in question["relevant"]:
        doc = item["document_id"]
        grades[doc] = max(grades.get(doc, 0), item["grade"])
        spans[doc].append(tuple(item["evidence_span"]))
    relevant = set(grades)
    ranked = doc_ranking(chunks, relevant)
    return dict(hit=hit_at_k(ranked, relevant, k),
                recall=recall_at_k(ranked, relevant, k), mrr=mrr(ranked, relevant, k),
                ndcg=ndcg_at_k(ranked, grades, k),
                fragment_hit=fragment_hit_at_k(chunks, spans, k),
                dup_rate=dup_rate_at_k(chunks, k))


def csv_value(value):
    return "" if isinstance(value, float) and not math.isfinite(value) else value


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: csv_value(row[key]) for key in columns} for row in rows)


def summarize(rows, group_by=None):
    buckets = defaultdict(list)
    for row in rows:
        key = (row["condition"], row["k"])
        if group_by:
            key += (row[group_by],)
        buckets[key].append(row)
    output = []
    for key, bucket in buckets.items():
        for metric in METRICS:
            values = [row[metric] for row in bucket if math.isfinite(row[metric])]
            low, high = bootstrap_ci(values, seed=SEED)
            summary = dict(condition=key[0], k=key[1], metric=metric, n=len(values),
                           mean=sum(values) / len(values) if values else math.nan,
                           ci_low=low, ci_high=high)
            if group_by:
                summary.update(group_by=group_by, group=key[2])
            output.append(summary)
    return output


def mask_text(text):
    return mask_handles(" ".join(str(text).split()))


def markdown_text(text):
    return mask_text(text).replace("|", "\\|").replace("<", "&lt;").replace(">", "&gt;")


def inspection_text(questions, get_chunks):
    lines = ["# Inspección de recuperación", ""]
    for question in questions:
        lines.extend([f"## {markdown_text(question['qid'])}: {markdown_text(question['question'])}", ""])
        relevant = {item["document_id"] for item in question["relevant"]}
        for backend in ("solr", "milvus"):
            lines.extend([f"### {backend}", "", "| rank | document_id | título | fragmento | relevante |",
                          "| --- | --- | --- | --- | --- |"])
            for index, chunk in enumerate(get_chunks(question, backend)[:5], 1):
                matched = bool(relevant & {chunk["document_id"], *chunk.get("provenance_ids", [])})
                snippet = mask_text(chunk["text"])[:160]
                cells = [str(chunk.get("rank", index)), markdown_text(chunk["document_id"]),
                         markdown_text(chunk.get("title", "")), markdown_text(snippet),
                         "✓" if matched else ""]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
    return "\n".join(lines)


def run(gold, split, out, groups):
    questions = [json.loads(line) for line in gold.read_text(encoding="utf-8-sig").splitlines()
                 if line.strip()]
    qids = [q["qid"] for q in questions]
    if len(qids) != len(set(qids)):
        raise ValueError("Los qid del gold deben ser únicos.")
    selected = [q for q in questions if split == "all" or q["split"] == split]
    inspection = sorted((q for q in questions if q["split"] == "dev" and
                         q["type"] in {"directa", "comparativa", "temporal"}),
                        key=lambda q: q["qid"])[:6]
    conditions = conditions_for(groups)
    jobs = [(condition, [q for q in selected if eligible(q, condition)])
            for condition in conditions]
    inspection_conditions = conditions_for(["main"])[:2]
    warmups = {}
    for condition, batch in jobs:
        if batch:
            warmups.setdefault(condition["backend"], (batch[0], condition))
    if inspection:
        for condition in inspection_conditions:
            warmups.setdefault(condition["backend"], (inspection[0], condition))
    for question, condition in warmups.values():
        retrieve("Consulta de calentamiento", **retrieval_args(question, condition))

    cache, per_question, latencies = {}, [], []
    observed_collections = Counter()

    def get_chunks(question, condition):
        kwargs = retrieval_args(question, condition)
        key = (question["qid"], condition["condition"])
        if key not in cache:
            start = time.perf_counter()
            chunks = retrieve(question["question"], **kwargs)
            elapsed = (time.perf_counter() - start) * 1000
            cache[key] = (chunks, elapsed)
            observed_collections.update(chunk["collection_name"] for chunk in chunks)
        return cache[key]

    for condition, batch in jobs:
        for index, question in enumerate(batch):
            chunks, elapsed = get_chunks(question, condition)
            latencies.append(dict(condition=condition["condition"],
                                  phase="cold" if index == 0 else "warm", latency_ms=elapsed))
            for k in KS:
                row = {key: condition[key] for key in
                       ("condition", "backend", "chunker", "encoder", "one_per_doc", "filtered")}
                row.update({key: question.get(key, "") for key in
                            ("qid", "split", "type", "language", "collection_target")})
                row.update(k=k, latency_ms=elapsed, **question_scores(question, chunks, k))
                per_question.append(row)

    def inspection_chunks(question, backend):
        condition = next(c for c in inspection_conditions if c["backend"] == backend)
        return get_chunks(question, condition)[0]

    inspection_md = inspection_text(inspection, inspection_chunks)
    latency_rows = []
    for condition in conditions:
        for phase in ("cold", "warm"):
            values = [r["latency_ms"] for r in latencies
                      if r["condition"] == condition["condition"] and r["phase"] == phase]
            latency_rows.append(dict(condition=condition["condition"], phase=phase, n=len(values),
                                     p50_ms=percentile(values, 0.5), p95_ms=percentile(values, 0.95)))
    all_conditions = conditions + (inspection_conditions if inspection else [])
    collections = {
        "milvus": sorted({f"c_{c['encoder']}_{c['chunker']}" for c in all_conditions
                          if c["backend"] in {"milvus", "hybrid"}}),
        "solr_chunkers": sorted({c["chunker"] for c in all_conditions
                                 if c["backend"] in {"solr", "hybrid"}})}
    packages = {}
    for package in ("pymilvus", "sentence-transformers", "requests", "numpy", "pandas"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    config = dict(date=datetime.now(timezone.utc).isoformat(), seed=SEED,
                  gold=str(gold), split=split, k=list(KS), conditions=conditions,
                  package_versions=packages, python_version=platform.python_version(),
                  collections=collections, inspection_qids=[q["qid"] for q in inspection],
                  bootstrap=dict(n=1000, confidence=0.95, method="percentile"),
                  latency_policy="Un calentamiento descartado por motor; primera consulta medida = cold.",
                  counts=dict(gold_questions=len(questions), questions=len(selected),
                              ranking_questions=sum(bool(q["relevant"]) for q in selected),
                              measured_pairs=len(latencies), retrieval_calls=len(cache),
                              warmup_calls=len(warmups), per_question_rows=len(per_question),
                              observed_chunks_by_collection=dict(observed_collections)))
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "per_question.csv", PER_QUESTION_COLUMNS, per_question)
    write_csv(out / "metrics.csv", SUMMARY_COLUMNS, summarize(per_question))
    write_csv(out / "metrics_by_group.csv", ["group_by", "group"] + SUMMARY_COLUMNS,
              [row for group in ("type", "language", "collection_target")
               for row in summarize(per_question, group)])
    write_csv(out / "latency.csv", ["condition", "phase", "n", "p50_ms", "p95_ms"], latency_rows)
    (out / "inspeccion.md").write_text(inspection_md, encoding="utf-8")
    (out / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=Path("eval/gold.jsonl"))
    parser.add_argument("--split", choices=("all", "dev", "test"), default="all")
    parser.add_argument("--out", type=Path, default=Path("reports/retrieval"))
    parser.add_argument("--conditions", default="main,ablations,filters,10k")
    args = parser.parse_args(argv)
    groups = [group.strip() for group in args.conditions.split(",")]
    if not groups or set(groups) - {"main", "ablations", "filters", "10k"}:
        parser.error("Condiciones válidas: main,ablations,filters,10k.")
    run(args.gold, args.split, args.out, groups)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
