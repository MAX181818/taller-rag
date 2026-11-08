"""Métricas de documentos y fragmentos sin dependencias externas."""
import math
import random


def doc_ranking(chunks: list[dict], relevant: set[str]) -> list[str]:
    """Resuelve procedencias y conserva la primera aparición de cada documento."""
    ranked, seen = [], set()
    for chunk in chunks:
        document = next((doc for doc in chunk.get("provenance_ids", [])
                         if doc in relevant), chunk["document_id"])
        if document not in seen:
            seen.add(document)
            ranked.append(document)
    return ranked


def hit_at_k(ranked, relevant, k) -> float:
    if not relevant:
        return math.nan
    return float(bool(set(ranked[:max(0, k)]) & relevant))


def recall_at_k(ranked, relevant, k) -> float:
    if not relevant:
        return math.nan
    return len(set(ranked[:max(0, k)]) & relevant) / len(relevant)


def mrr(ranked, relevant, k=10) -> float:
    if not relevant:
        return math.nan
    return next((1.0 / rank for rank, doc in enumerate(ranked[:max(0, k)], 1)
                 if doc in relevant), 0.0)


def ndcg_at_k(ranked, grades: dict[str, int], k) -> float:
    if not grades:
        return math.nan
    ideal = sum((2 ** grade - 1) / math.log2(i + 2)
                for i, grade in enumerate(sorted(grades.values(), reverse=True)[:max(0, k)]))
    if not ideal:
        return 0.0
    seen, dcg = set(), 0.0
    for i, doc in enumerate(ranked[:max(0, k)]):
        if doc not in seen:
            dcg += (2 ** grades.get(doc, 0) - 1) / math.log2(i + 2)
            seen.add(doc)
    return dcg / ideal


def fragment_hit_at_k(chunks, relevant_spans: dict[str, list[tuple[int, int]]], k) -> float:
    if not relevant_spans:
        return math.nan
    for chunk in chunks[:max(0, k)]:
        documents = {chunk["document_id"], *chunk.get("provenance_ids", [])}
        for document in documents:
            for start, end in relevant_spans.get(document, []):
                if max(start, chunk["start_char"]) < min(end, chunk["end_char"]):
                    return 1.0
    return 0.0


def dup_rate_at_k(chunks, k) -> float:
    top = chunks[:max(0, k)]
    documents, texts, duplicates = set(), set(), 0
    for chunk in top:
        doc = chunk["document_id"]
        text = " ".join(chunk["text"].lower().split())
        duplicates += int(doc in documents or text in texts)
        documents.add(doc)
        texts.add(text)
    return duplicates / len(top) if top else 0.0


def percentile(values, fraction):
    """Interpola linealmente entre posiciones ordenadas."""
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def bootstrap_ci(values, n=1000, seed=42) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if n <= 0:
        raise ValueError("El número de remuestreos debe ser positivo.")
    rng = random.Random(seed)
    means = [sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n)]
    return percentile(means, 0.025), percentile(means, 0.975)
