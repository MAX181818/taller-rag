"""Métricas y evaluación reproducible sin motores externos."""
import csv
import json
import math
import socket

import pytest

from app import metrics
from scripts import eval_retrieval as evaluator


def chunk(doc="a", text="Texto", start=0, end=10, provenance=None):
    return dict(document_id=doc, provenance_ids=provenance or [doc],
                start_char=start, end_char=end, text=text,
                collection_name="bluesky_posts", rank=1, score=1.0,
                title="Título @persona")


def test_document_metrics():
    ranked = ["a", "b", "c"]
    assert metrics.hit_at_k(ranked, {"b"}, 1) == 0
    assert metrics.hit_at_k(ranked, {"b"}, 3) == 1
    assert metrics.recall_at_k(ranked, {"b"}, 3) == 1
    assert metrics.recall_at_k(["b", "b"], {"b", "c"}, 3) == 0.5
    assert metrics.mrr(ranked, {"b"}) == 0.5
    assert metrics.mrr(ranked, {"b"}, 1) == 0
    assert metrics.ndcg_at_k(ranked, {"b": 2}, 3) == pytest.approx(0.6309, abs=1e-4)
    assert metrics.ndcg_at_k(["b", "c"], {"b": 2, "c": 1}, 3) == 1
    assert metrics.ndcg_at_k(["b", "b"], {"b": 2}, 3) == 1


@pytest.mark.parametrize("k", [0, -1])
def test_nonpositive_cutoff(k):
    assert metrics.hit_at_k(["a"], {"a"}, k) == 0
    assert metrics.recall_at_k(["a"], {"a"}, k) == 0
    assert metrics.mrr(["a"], {"a"}, k) == 0
    assert metrics.ndcg_at_k(["a"], {"a": 1}, k) == 0
    assert metrics.fragment_hit_at_k([chunk()], {"a": [(0, 1)]}, k) == 0
    assert metrics.dup_rate_at_k([chunk(), chunk()], k) == 0


def test_empty_relevance_is_excluded():
    for metric in (metrics.hit_at_k, metrics.recall_at_k, metrics.mrr):
        assert math.isnan(metric(["a"], set(), 3))
    assert math.isnan(metrics.ndcg_at_k(["a"], {}, 3))
    assert math.isnan(metrics.fragment_hit_at_k([chunk()], {}, 3))
    assert metrics.ndcg_at_k(["a"], {"a": 0}, 3) == 0


def test_provenance_and_unique_document_order():
    chunks = [chunk("x", provenance=["x", "b"]), chunk("b"), chunk("a"), chunk("a")]
    assert metrics.doc_ranking(chunks, {"b"}) == ["b", "a"]
    assert metrics.doc_ranking(chunks, set()) == ["x", "b", "a"]


def test_fragment_overlap_is_half_open_and_checks_all_aliases():
    chunks = [chunk("x", start=10, end=20, provenance=["x", "b"])]
    assert metrics.fragment_hit_at_k(chunks, {"b": [(19, 25)]}, 1) == 1
    assert metrics.fragment_hit_at_k(chunks, {"b": [(20, 25)]}, 1) == 0
    assert metrics.fragment_hit_at_k(chunks, {"b": [(0, 10)]}, 1) == 0
    assert metrics.fragment_hit_at_k(chunks, {"x": [(11, 12)]}, 1) == 1
    assert metrics.fragment_hit_at_k(chunks, {"b": [(11, 11)]}, 1) == 0
    assert metrics.fragment_hit_at_k([chunk("z")] + chunks, {"b": [(19, 25)]}, 1) == 0


def test_duplicate_rate_uses_documents_or_normalized_text():
    chunks = [chunk("a", " Uno  DOS "), chunk("a", "Otro"),
              chunk("b", "uno\ndos"), chunk("c", "Distinto")]
    assert metrics.dup_rate_at_k(chunks, 10) == 0.5
    assert metrics.dup_rate_at_k(chunks, 1) == 0
    assert metrics.dup_rate_at_k([], 3) == 0


def test_bootstrap_is_deterministic_and_handles_empty():
    assert metrics.bootstrap_ci([1, 1, 1]) == (1, 1)
    interval = metrics.bootstrap_ci([0, 1, 0.5], n=100, seed=8)
    assert interval == metrics.bootstrap_ci([0, 1, 0.5], n=100, seed=8)
    assert 0 <= interval[0] <= interval[1] <= 1
    assert all(math.isnan(x) for x in metrics.bootstrap_ci([]))
    with pytest.raises(ValueError):
        metrics.bootstrap_ci([1], n=0)


def question(qid, split="dev", kind="directa", relevant=True, target="bluesky_posts"):
    return dict(qid=qid, split=split, question=f"Pregunta {qid} @persona",
                language="es", type=kind, collection_target=target,
                relevant=[dict(document_id="b", grade=2, evidence_quote="Texto",
                               evidence_span=[0, 5])] if relevant else [],
                expected_abstention=not relevant)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


@pytest.fixture
def fake_retriever(monkeypatch):
    calls = []

    def forbidden(*args, **kwargs):
        raise AssertionError("Acceso de red no permitido")

    def retrieve(query, backend, k, chunker, encoder, filters, one_per_doc):
        calls.append((query, backend, k, chunker, encoder, filters, one_per_doc))
        if "sin" in query:
            return [chunk("z")]
        return [chunk("x", "Texto @secreto", provenance=["x", "b"]), chunk("c")]

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(evaluator, "retrieve", retrieve)
    return calls


def run_eval(tmp_path, rows, conditions="main", split="all"):
    gold = tmp_path / "gold.jsonl"
    gold.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    out = tmp_path / "salida"
    assert evaluator.main(["--gold", str(gold), "--split", split,
                           "--out", str(out), "--conditions", conditions]) == 0
    return out


def test_evaluation_columns_exclusion_and_single_retrieval(tmp_path, fake_retriever):
    out = run_eval(tmp_path, [question("q01"), question("sin", kind="sin_respuesta", relevant=False)])
    rows = read_csv(out / "per_question.csv")
    assert list(rows[0]) == [
        "condition", "backend", "chunker", "encoder", "one_per_doc", "filtered",
        "qid", "split", "type", "language", "collection_target", "k", "hit",
        "recall", "mrr", "ndcg", "fragment_hit", "dup_rate", "latency_ms"]
    assert len(rows) == 24
    assert {row["k"] for row in rows} == {"1", "3", "5", "10"}
    assert all(row["hit"] == "" and row["mrr"] == "" for row in rows if row["qid"] == "sin")
    assert all(float(row["hit"]) == 1 for row in rows if row["qid"] == "q01")
    summary = read_csv(out / "metrics.csv")
    hits = [r for r in summary if r["metric"] == "hit"]
    assert len(hits) == 12
    assert all(float(r["mean"]) == 1 and int(r["n"]) == 1 for r in hits)
    groups = read_csv(out / "metrics_by_group.csv")
    assert {r["group_by"] for r in groups} == {"type", "language", "collection_target"}
    assert all(r["n"] == "0" and r["mean"] == "" for r in groups
               if r["group"] == "sin_respuesta" and r["metric"] == "hit")
    assert len(fake_retriever) == 9  # Tres calentamientos y seis consultas medidas.
    assert all(call[2] == 10 for call in fake_retriever)
    latency = read_csv(out / "latency.csv")
    assert len(latency) == 6
    assert {r["phase"] for r in latency} == {"cold", "warm"}
    assert all(r["n"] == "1" for r in latency)
    inspection = (out / "inspeccion.md").read_text(encoding="utf-8")
    assert "q01" in inspection and "✓" in inspection
    assert "@secreto" not in inspection and "@persona" not in inspection
    config = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
    assert config["seed"] == 42 and config["counts"]["questions"] == 2
    assert "c_e5small_tok256" in config["collections"]["milvus"]
    assert config["package_versions"] and config["date"]


def test_all_conditions_and_split(tmp_path, fake_retriever):
    out = run_eval(tmp_path, [question("q01", split="test"),
                             question("q02", split="test", target="url_crawls"),
                             question("q03", split="test", target="otra"),
                             question("sin", relevant=False, kind="sin_respuesta")],
                   "main,ablations,filters,10k", "test")
    rows = read_csv(out / "per_question.csv")
    assert {r["qid"] for r in rows} == {"q01", "q02", "q03"}
    assert len({r["condition"] for r in rows}) == 15
    assert len(rows) == (11 * 3 + 4 * 2) * 4
    calls = [call for call in fake_retriever if call[0].startswith("Pregunta")]
    assert len(calls) == 41
    assert any(c[3] == "tok256_10k" and c[1] == "solr" for c in calls)
    assert any(c[4] == "minilm" and c[1] == "milvus" for c in calls)
    assert any(c[6] for c in calls)
    assert {c[5]["collection_name"] for c in calls if c[5]} == {"bluesky_posts", "url_crawls"}
    assert all(c[4] == "e5small" for c in calls if c[1] == "solr")


def test_inspection_fixed_six_dev_questions_and_cache(tmp_path, fake_retriever):
    rows = [question(f"q{i:02}", kind=["directa", "comparativa", "temporal"][i % 3])
            for i in range(8, 0, -1)]
    out = run_eval(tmp_path, rows, "main", "dev")
    inspection = (out / "inspeccion.md").read_text(encoding="utf-8")
    assert all(f"q{i:02}" in inspection for i in range(1, 7))
    assert "q07" not in inspection and "q08" not in inspection
    assert len(fake_retriever) == 27


def test_latency_separates_first_measured_query_and_discards_warmup(
        tmp_path, fake_retriever, monkeypatch):
    timestamps = iter([0, 0.1, 0.1, 0.3, 0.3, 0.6, 0.6, 1.0, 1.0, 1.5, 1.5, 2.1])
    monkeypatch.setattr(evaluator.time, "perf_counter", lambda: next(timestamps))
    out = run_eval(tmp_path, [question("q01"), question("q02")])
    rows = read_csv(out / "latency.csv")
    assert [float(row["p50_ms"]) for row in rows] == pytest.approx(
        [100, 200, 300, 400, 500, 600])
    assert [row["phase"] for row in rows] == ["cold", "warm"] * 3
    per_question = read_csv(out / "per_question.csv")
    assert [float(row["latency_ms"]) for row in per_question[:4]] == pytest.approx([100] * 4)


def test_empty_split_writes_headers_without_measuring(tmp_path, fake_retriever):
    out = run_eval(tmp_path, [question("sin", kind="sin_respuesta", relevant=False)], split="test")
    assert read_csv(out / "per_question.csv") == []
    assert read_csv(out / "metrics.csv") == []
    assert fake_retriever == []


def test_invalid_conditions_are_rejected(tmp_path, fake_retriever):
    with pytest.raises(SystemExit) as error:
        evaluator.main(["--conditions", "desconocida"])
    assert error.value.code == 2
