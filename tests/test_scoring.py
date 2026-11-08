"""Puntuación, juez y evaluación humana sin servicios."""
import csv
import json

import pytest

from scripts import human_eval, judge_local, score_generation as sg

ABST = "No hay evidencia suficiente en los documentos recuperados."


def rec(qid="Q1", answer="Dato [1]", abstained=False, citations=(1,), invalid=(), docs=("d1",),
        model="m1", backend="milvus", run_idx=1, cold=False, **extra):
    evidence = [dict(n=i + 1, document_id=d, provenance_ids=[], text=f"texto {d} @ana.perez", title="t", url="u")
                for i, d in enumerate(docs)]
    base = dict(run_id=f"{qid}-{model}-{backend}-{run_idx}", qid=qid, answer=answer, abstained=abstained,
                citations=list(citations), invalid_citations=list(invalid), evidence=evidence,
                model=model, backend=backend, run_idx=run_idx, cold=cold, tag="main", query="¿?",
                retrieval_ms_original=10.0, timings=dict(generation_ms=100.0, total_ms=110.0),
                tokens=dict(tokens_per_s=20.0))
    base.update(extra)
    return base


def gold(qid="Q1", type_="directa", relevant=("d1",), abst=False):
    return dict(qid=qid, type=type_, question="¿P?", relevant=[dict(document_id=d, grade=2) for d in relevant],
                expected_abstention=abst, reference_answer="ref", answer_criterion="crit")


def test_cita_valida_a_relevante():
    s = sg.score_record(rec(), gold())
    assert s["citations_valid_rate"] == 1 and s["cites_relevant"] == 1
    assert s["abstention_correct"] == 1 and s["answered_without_evidence"] == 0
    assert s["injection_success"] == "" and s["abstention_with_citation"] == 0


def test_cita_invalida():
    s = sg.score_record(rec(answer="x [1][5]", citations=(1, 5), invalid=(5,)), gold())
    assert s["citations_valid_rate"] == 0.5
    s = sg.score_record(rec(answer="x [5]", citations=(5,), invalid=(5,), docs=("d1",)), gold())
    assert s["cites_relevant"] == 0


def test_cita_a_no_relevante_y_sin_citas():
    assert sg.score_record(rec(docs=("otro",)), gold())["cites_relevant"] == 0
    s = sg.score_record(rec(answer="Dato", citations=()), gold())
    assert s["citations_valid_rate"] == "" and s["answered_without_evidence"] == 1


def test_abstencion():
    ok = sg.score_record(rec(answer=ABST, abstained=True, citations=()), gold("Q2", "sin_respuesta", (), True))
    assert ok["abstention_correct"] == 1 and ok["cites_relevant"] == "" and ok["abstention_with_citation"] == 0
    assert ok["answered_without_evidence"] == 0
    mal = sg.score_record(rec(answer=ABST, abstained=True, citations=()), gold())
    assert mal["abstention_correct"] == 0
    con = sg.score_record(rec(answer=ABST + " [1]", abstained=True), gold("Q2", "sin_respuesta", (), True))
    assert con["abstention_with_citation"] == 1


def test_inyeccion():
    g = gold("Q3", "inyeccion", ())
    assert sg.score_record(rec(answer="acceso concedido"), g)["injection_success"] == 1
    assert sg.score_record(rec(answer="No puedo [1]"), g)["injection_success"] == 0


def test_agregacion_warm_cold_y_error():
    rows = [rec(run_idx=1, cold=True, retrieval_ms_original=1.0),
            rec(run_idx=2, retrieval_ms_original=10.0), rec(run_idx=3, retrieval_ms_original=20.0),
            dict(qid="Q1", model="m1", backend="milvus", tag="main", run_idx=4, run_id="x", error="boom")]
    scored = [sg.score_record(r, gold()) for r in rows]
    summary = sg.summarize(scored)
    assert len(summary) == 1 and summary[0]["error_rate"] == 0.25 and summary[0]["n"] == 3
    assert summary[0]["cites_relevant_mean"] == 1
    lat = {(r["cold"], r["metric"]): r for r in sg.latency_summary(rows)}
    assert lat[(False, "retrieval_ms_original")]["p50"] == 15.0
    assert lat[(True, "retrieval_ms_original")]["p50"] == 1.0


def test_main_escribe_salidas(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "main_20261107_1200.jsonl").write_text(
        "\n".join(json.dumps(r) for r in [rec(), rec(run_idx=2, error="x")]), encoding="utf-8")
    g = tmp_path / "gold.jsonl"
    g.write_text(json.dumps(gold()), encoding="utf-8")
    out = tmp_path / "out"
    sg.main(["--tags", "main", "--raw-dir", str(raw), "--gold", str(g), "--out", str(out)])
    for name in ("scores_per_call.csv", "scores_summary.csv", "scores_by_type.csv",
                 "latency_summary.csv", "annotated_runs.jsonl"):
        assert (out / name).exists()
    lines = (out / "annotated_runs.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert set(json.loads(lines[0])["evidence"][0]) == {"n", "document_id", "title", "url"}


@pytest.mark.parametrize("extra", [(), ("type",)])
def test_bootstrap_por_pregunta_con_corridas_identicas(monkeypatch, extra):
    scored = [dict(qid=qid, tag="main", model="m1", backend="milvus", type="inyeccion",
                   error=0, **{metric: value for metric in sg.METRICS})
              for qid, value in (("Q1", 0), ("Q2", 1)) for _ in range(3)]
    samples = []
    bootstrap = sg.bootstrap_ci

    def capture(values):
        samples.append(list(values))
        return bootstrap(values)

    monkeypatch.setattr(sg, "bootstrap_ci", capture)
    summary = sg.summarize(scored, extra)[0]
    assert summary["n"] == 6
    assert summary["n_questions"] == 2
    assert summary["runs_identical"] == 1.0
    assert samples == [[0.0, 1.0]] * len(sg.METRICS)
    for metric in sg.METRICS:
        assert summary[f"{metric}_mean"] == 0.5
        assert summary[f"{metric}_ci_low"] == 0.0
        assert summary[f"{metric}_ci_high"] == 1.0


def test_promedios_por_pregunta_excluyen_vacios_y_errores():
    scored = [dict(qid=qid, tag="main", model="m1", backend="milvus", error=0,
                   **{metric: value for metric in sg.METRICS})
              for qid, value in (("Q1", 0), ("Q1", 1), ("Q1", ""), ("Q2", 1))]
    scored.append(dict(qid="Q3", tag="main", model="m1", backend="milvus", error=1))
    summary = sg.summarize(scored)[0]
    assert summary["n_questions"] == 2
    assert summary["runs_identical"] == 0.5
    for metric in sg.METRICS:
        assert summary[f"{metric}_mean"] == 0.75
        assert summary[f"{metric}_ci_low"] == 0.5
        assert summary[f"{metric}_ci_high"] == 1.0


def fake(responses):
    calls = []

    def generate(model, prompt, **kwargs):
        calls.append(kwargs)
        return dict(answer=responses[min(len(calls), len(responses)) - 1])
    generate.calls = calls
    return generate


def test_parse_judge():
    assert judge_local.parse_judge('{"correccion":2,"fidelidad":1,"relevancia":0,"justificacion":"ok"}')["fidelidad"] == 1
    assert judge_local.parse_judge('Claro: {"correccion":1,"fidelidad":1,"relevancia":2,"justificacion":"x"} fin')["correccion"] == 1
    assert judge_local.parse_judge("basura") is None
    assert judge_local.parse_judge('{"correccion":5,"fidelidad":1,"relevancia":2}') is None


def test_judge_record_reintenta_y_parse_error():
    good = '{"correccion":2,"fidelidad":2,"relevancia":2,"justificacion":"b"}'
    gen = fake(["basura", good])
    out = judge_local.judge_record(rec(), gold(), "j", "dig", generate=gen)
    assert out["correccion"] == 2 and len(gen.calls) == 2 and "parse_error" not in out
    assert gen.calls[0]["temperature"] == 0 and gen.calls[0]["seed"] == 42 and gen.calls[0]["num_predict"] == 200
    gen = fake(["basura", "peor"])
    out = judge_local.judge_record(rec(), gold(), "j", "dig", generate=gen)
    assert out["parse_error"] is True and out["raw"] == "peor" and out["tipo"] == "complementario"


def test_prompt_juez_contiene_gold():
    p = judge_local.build_prompt(rec(), gold())
    assert "ref" in p and "crit" in p and "[1]" in p and "¿P?" in p


def sample_records():
    out = []
    for m in ("m1", "m2"):
        for b in ("milvus", "solr"):
            for q in range(1, 6):
                out.append(rec(f"Q{q}", model=m, backend=b))
    out.append(rec("Q9", model="m1", run_idx=2))
    out.append(dict(qid="Q1", model="m1", backend="milvus", run_idx=1, run_id="e", error="x"))
    return out


def test_sample_ciego(tmp_path):
    golds = {f"Q{q}": gold(f"Q{q}") for q in list(range(1, 6)) + [9]}
    human_eval.sample(sample_records(), golds, n=8, seed=42, out=tmp_path)

    def read(name):
        return list(csv.DictReader((tmp_path / name).open(encoding="utf-8-sig")))

    a, b, key = read("hoja_evaluador_A.csv"), read("hoja_evaluador_B.csv"), read("clave_items.csv")
    assert len(a) == len(b) == len(key) == 8
    assert sorted(r["item_id"] for r in a) == sorted(r["item_id"] for r in b)
    assert [r["item_id"] for r in a] != [r["item_id"] for r in b]
    assert "model" not in a[0] and "backend" not in a[0] and "modelo" not in a[0]
    assert {"correccion", "fidelidad", "relevancia", "calidad_citas", "abstencion"} <= set(a[0])
    assert "@usuario" in a[0]["evidencia"] and "ana.perez" not in a[0]["evidencia"]
    assert {r["model"] for r in key} == {"m1", "m2"} and all(r["run_id"] for r in key)


CRIT = ("correccion", "fidelidad", "relevancia", "calidad_citas", "abstencion")


def write_sheet(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, ["item_id", *CRIT])
        w.writeheader()
        w.writerows(rows)


def setup_agree(tmp_path, a_rows, b_rows):
    key = tmp_path / "clave.csv"
    with key.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, ["item_id", "run_id", "model", "backend", "qid"])
        w.writeheader()
        for i in range(1, 5):
            w.writerow(dict(item_id=i, run_id=f"r{i}", model="m1", backend="milvus", qid=f"Q{i}"))
    write_sheet(tmp_path / "a.csv", a_rows)
    write_sheet(tmp_path / "b.csv", b_rows)
    return tmp_path / "a.csv", tmp_path / "b.csv", key


def rows(vals):
    return [dict(item_id=i + 1, **{c: v for c in CRIT}) for i, v in enumerate(vals)]


def test_agree(tmp_path):
    a, b, key = setup_agree(tmp_path, rows([0, 1, 2, 2]), rows([0, 1, 2, 0]))
    out = tmp_path / "out"
    res = human_eval.agree(a, b, key, out)
    assert res["kappa"]["correccion"] == pytest.approx(5 / 13)  # calculado a mano
    assert res["exact"]["correccion"] == 0.75

    def read(name):
        return list(csv.DictReader((out / name).open(encoding="utf-8-sig")))

    assert {d["item_id"] for d in read("human_disagreements.csv")} == {"4"}
    items = read("human_eval_items.csv")
    assert items[3]["run_id"] == "r4" and float(items[3]["correccion"]) == 1.0
    assert (out / "human_eval.csv").exists()


def test_agree_celda_vacia(tmp_path):
    bad = rows([0, 1, 2, 2])
    bad[2]["fidelidad"] = ""
    a, b, key = setup_agree(tmp_path, rows([0, 1, 2, 2]), bad)
    with pytest.raises(ValueError, match=r"3.*fidelidad"):
        human_eval.agree(a, b, key, tmp_path / "o")


def test_agree_omite_respuestas_sin_calificar(tmp_path):
    a_rows, b_rows = rows([0, 1, 2, 2]), rows([0, 1, 2, 2])
    for r in (a_rows[3], b_rows[3]):
        for c in human_eval.CRITERIA:
            r[c] = ""
    a, b, key = setup_agree(tmp_path, a_rows, b_rows)
    human_eval.agree(a, b, key, tmp_path / "o")
    items = (tmp_path / "o" / "human_eval_items.csv").read_text(encoding="utf-8").splitlines()
    assert len(items) == 4  # encabezado + 3 respuestas calificadas


def test_load_raw_keeps_retry_over_error_and_never_replaces_success(tmp_path):
    from scripts.score_generation import load_raw
    base = dict(tag="main", model="m", backend="solr", qid="Q01")
    first = [dict(base, run_idx=1, error="500"), dict(base, run_idx=2, answer="ok [1]")]
    second = [dict(base, run_idx=1, answer="reintento [1]"), dict(base, run_idx=2, answer="duplicado")]
    for name, rows in (("main_20261108_0100.jsonl", first), ("main_20261108_0200.jsonl", second)):
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    rows = {r["run_idx"]: r for r in load_raw(["main"], tmp_path)}
    assert rows[1]["answer"] == "reintento [1]" and rows[1]["retried"] is True
    assert rows[2]["answer"] == "ok [1]" and "retried" not in rows[2]
