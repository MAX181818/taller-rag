"""Verifica el análisis con entradas pequeñas y sin servicios."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze.py"
    assert path.exists(), "Falta el analizador"
    spec = importlib.util.spec_from_file_location("analyze", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def write_csv(root, name, rows):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def write_json(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def write_jsonl(root, name, rows):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def sample(tmp_path):
    root = tmp_path / "reports"
    metrics = [dict(condition=c, k=k, metric=m, n=4, mean=.75, ci_low=.5, ci_high=1)
               for c in ("main_solr_tok256_e5small", "ablations_solr_sent256_e5small",
                         "filters_solr_tok256_e5small_filtered", "10k_solr_tok256_10k_e5small")
               for k in (1, 3, 5, 10) for m in ("hit", "recall", "mrr", "ndcg", "fragment_hit", "excluded")]
    write_csv(root, "retrieval/metrics.csv", metrics)
    write_csv(root, "retrieval/metrics_by_group.csv", [dict(metrics[0], group_by="type", group="directa")])
    write_csv(root, "retrieval/latency.csv", [dict(condition=metrics[0]["condition"], phase="warm", n=4, p50_ms=2, p95_ms=4)])
    write_csv(root, "retrieval/per_question.csv", [dict(condition=metrics[0]["condition"], qid="Q01", k=5, recall=1)])
    scores = []
    for tag in ("main", "abl_prompt"):
        row = dict(tag=tag, model="m_1%&", backend="solr", n=4, n_total=4, error_rate=0)
        for metric in ("abstention_correct", "cites_relevant", "citations_valid_rate", "injection_success"):
            row.update({metric + "_mean": .75, metric + "_ci_low": .5, metric + "_ci_high": 1})
        scores.append(row)
    write_csv(root, "generation/scores_summary.csv", scores)
    write_csv(root, "generation/scores_by_type.csv", [dict(scores[0], type="directa")])
    write_csv(root, "generation/latency_summary.csv", [dict(tag="main", model="m_1%&", backend="solr", cold=cold, metric="generation_ms", n=n, p50=10, p95=20, tokens_per_s_mean=speed) for cold, n, speed in ((False, 3, 40), (True, 1, 20))])
    call = dict(run_id="r1", qid="Q01", type="directa", tag="main", model="m_1%&", backend="solr", run_idx=1, cold=False, error=0, citations_valid_rate="", cites_relevant=0, abstention_correct=1, abstention_with_citation=0, injection_success="", answered_without_evidence=1)
    write_csv(root, "generation/scores_per_call.csv", [call])
    write_jsonl(root, "generation/annotated_runs.jsonl", [dict(run_id="r1", qid="Q01", type="directa", model="m_1%&", config={"backend": "solr"}, answer="Hola @persona " + "x" * 400, automatic={"answered_without_evidence": 1}, judge=None)])
    write_jsonl(root, "generation/judge.jsonl", [dict(run_id="r1", qid="Q01", model="m_1%&", backend="solr", correccion=1, fidelidad=0, relevancia=2)])
    write_csv(root, "generation/human_eval.csv", [dict(model="m_1%&", backend="solr", criterio="fidelidad", n=2, promedio=1.5, kappa_global=.5, acuerdo_exacto_global=.5)])
    write_json(root, "runs/raw/main_config.json", {"model_info": {"m_1%&": {"parameter_size": "1B", "quantization_level": "Q4_0", "family": "f", "digest": "123456789012abcd"}}})
    write_jsonl(root, "runs/raw/main_20261107_1200.jsonl", [dict(run_id="r1", model="m_1%&", timestamp="2026-11-07T12:00:10+00:00", timings={"generation_ms": 10000}, tokens={"tokens_per_s": 40})])
    write_csv(root, "runs/raw/sysmon_main.csv", [dict(timestamp=t, container="api", cpu_pct=1, mem_used_mb=2, mem_limit_mb=3, gpu_mem_used_mb=v, gpu_mem_total_mb=100, gpu_util_pct=1) for t, v in (("2026-11-07T12:00:05+00:00", 50), ("2026-11-07T13:00:00+00:00", 99))])
    write_json(root, "embeddings/sensibilidad.json", {"e5small": {"norm_mean": .999}})
    write_csv(root, "embeddings/solapamiento.csv", [dict(k=5, overlap=.6)])
    write_json(root, "perfilado/perfil.json", {"rows": 100})
    write_json(root, "chunking/stats.json", {"tok256": {"n_chunks": 30}})
    return root, tmp_path / "tex"


def test_outputs_and_provenance(sample):
    root, tex = sample
    assert module().main(["--reports", str(root), "--tex", str(tex)]) == 0
    names = ("recuperacion_principal", "recuperacion_ablaciones", "recuperacion_filtros", "recuperacion_10k", "latencia_recuperacion", "generacion_principal", "generacion_ablaciones", "latencia_generacion", "modelos", "juez", "humana", "embeddings_sensibilidad")
    for name in names:
        assert (root / "tables" / (name + ".csv")).is_file()
        assert (tex / (name + ".tex")).is_file()
    latex = (tex / "generacion_principal.tex").read_text(encoding="utf-8")
    assert r"m\_1\%\&" in latex
    assert "[0.500, 1.000]" in latex
    assert r"\toprule" in latex
    text = (root / "RESULTADOS.md").read_text(encoding="utf-8")
    for title in ("Recuperación", "Generación", "Rendimiento", "Embeddings", "Robustez (inyección)", "Casos para análisis de errores"):
        assert "## " + title in text
    assert "| condición | k | métrica | n | media | IC 95 % |" in text
    assert "| main_solr_tok256_e5small | 1 | recall | 4 | 0.750 | [0.500, 1.000] |" in text
    assert "k: 1.000" not in text
    retrieval = text.split("## Recuperación", 1)[1].split("## Generación", 1)[0]
    assert " | 3 | " not in retrieval
    assert "excluded" not in retrieval
    for metric in ("hit", "recall", "mrr", "ndcg", "fragment_hit"):
        assert f"| {metric} |" in retrieval
    assert retrieval.count("Fuente:") == 4
    assert "| 2.0 | 4.0 |" in text
    assert "| tag | model | backend |" in text
    for section in text.split("\n## ")[1:]:
        assert "| " in section and "Fuente: `" in section
    assert text.count("Fuente:") == sum(line.startswith("| ---") for line in text.splitlines())
    assert "@persona" not in text
    assert "123456789012abcd" not in (tex / "modelos.tex").read_text(encoding="utf-8")
    with (root / "tables/modelos.csv").open(encoding="utf-8-sig", newline="") as source:
        assert next(csv.DictReader(source))["digest"] == "123456789012"
    from PIL import Image
    for name in ("recuperacion_por_k", "ablaciones_recuperacion", "generacion_por_modelo", "juez_por_modelo", "latencia_vs_calidad", "tokens_vram", "errores_por_tipo"):
        with Image.open(root / "figures" / (name + ".png")) as img:
            assert img.info["dpi"][0] == pytest.approx(150, abs=.1)
            assert img.width > 200


def test_missing_inputs_omit_sections(tmp_path, capsys):
    root = tmp_path / "reports"
    module().main(["--reports", str(root), "--tex", str(tmp_path / "tex")])
    text = (root / "RESULTADOS.md").read_text(encoding="utf-8")
    assert "## Generación" not in text
    assert "## Recuperación" not in text
    assert not list((root / "tables").glob("*.csv"))
    assert "Aviso" in capsys.readouterr().err


def test_markdown_escaping_and_formats():
    analyze = module()
    lines = []
    analyze.report_section(lines, "Ejemplo", [dict(
        model="12", n=40, k=5, p95_ms=12.34, rate_mean=.5,
        rate_ci_low=.25, rate_ci_high=.75, answer="a|b\nc", correccion_n=4, source="datos.csv")])
    text = "\n".join(lines)
    assert "| 12 | 40 | 5 | 12.3 | 0.500 [0.250, 0.750] | a&#124;b c | 4 |" in text
    assert text.count("Fuente: `datos.csv`") == 1


def test_cases_deterministic_and_categories():
    analyze = module()
    rows = [dict(qid=f"Q{i:02}", model=m, backend="solr", run_id=f"{i}-{m}", type="directa", answer="@persona " + "z" * 400, automatic={"answered_without_evidence": 1, "abstention_correct": 0, "cites_relevant": hit, "injection_success": 1}, judge={"fidelidad": 0}) for i in reversed(range(8)) for m, hit in (("b", 0), ("a", 1))]
    rows.append(dict(qid="Q99", model="a", backend="solr", type="sin_respuesta", answer="No sé", automatic={"abstention_correct": 1}))
    cases = analyze.select_cases(rows)
    assert cases == analyze.select_cases(list(reversed(rows)))
    assert set(cases) == {"answered_without_evidence", "fidelidad_cero", "abstencion_correcta_sin_respuesta", "abstencion_incorrecta", "discrepancia_modelos", "injection_success"}
    assert [(r["qid"], r["model"]) for r in cases["answered_without_evidence"]] == [("Q00", "a"), ("Q00", "b"), ("Q01", "a"), ("Q01", "b"), ("Q02", "a")]
    assert all(len(group) <= 5 for group in cases.values())
    assert all(len(r["answer"]) <= 300 and "@persona" not in r["answer"] and r["reason"] for group in cases.values() for r in group)
    assert len(cases["abstencion_correcta_sin_respuesta"]) == 1


def test_incorrect_abstention_requires_actual_abstention():
    rows = [dict(qid="Q1", model="m", backend="solr", type="directa", answer="Una respuesta", automatic={"abstention_correct": 0}, abstained=False)]
    assert module().select_cases(rows)["abstencion_incorrecta"] == []


def test_vram_time_assignment_and_weighted_speed(sample):
    root, _ = sample
    rows = module().performance_rows(root)
    assert rows[0]["tokens_per_s"] == 35
    assert rows[0]["vram_max_mb"] == 50
    (root / "runs/raw/main_20261107_1200.jsonl").unlink()
    assert module().performance_rows(root)[0].get("vram_max_mb") is None


def test_vram_provenance_in_report(sample):
    root, tex = sample
    module().analyze(root, tex)
    text = (root / "RESULTADOS.md").read_text(encoding="utf-8")
    assert "vram_max_mb" in text and "50.000" in text
    performance = text.split("## Rendimiento", 1)[1].split("## Embeddings", 1)[0]
    assert (root / "runs/raw/sysmon_main.csv").as_posix() in performance
    fidelity_case = text.split("### fidelidad_cero\n", 1)[1].split("###", 1)[0]
    assert (root / "generation/annotated_runs.jsonl").as_posix() in fidelity_case
    assert (root / "generation/judge.jsonl").as_posix() in fidelity_case


def test_partial_inputs_do_not_invent_quality(tmp_path):
    root = tmp_path / "reports"
    write_csv(root, "generation/latency_summary.csv", [dict(tag="main", model="m", backend="solr", cold=False, metric="generation_ms", n=1, p50=5, p95=7, tokens_per_s_mean=10)])
    module().analyze(root, tmp_path / "tex")
    assert (root / "tables/latencia_generacion.csv").exists()
    assert not (root / "figures/latencia_vs_calidad.png").exists()
    assert not (root / "tables/generacion_principal.csv").exists()
    assert "## Robustez" not in (root / "RESULTADOS.md").read_text(encoding="utf-8")


def test_case_join_uses_run_id_not_only_question(tmp_path):
    root = tmp_path / "reports"
    annotated = [dict(run_id=run, qid="Q1", model="m", type="directa", config={"backend": "solr"}, answer="Texto", automatic={}) for run in ("a", "b")]
    calls = [dict(run_id="a", abstention_correct=1), dict(run_id="b", abstention_correct=0)]
    judges = [dict(run_id="b", fidelidad=0)]
    records = module().case_records(root, annotated, calls, judges)
    assert records[0]["automatic"]["abstention_correct"] == 1
    assert records[0]["judge"] is None
    assert records[1]["judge"]["fidelidad"] == 0


def test_overlapping_models_do_not_receive_ambiguous_vram(sample):
    root, _ = sample
    write_jsonl(root, "runs/raw/extra_20261107_1200.jsonl", [dict(model="otro", timestamp="2026-11-07T12:00:10+00:00", timings={"generation_ms": 10000})])
    assert "vram_max_mb" not in module().performance_rows(root)[0]
