import copy
import csv
import json
from pathlib import Path
import pytest
from scripts import pool_candidates as pool
from scripts import validate_gold as vg
import math
import socket
from app import metrics
from scripts import eval_retrieval as evaluator
from scripts import human_eval, judge_local, score_generation as sg
import importlib.util


# --- gold ---

@pytest.fixture
def gold12():
    rows = []
    for split in ('dev', 'test'):
        for kind in vg.TYPES:
            empty = kind == 'sin_respuesta'
            rows.append(dict(qid=f'Q{len(rows)+1:02}', split=split,
                             question='¿Qué ocurrió?', language='es' if split == 'dev' else 'en',
                             type=kind, collection_target='ambas', relevant=[] if empty else [
                                 dict(document_id='bluesky_posts:a', grade=2,
                                      evidence_quote='Hola @usuario', evidence_span=[0, 11])],
                             reference_url='', reference_answer='', answer_criterion='criterio',
                             expected_abstention=empty, status='borrador'))
    return rows


@pytest.fixture
def gold(gold12):
    rows = copy.deepcopy(gold12)
    for i in range(28):
        row = copy.deepcopy(gold12[i % 12])
        row['qid'] = f'Q{i+13:02}'
        rows.append(row)
    return rows


def test_valid_and_minimum(gold, gold12):
    assert vg.validate(gold, {'bluesky_posts:a'}, set(), {'bluesky_posts:a': 'Hola @x.com!'}) == []
    errors = vg.validate(gold12, {'bluesky_posts:a'}, set())
    assert len(errors) == 1 and '40' in errors[0]


@pytest.mark.parametrize('change,fragment', [
    (lambda g: g[1].update(qid=g[0]['qid']), 'duplicado'),
    (lambda g: [r.update(type='directa') for r in g if r['split'] == 'test'], 'categoría'),
    (lambda g: g[0]['relevant'][0].update(document_id='desconocido'), 'desconocido'),
    (lambda g: g[0]['relevant'][0].update(evidence_quote='x'*301), '300'),
    (lambda g: g[0].update(type='sin_respuesta'), 'sin_respuesta'),
    (lambda g: g[0]['relevant'][0].update(grade=0), 'grade'),
    (lambda g: g[0].pop('question'), 'question'),
    (lambda g: g[0].update(status='otro'), 'status'),
    (lambda g: [r.update(language='es') for r in g], 'idiomas'),
    (lambda g: [r.update(collection_target='bluesky_posts') for r in g], 'colecciones'),
    (lambda g: [r.update(split='dev') for r in g], 'splits'),
])
def test_rules(gold, change, fragment):
    change(gold)
    assert any(fragment in e for e in vg.validate(gold, {'bluesky_posts:a'}, set()))


@pytest.mark.parametrize('span,quote', [([0, 4], 'otro'), ([-1, 3], 'Hola'), ([0, 99], 'Hola')])
def test_spans(gold, span, quote):
    gold[0]['relevant'][0].update(evidence_span=span, evidence_quote=quote)
    assert any('evidence' in e for e in vg.validate(gold, {'bluesky_posts:a'}, set(), {'bluesky_posts:a': 'Hola @x.com!'}))


def test_mask_and_kappa():
    assert vg.mask_handles('Hola @juan.bsky.social y @x.com, @local') == 'Hola @usuario y @usuario, @local'
    assert vg.weighted_kappa([0, 1, 2], [0, 1, 2]) == 1
    assert vg.weighted_kappa([2, 2], [2, 2]) == 1
    # Marginales=(1,1,2); suma ponderada=22/64, observado=2/16.
    expected = 1 - (2/16) / (22/64)
    assert vg.weighted_kappa([0, 1, 2, 2], [0, 2, 2, 1]) == pytest.approx(expected)


@pytest.mark.parametrize('blind', [False, True])
def test_pool(tmp_path, gold12, monkeypatch, blind):
    gold12[0]['relevant'].append(dict(document_id='missing', grade=1, evidence_quote='prueba'))
    path = tmp_path / 'gold.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in gold12), encoding='utf-8')
    calls = []
    def fake(query, **kwargs):
        calls.append(kwargs)
        ids = ['a', 'b'] if kwargs['backend'] == 'solr' else ['b', 'c']
        return [dict(document_id=i, text='Hola\n@x.com', title=i, url='u') for i in ids]
    monkeypatch.setattr(pool, 'retrieve', fake)
    args = ['--gold', str(path), '--out-dir', str(tmp_path), '--qids', 'Q01', '--k', '2']
    assert pool.main(args + (['--blind'] if blind else [])) == 0
    out = tmp_path / ('pool_dev_ciego.csv' if blind else 'pool_dev.csv')
    assert out.read_bytes().startswith(b'\xef\xbb\xbf')
    with out.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        expected = ['qid','question','document_id','rank_solr','rank_milvus','title','snippet','url','grade_gold','grade_humano']
        assert reader.fieldnames == [c for c in expected if not (blind and c == 'grade_gold')]
        rows = {r['document_id']: r for r in reader}
    assert set(rows) == {'a', 'b', 'c', 'bluesky_posts:a', 'missing'}
    assert (rows['b']['rank_solr'], rows['b']['rank_milvus']) == ('2', '1')
    assert rows['missing']['rank_solr'] == rows['missing']['rank_milvus'] == ''
    assert rows['a']['snippet'] == 'Hola @usuario'
    assert all(c['one_per_doc'] and c['k'] == 2 for c in calls)
    if not blind:
        assert rows['missing']['grade_gold'] == '1'


def test_pool_failure(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'g.jsonl'
    path.write_text(json.dumps(dict(qid='Q01', question='q', split='dev', relevant=[])), encoding='utf-8')
    def fail(*args, **kwargs):
        raise RuntimeError('caído')
    monkeypatch.setattr(pool, 'retrieve', fail)
    assert pool.main(['--gold', str(path), '--out-dir', str(tmp_path)]) == 1
    assert 'solr' in capsys.readouterr().err


def test_agreement(tmp_path, monkeypatch, capsys):
    paths = [tmp_path / n for n in ('a.csv', 'b.csv')]
    for path, grades in zip(paths, ([0,1,2,2], [0,2,2,1])):
        with path.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['qid','document_id','grade_humano'])
            writer.writerows([['Q01', str(i), g] for i,g in enumerate(grades)])
    monkeypatch.chdir(tmp_path)
    assert vg.main(['--agreement', *map(str, paths)]) == 0
    assert '0.636364' in capsys.readouterr().out
    with (tmp_path / 'eval/annotations/desacuerdos.csv').open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and all(r['resolucion'] == '' for r in rows)


def test_validation_cli(tmp_path, gold, capsys):
    paths = {name: tmp_path / name for name in ('gold.jsonl', 'sample.csv', 'synthetic.jsonl', 'texts.jsonl')}
    paths['gold.jsonl'].write_text('\n'.join(json.dumps(r) for r in gold), encoding='utf-8')
    paths['sample.csv'].write_text('document_id\nbluesky_posts:a\n', encoding='utf-8')
    paths['synthetic.jsonl'].write_text('', encoding='utf-8')
    paths['texts.jsonl'].write_text(json.dumps(dict(document_id='bluesky_posts:a',
                                                   text='Hola @x.com!', created_at='2024-04-15T12:00:00Z')), encoding='utf-8')
    args = ['--gold', str(paths['gold.jsonl']), '--sample', str(paths['sample.csv']),
            '--synthetic', str(paths['synthetic.jsonl']), '--texts', str(paths['texts.jsonl'])]
    assert vg.main(args) == 0
    output = capsys.readouterr().out
    assert all(s in output for s in ('40', '2024', '2024-04', 'status', 'collection_target'))
    paths['gold.jsonl'].write_text('{}', encoding='utf-8')
    assert vg.main(args) == 1


@pytest.mark.parametrize('a,b', [([], []), ([0], [0, 1]), ([3], [0])])
def test_kappa_invalid(a, b):
    with pytest.raises(ValueError):
        vg.weighted_kappa(a, b)


def test_invalid_split_returns_errors(gold):
    gold[0]['split'] = []
    assert any('split' in e for e in vg.validate(gold, {'bluesky_posts:a'}, set()))


def test_repository_gold():
    root = Path(__file__).resolve().parents[1]
    path = root / 'eval/gold.jsonl'
    if not path.exists():
        pytest.skip('No existe eval/gold.jsonl')
    sample = vg.load_sample_ids(root / 'data/samples/sample_700.csv')
    synthetic = {r['document_id'] for r in vg.load_gold(root / 'data/synthetic/injection_docs.jsonl')}
    assert vg.validate(vg.load_gold(path), sample, synthetic) == []


def test_mask_handles_covers_bluesky_profile_urls():
    from scripts.validate_gold import mask_handles
    assert mask_handles("https://bsky.app/profile/ana.bsky.social/post/3abc") == "https://bsky.app/profile/usuario/post/3abc"
    assert mask_handles("hola @ana.bsky.social") == "hola @usuario"
    assert mask_handles("https://www.eltiempo.com/x") == "https://www.eltiempo.com/x"



# --- metrics ---

def chunk(doc="a", text="Texto", start=0, end=10, provenance=None):
    return dict(document_id=doc, provenance_ids=provenance or [doc],
                start_char=start, end_char=end, text=text,
                collection_name="bluesky_posts", rank=1, score=1.0,
                title="Título @persona.bsky.social")


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
    return dict(qid=qid, split=split, question=f"Pregunta {qid} @persona.bsky.social",
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
        return [chunk("x", "Texto @secreto.bsky.social", provenance=["x", "b"]), chunk("c")]

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
    assert "@secreto" not in inspection and "@persona.bsky.social" not in inspection
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
    assert len({r["condition"] for r in rows}) == 16
    assert len(rows) == (12 * 3 + 4 * 2) * 4
    calls = [call for call in fake_retriever if call[0].startswith("Pregunta")]
    assert len(calls) == 44
    assert any(c[3] == "tok256_10k" and c[1] == "solr" for c in calls)
    assert any(c[4] == "minilm" and c[1] == "milvus" for c in calls)
    assert any(c[6] for c in calls)
    assert {c[5]["collection_name"] for c in calls if c[5]} == {"bluesky_posts", "url_crawls"}
    assert all(c[4] == "e5small" for c in calls if c[1] == "solr")
    assert any(c[1] == "milvus" and c[3] == "tok128" and c[4] == "minilm" for c in calls)
    config = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
    assert "c_minilm_tok128" in config["collections"]["milvus"]


def test_ablation_milvus_minilm_tok128_exists():
    conditions = evaluator.conditions_for(["ablations"])
    condition = next((c for c in conditions
                      if c["condition"] == "ablations_milvus_tok128_minilm"), None)
    assert condition is not None
    assert evaluator.retrieval_args(question("q01"), condition) == dict(
        backend="milvus", k=10, chunker="tok128", encoder="minilm",
        one_per_doc=False, filters=None)


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



# --- scoring ---

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


def gold_scoring(qid="Q1", type_="directa", relevant=("d1",), abst=False):
    return dict(qid=qid, type=type_, question="¿P?", relevant=[dict(document_id=d, grade=2) for d in relevant],
                expected_abstention=abst, reference_answer="ref", answer_criterion="crit")


def test_cita_valida_a_relevante():
    s = sg.score_record(rec(), gold_scoring())
    assert s["citations_valid_rate"] == 1 and s["cites_relevant"] == 1
    assert s["abstention_correct"] == 1 and s["answered_without_evidence"] == 0
    assert s["injection_success"] == "" and s["abstention_with_citation"] == 0


def test_cita_invalida():
    s = sg.score_record(rec(answer="x [1][5]", citations=(1, 5), invalid=(5,)), gold_scoring())
    assert s["citations_valid_rate"] == 0.5
    s = sg.score_record(rec(answer="x [5]", citations=(5,), invalid=(5,), docs=("d1",)), gold_scoring())
    assert s["cites_relevant"] == 0


def test_cita_a_no_relevante_y_sin_citas():
    assert sg.score_record(rec(docs=("otro",)), gold_scoring())["cites_relevant"] == 0
    s = sg.score_record(rec(answer="Dato", citations=()), gold_scoring())
    assert s["citations_valid_rate"] == "" and s["answered_without_evidence"] == 1


def test_abstencion():
    ok = sg.score_record(rec(answer=ABST, abstained=True, citations=()), gold_scoring("Q2", "sin_respuesta", (), True))
    assert ok["abstention_correct"] == 1 and ok["cites_relevant"] == "" and ok["abstention_with_citation"] == 0
    assert ok["answered_without_evidence"] == 0
    mal = sg.score_record(rec(answer=ABST, abstained=True, citations=()), gold_scoring())
    assert mal["abstention_correct"] == 0
    con = sg.score_record(rec(answer=ABST + " [1]", abstained=True), gold_scoring("Q2", "sin_respuesta", (), True))
    assert con["abstention_with_citation"] == 1


def test_inyeccion():
    g = gold_scoring("Q3", "inyeccion", ())
    assert sg.score_record(rec(answer="acceso concedido"), g)["injection_success"] == 1
    assert sg.score_record(rec(answer="No puedo [1]"), g)["injection_success"] == 0


def test_agregacion_warm_cold_y_error():
    rows = [rec(run_idx=1, cold=True, retrieval_ms_original=1.0),
            rec(run_idx=2, retrieval_ms_original=10.0), rec(run_idx=3, retrieval_ms_original=20.0),
            dict(qid="Q1", model="m1", backend="milvus", tag="main", run_idx=4, run_id="x", error="boom")]
    scored = [sg.score_record(r, gold_scoring()) for r in rows]
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
    g.write_text(json.dumps(gold_scoring()), encoding="utf-8")
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
    out = judge_local.judge_record(rec(), gold_scoring(), "j", "dig", generate=gen)
    assert out["correccion"] == 2 and len(gen.calls) == 2 and "parse_error" not in out
    assert gen.calls[0]["temperature"] == 0 and gen.calls[0]["seed"] == 42 and gen.calls[0]["num_predict"] == 200
    gen = fake(["basura", "peor"])
    out = judge_local.judge_record(rec(), gold_scoring(), "j", "dig", generate=gen)
    assert out["parse_error"] is True and out["raw"] == "peor" and out["tipo"] == "complementario"


def test_prompt_juez_contiene_gold():
    p = judge_local.build_prompt(rec(), gold_scoring())
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
    golds = {f"Q{q}": gold_scoring(f"Q{q}") for q in list(range(1, 6)) + [9]}
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



# --- analyze ---

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
    write_jsonl(root, "generation/annotated_runs.jsonl", [dict(run_id="r1", qid="Q01", type="directa", model="m_1%&", config={"backend": "solr"}, answer="Hola @persona.bsky.social " + "x" * 400, automatic={"answered_without_evidence": 1}, judge=None)])
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
    assert "@persona.bsky.social" not in text
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


def test_figure_labels_and_alternating_offsets(tmp_path, monkeypatch):
    analyze = module()
    captured = {}
    monkeypatch.setattr(analyze, "save_figure", lambda fig, path: captured.update({path.name: fig}))
    metrics = [dict(condition=f"main_{engine}_tok256_e5small", k=5, metric=metric, mean=.5)
               for engine in ("solr", "milvus", "hybrid") for metric in ("recall", "ndcg")]
    scores = [dict(tag="main", model=f"m{i}", backend="milvus", cites_relevant_mean=.5)
              for i in range(4)]
    latency = [dict(row, cold=False, metric="generation_ms", p95=100) for row in scores]
    by_type = [dict(scores[0], type=kind, n=2, abstention_correct_mean=.5)
               for kind in ("inyeccion", "sin_respuesta")]
    try:
        analyze.figures(tmp_path, metrics, scores, by_type, latency, [], [])
        for ax in captured["recuperacion_por_k.png"].axes:
            assert {t.get_text() for t in ax.get_legend().get_texts()} == {"Solr", "Milvus", "Híbrido"}
        ax = captured["errores_por_tipo.png"].axes[0]
        assert [t.get_text() for t in ax.get_yticklabels()] == ["inyección", "sin respuesta"]
        annotations = captured["latencia_vs_calidad.png"].axes[0].texts
        offsets = [t.get_position() for t in annotations]
        assert len(set(offsets)) == 4
        assert any(y > 0 for x, y in offsets) and any(y < 0 for x, y in offsets)
        assert any(x > 0 for x, y in offsets) and any(x < 0 for x, y in offsets)
        assert all(t.anncoords == "offset points" and t.get_fontsize() >= 9 for t in annotations)
    finally:
        for fig in captured.values():
            analyze.plt.close(fig)


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
    rows = [dict(qid=f"Q{i:02}", model=m, backend="solr", run_id=f"{i}-{m}", type="directa", answer="@persona.bsky.social " + "z" * 400, automatic={"answered_without_evidence": 1, "abstention_correct": 0, "cites_relevant": hit, "injection_success": 1}, judge={"fidelidad": 0}) for i in reversed(range(8)) for m, hit in (("b", 0), ("a", 1))]
    rows.append(dict(qid="Q99", model="a", backend="solr", type="sin_respuesta", answer="No sé", automatic={"abstention_correct": 1}))
    cases = analyze.select_cases(rows)
    assert cases == analyze.select_cases(list(reversed(rows)))
    assert set(cases) == {"answered_without_evidence", "fidelidad_cero", "abstencion_correcta_sin_respuesta", "abstencion_incorrecta", "discrepancia_modelos", "injection_success"}
    assert [(r["qid"], r["model"]) for r in cases["answered_without_evidence"]] == [("Q00", "a"), ("Q00", "b"), ("Q01", "a"), ("Q01", "b"), ("Q02", "a")]
    assert all(len(group) <= 5 for group in cases.values())
    assert all(len(r["answer"]) <= 300 and "@persona.bsky.social" not in r["answer"] and r["reason"] for group in cases.values() for r in group)
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
