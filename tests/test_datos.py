import pandas as pd
import pytest
from scripts.make_sample import make_sample, normalize_created_at
import hashlib
import json
from time import perf_counter
from transformers import AutoTokenizer
from app.chunking import CHUNKERS, Span, chunk_fixed_tokens, chunk_id, chunk_sentences
from scripts.build_chunks import build_chunks, load_documents, main
import importlib
import math
import sys
from types import SimpleNamespace
import csv


# --- sample ---

def corpus():
    return pd.DataFrame([{"document_id": f"{i:05d}", "text_sha256": f"h{i}",
                          "text": f"Texto {i}", "collection_name": "A" if i < 9000 else "B"}
                         for i in range(12000)])


def test_dedup_keeps_min_id_and_all_provenance():
    df = pd.DataFrame({"document_id": ["b", "a", "c"], "text_sha256": ["h"] * 3,
                       "text": ["Texto"] * 3, "collection_name": ["A"] * 3})
    assert make_sample(df, 1).to_dict("records") == [
        {"document_id": "a", "text_sha256": "h", "collection_name": "A", "provenance_ids": "a|b|c"}]


def test_sample_is_deterministic_and_stratified():
    df = corpus()
    a = make_sample(df, 700)
    pd.testing.assert_frame_equal(a, make_sample(df.sample(frac=1, random_state=8), 700))
    assert len(a) == 700 and a.document_id.is_unique
    assert abs((a.collection_name == "A").mean() - 0.75) <= 0.01
    assert list(a.columns) == ["document_id", "text_sha256", "collection_name", "provenance_ids"]


def test_samples_are_nested():
    df = corpus()
    small, large = make_sample(df, 700), make_sample(df, 10000)
    assert len(large) == 10000
    assert set(small.document_id) <= set(large.document_id)
    for sample in (small, large):
        assert abs((sample.collection_name == "A").mean() - 0.75) <= 0.01


def test_excludes_empty_text():
    df = pd.DataFrame({"document_id": list("abcde"), "text_sha256": list("abcde"),
                       "text": ["", "  \t", None, "Texto", "\n"], "collection_name": ["A"] * 5})
    assert make_sample(df, 1).document_id.tolist() == ["d"]


@pytest.mark.parametrize("value,expected", [
    ("2026-08-16T21:20:04-03:00", 1786926004), ("2024-01-02", 1704153600),
    (None, -1), ("basura", -1), ("", -1)])
def test_normalize_created_at(value, expected):
    assert normalize_created_at(value) == expected


def test_invalid_sample_size():
    with pytest.raises(ValueError):
        make_sample(corpus(), 12001)
    with pytest.raises(ValueError):
        make_sample(corpus(), -1)
    assert make_sample(corpus(), 0).empty


def test_manifest_comparison(tmp_path):
    """Contrasta cifras y excluye conteos personales del documento publicado."""
    import json
    from scripts.profile_corpus import manifest_comparison

    archivo = tmp_path / "corpus.parquet"
    archivo.write_bytes(b"datos")
    manifiesto = tmp_path / "manifiesto.md"
    manifiesto.write_text(json.dumps({
        "rows": 3, "columns": 4, "output_bytes": 5, "output_sha256": "abc",
        "nonempty_counts": {"text": 1, "author_id": 2},
        "distributions": {"collection_name": {"A": 3}},
        "unique_texts": 2, "duplicate_text_rows": 1,
        "validation": {"ids_unique": True, "all_columns_round_trip": True},
    }), encoding="utf-8")
    df = pd.DataFrame({"document_id": ["a", "b", "c"], "text": ["Texto", "  ", None],
                       "collection_name": ["A"] * 3, "author_id": ["x", "y", None]})
    perfil = {"rows": 3, "fields": 4, "sha256_before": "abc", "unique_text_hashes": 2,
              "dup_text_rows": 1, "document_id_unique": True}
    resultado = manifest_comparison(manifiesto, df, perfil, archivo)
    cifras = {fila["figure"]: fila for fila in resultado["comparison"]}
    assert cifras["nonempty_counts.text"]["measured"] == 1
    assert cifras["output_bytes"]["matches"] is True
    assert cifras["distributions.collection_name.A"]["matches"] is True
    assert cifras["validation.all_columns_round_trip"]["matches"] is None
    assert "author_id" not in json.dumps(resultado)
    for path in (None, tmp_path / "ausente"):
        missing = manifest_comparison(path, df, perfil, archivo)
        assert missing["status"] == "Manifiesto no proporcionado: se omite la comparaci\u00f3n."
        assert missing["comparison"] == []



# --- chunking ---

@pytest.fixture(scope="session")
def tok():
    return AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small", local_files_only=True)


@pytest.fixture(params=[
    "  Bogotá amplía las rutas escolares.\n\nLa Secretaría presentó el informe de educación. " * 80,
    "Hoy llueve en Medellín 🌧️. Consulta https://x.co/a.b/c?d=1… para salir.\n¡Cuida el agua! " * 70,
])
def original(request):
    return request.param


def document(document_id, text):
    return dict(document_id=document_id, text=text, title="Noticia local", url="https://example.org/nota",
                collection_name="url_crawls", source="fixture", language_base="es",
                created_at="2024-01-02", provenance_ids=[document_id])


@pytest.mark.parametrize("chunker", CHUNKERS)
def test_offsets_reproduce_original_text(tok, original, chunker):
    records, stats = build_chunks([document("a", original)], tok, chunker)
    spans = CHUNKERS[chunker](original, tok)
    assert spans and len(spans) == len(records)
    for span, record in zip(spans, records):
        assert isinstance(span, Span)
        assert 0 <= span.start < span.end <= len(original)
        assert record["text"] == original[span.start:span.end]
        assert (record["start_char"], record["end_char"]) == (span.start, span.end)
        assert record["n_tokens"] == len(tok(record["text"], add_special_tokens=False)["input_ids"])
    assert stats["offset_mismatches"] == 0


@pytest.mark.parametrize("name,size,overlap", [("tok256", 256, 32), ("tok128", 128, 16)])
def test_overlap_is_32_tokens(tok, name, size, overlap):
    text = "el " * 800
    spans = CHUNKERS[name](text, tok)
    for left, right in zip(spans, spans[1:]):
        assert len(tok(text[left.start:left.end], add_special_tokens=False)["input_ids"]) == size
        assert len(tok(text[right.start:left.end], add_special_tokens=False)["input_ids"]) == overlap
    assert spans[-1].end == len(text.rstrip())


@pytest.mark.parametrize("name", CHUNKERS)
@pytest.mark.parametrize("text", ["", "  \n\t  "])
def test_empty_and_whitespace_give_zero_chunks(tok, name, text):
    assert CHUNKERS[name](text, tok) == []


@pytest.mark.parametrize("name", CHUNKERS)
def test_huge_text_terminates_and_covers_all(tok, name, record_property):
    text = "palabra " * 60000
    started = perf_counter()
    spans = CHUNKERS[name](text, tok)
    elapsed = perf_counter() - started
    record_property("seconds", elapsed)
    assert elapsed <= 5, elapsed
    covered = bytearray(len(text))
    for span in spans:
        covered[span.start:span.end] = b"\x01" * (span.end - span.start)
    assert all(covered[i] for i, char in enumerate(text) if not char.isspace())


def test_sentence_chunker_never_splits_sentence_under_limit(tok):
    sentences = [f"El municipio {i} presenta un informe sobre transporte y educación." for i in range(100)]
    text = "\n\n".join(sentences)
    spans = chunk_sentences(text, tok)
    assert len(spans) > 1
    cursor = 0
    for sentence in sentences:
        start = text.index(sentence, cursor)
        end = start + len(sentence)
        assert any(span.start <= start and end <= span.end for span in spans)
        cursor = end
    assert all(len(tok(text[s.start:s.end], add_special_tokens=False)["input_ids"]) <= 256 for s in spans)


def test_sentence_chunker_does_not_split_inside_url(tok):
    url = "https://x.co/a.b/c?d=1…"
    text = "El reporte está en " + url + " Consulta los horarios."
    spans = chunk_sentences(text, tok, max_tokens=24)
    start, end = text.index(url), text.index(url) + len(url)
    assert any(s.start <= start and end <= s.end for s in spans)
    assert not any(start < boundary < end for s in spans for boundary in (s.start, s.end))


def test_sentence_chunker_falls_back_for_long_sentence(tok):
    text = "el " * 700 + "."
    spans = chunk_sentences(text, tok)
    assert len(spans) > 1
    assert spans[0].start == 0 and spans[-1].end == len(text)
    assert all(len(tok(text[s.start:s.end], add_special_tokens=False)["input_ids"]) <= 256 for s in spans)


@pytest.mark.parametrize("ending", [".", "…"])
def test_sentence_chunker_preserves_sentences_ending_in_url(tok, ending):
    first = "el " * 150 + "https://example.org/nota" + ending
    second = "el " * 150 + "."
    text = first + " " + second
    spans = chunk_sentences(text, tok)
    assert spans == [Span(0, len(first)), Span(len(first) + 1, len(text))]


@pytest.mark.parametrize("name,size", [("tok256", 256), ("tok128", 128)])
def test_fixed_token_windows_respect_size_after_retokenization(tok, name, size):
    text = "anticonstitucionalmente " * 300
    spans = CHUNKERS[name](text, tok)
    assert all(len(tok(text[s.start:s.end], add_special_tokens=False)["input_ids"]) <= size for s in spans)


def test_chunk_id_stable():
    expected = hashlib.sha256(b"doc|tok256|0|10").hexdigest()[:32]
    assert chunk_id("doc", "tok256", 0, 10) == expected
    assert chunk_id("doc", "tok256", 0, 10) == chunk_id("doc", "tok256", 0, 10)
    assert chunk_id("doc", "tok128", 0, 10) != expected


def test_duplicate_chunks_merge_all_occurrences(tok):
    a = document("a", "La energía llega al barrio.")
    a["provenance_ids"] = ["a", "a_original"]
    b = document("b", "LA ENERGÍA   llega al barrio.")
    records, stats = build_chunks([b, a], tok, "sent256")
    assert len(records) == 1
    record = records[0]
    assert record["document_id"] == "a" and record["text"] == a["text"]
    assert record["provenance_ids"] == ["a", "a_original", "b"]
    assert record["occurrences"] == [
        {"document_id": "a", "start_char": 0, "end_char": len(a["text"])},
        {"document_id": "b", "start_char": 0, "end_char": len(b["text"])},
    ]
    assert stats["chunks_merged"] == 1
    assert stats["n_documents"] == 2 and stats["n_chunks"] == 1
    assert stats["single_chunk_documents_pct"] == 100
    assert stats["chunks_per_document_by_collection"]["url_crawls"] == {"mean": 1.0, "p50": 1.0, "max": 1}
    assert build_chunks([a, b], tok, "sent256") == (records, stats)


def test_repetitions_within_one_document_are_preserved(tok):
    records, stats = build_chunks([document("a", "el " * 600)], tok, "tok256")
    assert len(records) == 3
    assert stats["chunks_merged"] == 0
    assert len({r["chunk_id"] for r in records}) == 3


def test_metadata_defaults_and_stats(tok):
    doc = document("a", "El clima es estable.")
    doc["language_base"] = None
    doc["created_at"] = "fecha inválida"
    records, stats = build_chunks([doc, document("b", " \t ")], tok, "tok256")
    assert records[0]["language"] == "und" and records[0]["created_ts"] == -1
    assert records[0]["source"] == "fixture" and records[0]["title"] == "Noticia local"
    assert stats["n_documents"] == 2 and stats["single_chunk_documents_pct"] == 50
    assert stats["chunks_per_document_by_collection"]["url_crawls"] == {"mean": 0.5, "p50": 0.5, "max": 1}
    assert stats["tokens_per_chunk"]["max"] == records[0]["n_tokens"]


def test_build_cli_filters_sample_and_keeps_chunker_stats(tok, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    docs = [document("a", "La lluvia será moderada."), document("b", "Texto excluido.")]
    pd.DataFrame(docs).drop(columns="provenance_ids").to_parquet("corpus.parquet")
    pd.DataFrame([dict(document_id="a", provenance_ids="a|old", collection_name="url_crawls",
                       text_sha256="hash")]).to_csv("sample.csv", index=False)
    extra = document("synthetic", "La escuela renueva su biblioteca.")
    extra["language_base"] = "en"
    tmp_path.joinpath("extra.jsonl").write_text(json.dumps(extra) + "\n", encoding="utf-8")
    loaded = load_documents("corpus.parquet", "sample.csv", "extra.jsonl")
    assert [d["document_id"] for d in loaded] == ["a", "synthetic"]
    assert loaded[0]["provenance_ids"] == ["a", "old"]
    for name in ("tok256", "sent256"):
        main(["--input", "corpus.parquet", "--sample", "sample.csv", "--extra", "extra.jsonl",
              "--chunker", name, "--out", f"chunks_{name}.jsonl"])
        summary = json.loads(capsys.readouterr().out)
        assert summary["chunker"] == name and summary["n_documents"] == 2
        rows = [json.loads(line) for line in tmp_path.joinpath(f"chunks_{name}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert {r["document_id"] for r in rows} == {"a", "synthetic"}
        assert rows[0]["created_ts"] == 1704153600
    stats = json.loads(tmp_path.joinpath("reports/chunking/stats.json").read_text(encoding="utf-8"))
    assert set(stats) == {"tok256", "sent256"}


def test_missing_sample_document_is_rejected(tmp_path):
    pd.DataFrame([document("a", "Texto.")]).to_parquet(tmp_path / "corpus.parquet")
    pd.DataFrame([dict(document_id="missing", provenance_ids="missing")]).to_csv(tmp_path / "sample.csv", index=False)
    with pytest.raises(ValueError, match="missing"):
        load_documents(tmp_path / "corpus.parquet", tmp_path / "sample.csv")


@pytest.mark.parametrize("size,overlap", [(0, 0), (8, -1), (8, 8)])
def test_invalid_window_is_rejected(tok, size, overlap):
    with pytest.raises(ValueError):
        chunk_fixed_tokens("Texto", tok, size=size, overlap=overlap)


def test_label_suffix_keeps_segmentation_and_changes_ids(tok, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    docs = [document("a", "el " * 600)]
    plain, _ = build_chunks(docs, tok, "tok256")
    labeled, _ = build_chunks(docs, tok, "tok256", label_suffix="_10k")
    assert [r["text"] for r in labeled] == [r["text"] for r in plain]
    assert all(r["chunker"] == "tok256_10k" for r in labeled)
    assert {r["chunk_id"] for r in labeled}.isdisjoint(r["chunk_id"] for r in plain)
    for row in labeled:
        assert row["chunk_id"] == hashlib.sha256(
            f'a|tok256_10k|{row["start_char"]}|{row["end_char"]}'.encode()).hexdigest()[:32]
    pd.DataFrame(docs).drop(columns="provenance_ids").to_parquet("corpus.parquet")
    pd.DataFrame([dict(document_id="a", provenance_ids="a")]).to_csv("sample.csv", index=False)
    main(["--input", "corpus.parquet", "--sample", "sample.csv", "--chunker", "tok256",
          "--label-suffix", "_10k", "--out", "chunks.jsonl"])
    assert json.loads(capsys.readouterr().out)["chunker"] == "tok256_10k"
    assert [json.loads(line) for line in tmp_path.joinpath("chunks.jsonl").read_text(encoding="utf-8").splitlines()] == labeled
    assert set(json.loads(tmp_path.joinpath("reports/chunking/stats.json").read_text(encoding="utf-8"))) == {"tok256_10k"}



# --- embeddings ---

@pytest.fixture
def embeddings(monkeypatch):
    calls, loads = [], []

    class Model:
        def __init__(self, name, device, revision=None):
            loads.append((name, device, revision))

        def encode(self, texts, normalize_embeddings):
            calls.append((texts, normalize_embeddings))
            return [[0.6, 0.8] if normalize_embeddings else [3.0, 4.0] for _ in texts]

        def get_sentence_embedding_dimension(self):
            return 2

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model))
    module = importlib.import_module("app.embeddings")
    module._model.cache_clear()
    yield module, calls, loads
    module._model.cache_clear()


def test_prefixes_applied(embeddings):
    module, calls, loads = embeddings
    for encoder in ("e5small", "minilm"):
        module.embed(["x"], "query", encoder)
        module.embed(["x"], "passage", encoder)
    assert [texts for texts, _ in calls] == [["query: x"], ["passage: x"], ["x"], ["x"]]
    assert len(loads) == 2 and all(device == "cpu" for _, device, _ in loads)


def test_vectors_normalized(embeddings):
    module, calls, _ = embeddings
    vectors = module.embed(["x", "y"], "passage")
    assert isinstance(vectors, list) and isinstance(vectors[0], list)
    assert all(math.sqrt(sum(x * x for x in vec)) == pytest.approx(1) for vec in vectors)
    assert calls[0][1] is True


@pytest.mark.parametrize("encoder,model,revision", [
    ("e5small", "intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3"),
    ("minilm", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"),
])
def test_fingerprint_matches_loaded_revision(embeddings, monkeypatch, tmp_path, encoder, model, revision):
    module, _, loads = embeddings
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    ref = tmp_path / "hub" / ("models--" + model.replace("/", "--")) / "refs/main"
    ref.parent.mkdir(parents=True)
    ref.write_text("otra-revision\n", encoding="utf-8")
    result = module.fingerprint(encoder)
    assert result["revision"] == revision
    assert loads == [(model, "cpu", revision)]
    assert result["dim"] == 2


def test_invalid_encoder_or_kind_rejected(embeddings):
    module, _, loads = embeddings
    with pytest.raises(ValueError):
        module.embed(["x"], "invalid")
    with pytest.raises(ValueError):
        module.dim("invalid")
    assert loads == []
    assert module.embed([], "query") == []


def test_collection_name():
    from app.config import collection_name
    assert collection_name("e5small", "tok256") == "c_e5small_tok256"



# --- embeddings_probe ---

@pytest.fixture
def probe():
    return importlib.import_module("scripts.embeddings_probe")


@pytest.mark.parametrize("left,right,want", [([1, 2], [1, 2], 1), ([1], [2], 0), ([1, 1, 2], [2, 3], 1 / 3), ([], [], 1)])
def test_jaccard_usa_conjuntos(probe, left, right, want):
    assert probe.jaccard(left, right) == pytest.approx(want)


def test_coseno_identico_y_ortogonal(probe):
    assert probe.cosine([3, 4], [3, 4]) == pytest.approx(1)
    assert probe.cosine([1, 0], [0, 1]) == pytest.approx(0)


def test_resumen_brecha_con_vectores_de_juguete(probe):
    rows = [dict(encoder="e5small", prefijo="query: ", tipo=kind,
                 similitud=probe.cosine([1, 0], vector))
            for kind, vector in [("parafrasis", [1, 0]), ("traduccion", [0, 1]),
                                 ("no_relacionados", [-1, 0])]]
    summary = probe.summarize(rows)
    assert len(summary) == 3
    assert all(row["brecha"] == pytest.approx(1.5) for row in summary)
    assert summary[0]["media"] == summary[0]["min"] == summary[0]["max"] == 1


def test_sin_prefijo_codifica_texto_crudo(probe, monkeypatch):
    class Model:
        def encode(self, texts, normalize_embeddings):
            assert texts == ["hola", "query: literal"]
            assert normalize_embeddings is True
            return [[1, 0], [0, 1]]
    monkeypatch.setattr(probe.embeddings, "_model", lambda encoder: Model())
    assert probe._embed_raw(["hola", "query: literal"], "e5small") == [[1, 0], [0, 1]]


def test_cli_genera_informes_con_fakes(probe, monkeypatch, tmp_path, capsys):
    calls = []
    monkeypatch.setattr(probe.embeddings, "embed", lambda texts, kind, encoder: [[1, 0] for _ in texts])
    monkeypatch.setattr(probe, "_embed_raw", lambda texts, encoder: [[1, 0] for _ in texts])
    monkeypatch.setattr(probe.embeddings, "fingerprint", lambda encoder: {"model": encoder, "dim": 2})
    monkeypatch.setattr(probe, "get_client", lambda: object())

    def search(client, name, qvec, k, filters=None, fingerprint=None):
        calls.append((name, k, fingerprint))
        return [dict(document_id=f"doc{i // 2}", score=1 - i / 100,
                     collection_name="corpus", language="es", title="" if i else "Título @secreto.bsky.social | noticia",
                     text="Texto @oculto.bsky.social " + "x" * 100) for i in range(k)]

    monkeypatch.setattr(probe, "search", search)
    probe.main(["--out", str(tmp_path), "--k", "3"])
    md = (tmp_path / "vecinos.md").read_text(encoding="utf-8")
    for query, language in probe.QUERIES:
        assert query in md
    assert len(probe.QUERIES) == 8
    assert [language for _, language in probe.QUERIES].count("es") == 4
    assert md.count("| 3 |") == 16
    assert "| 4 |" not in md
    assert "e5small" in md and "minilm" in md
    assert "@secreto" not in md and "@oculto" not in md and "@usuario" in md
    assert "\\| noticia" in md
    assert all(k == 10 and fp["model"] in name for name, k, fp in calls)
    data = json.loads((tmp_path / "sensibilidad.json").read_text(encoding="utf-8"))
    assert set(data["fingerprints"]) == {"e5small", "minilm"}
    assert len(data["pares"]) == 18
    assert len(data["similitudes"]) == 54
    assert len(data["resumen"]) == 9
    assert all(row["similitud"] == pytest.approx(1) for row in data["similitudes"])
    assert all(row["brecha"] == pytest.approx(0) for row in data["resumen"])
    with (tmp_path / "solapamiento.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert all(float(row["jaccard_top10"]) == 1 and row["n_e5"] == row["n_minilm"] == "5" for row in rows)
    assert json.loads(capsys.readouterr().out)["consultas"] == 8


def test_k_invalido_no_abre_servicios(probe, monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "get_client", lambda: pytest.fail("No debe abrir servicios"))
    with pytest.raises(SystemExit):
        probe.main(["--out", str(tmp_path), "--k", "0"])



# --- privacy ---

@pytest.mark.parametrize("text,want", [
    ("Hola @ana.bsky.social y @x.com, @local", "Hola @usuario y @usuario, @local"),
    ("https://bsky.app/profile/ana.bsky.social/post/3", "https://bsky.app/profile/usuario/post/3"),
    ("bsky.app/profile/did:plc:abc/post/3", "bsky.app/profile/usuario/post/3"),
    ("@usuario https://example.com/x", "@usuario https://example.com/x"),
    ("correo a@correo.org.", "correo a@correo.org."),
])
def test_mask_handles(text, want):
    from app.privacy import mask_handles
    assert mask_handles(text) == want
    assert mask_handles(want) == want


def test_reexport_y_consumidores():
    from app.privacy import mask_handles
    from scripts import validate_gold, embeddings_probe, eval_retrieval
    assert validate_gold.mask_handles is mask_handles
    text = "@ana.bsky.social https://bsky.app/profile/ana.bsky.social/post/1"
    assert embeddings_probe._cell(text) == mask_handles(text)
    assert eval_retrieval.mask_text(text) == mask_handles(text)


def test_contexto_enmascara_perfiles():
    from app.prompts import build_context
    text = "https://bsky.app/profile/ana.bsky.social/post/1"
    context, _ = build_context([dict(text=text)], 1000)
    assert "ana.bsky.social" not in context


def test_human_eval_enmascara_antes_del_recorte(tmp_path):
    from scripts import human_eval
    prefix = "x" * 389 + " "
    text = prefix + "@ana.bsky.social final"
    record = dict(qid="Q01", run_id="r", run_idx=1, model="m", backend="solr",
                  answer="respuesta", evidence=[dict(n=1, text=text)])
    human_eval.sample([record], {"Q01": dict(type="directa")}, 1, 42, tmp_path)
    with (tmp_path / "hoja_evaluador_A.csv").open(encoding="utf-8-sig") as source:
        item = next(csv.DictReader(source))
    assert item["evidencia"] == "[1] " + (prefix + "@usuario final")[:400]


def test_validate_gold_como_script():
    from pathlib import Path
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "scripts/validate_gold.py", "--help"],
                            cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--gold" in result.stdout
