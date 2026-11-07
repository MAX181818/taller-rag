"""Contratos de segmentación, procedencia y construcción sin red."""
import hashlib
import json
from time import perf_counter

import pandas as pd
import pytest
from transformers import AutoTokenizer

from app.chunking import CHUNKERS, Span, chunk_fixed_tokens, chunk_id, chunk_sentences
from scripts.build_chunks import build_chunks, load_documents, main


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
