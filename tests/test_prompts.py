"""Contexto íntegro, trazabilidad y anonimización de publicaciones."""
from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from app import prompts


def chunk(text="Texto", **fields):
    return dict(text=text, source="noticias", collection_name="archivo",
                title="Título", created_ts=1704153600, **fields)


def test_context_respects_budget_with_whole_chunks():
    chunks = [chunk(char * 2500) for char in "abc"]
    original = deepcopy(chunks)
    context, used = prompts.build_context(chunks, 6000)
    assert len(context) <= 6000
    assert len(used) == 2
    assert "a" * 2500 in context and "b" * 2500 in context
    assert "c" * 2500 not in context and "[3]" not in context
    assert chunks == original


def test_citation_numbers_match_used_evidence():
    chunks = [chunk("x" * 700, rank=1), chunk("Segundo", rank=2),
              chunk("Tercero", rank=3)]
    context, used = prompts.build_context(chunks, 250)
    assert [row["n"] for row in used] == [1, 2]
    assert [row["text"] for row in used] == ["Segundo", "Tercero"]
    assert context == (
        "[1] (fuente: noticias | colección: archivo | fecha: 2024-01-02 | título: Título)\nSegundo\n\n"
        "[2] (fuente: noticias | colección: archivo | fecha: 2024-01-02 | título: Título)\nTercero")


def test_context_uses_rank_and_counts_separators():
    chunks = [chunk("B", rank=2), chunk("A", rank=1)]
    context, used = prompts.build_context(chunks, 165)
    assert [row["text"] for row in used] == ["A"]
    assert context.endswith("\nA")
    full, used = prompts.build_context(chunks, 166)
    assert len(full) == 166 and [row["text"] for row in used] == ["A", "B"]


@pytest.mark.parametrize("budget", [0, -1, 1])
def test_context_skips_empty_or_unusable_chunks(budget):
    assert prompts.build_context([{}, {"text": None}, {"text": "  "}, chunk()], budget) == ("", [])


def test_context_formats_unknown_date_and_title():
    context, used = prompts.build_context([
        {"text": "Prueba", "source": "bluesky", "collection_name": "posts", "created_ts": -1},
        {"text": "Otra", "source": "noticias", "collection_name": "archivo", "title": ""},
    ], 6000)
    assert "[1] (fuente: bluesky | colección: posts | fecha: s/f | título: s/t)\nPrueba" in context
    assert "[2] (fuente: noticias | colección: archivo | fecha: s/f | título: s/t)\nOtra" in context
    assert len(used) == 2


def test_prompt_version_has_hash(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    template, version = prompts.load_prompt()
    raw = (Path(__file__).resolve().parents[1] / "prompts/rag_v1.txt").read_bytes()
    assert template == raw.decode("utf-8")
    assert version == "rag_v1@" + hashlib.sha256(raw).hexdigest()[:12]


def test_handles_masked_in_context():
    chunks = [chunk("@juan.bsky.social y @ana.example.org; @local; correo a@correo.org.")]
    context, used = prompts.build_context(chunks, 6000)
    assert "@usuario y @usuario; @local; correo a@correo.org." in context
    assert "@juan.bsky.social" not in context
    assert used[0]["text"] == chunks[0]["text"]


def test_build_prompt_preserves_literal_braces_in_inputs():
    result = prompts.build_prompt("¿{context}?", '{"question": "dato"}',
                                  "Pregunta: {question}\n<documentos>{context}</documentos>")
    assert result == 'Pregunta: ¿{context}?\n<documentos>{"question": "dato"}</documentos>'
