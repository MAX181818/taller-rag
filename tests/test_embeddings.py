"""Contratos de codificación con modelos falsos y sin descargas."""
import importlib
import math
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def embeddings(monkeypatch):
    calls, loads = [], []

    class Model:
        def __init__(self, name, device):
            loads.append((name, device))

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
    assert len(loads) == 2 and all(device == "cpu" for _, device in loads)


def test_vectors_normalized(embeddings):
    module, calls, _ = embeddings
    vectors = module.embed(["x", "y"], "passage")
    assert isinstance(vectors, list) and isinstance(vectors[0], list)
    assert all(math.sqrt(sum(x * x for x in vec)) == pytest.approx(1) for vec in vectors)
    assert calls[0][1] is True


def test_fingerprint_reads_cached_revision(embeddings, monkeypatch, tmp_path):
    module, _, _ = embeddings
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert module.fingerprint("e5small")["revision"] == "desconocida"
    ref = tmp_path / "hub/models--intfloat--multilingual-e5-small/refs/main"
    ref.parent.mkdir(parents=True)
    ref.write_text("abc123\n", encoding="utf-8")
    assert module.fingerprint("e5small") == dict(
        model="intfloat/multilingual-e5-small", revision="abc123", query_prefix="query: ",
        passage_prefix="passage: ", normalize=True, dim=2)
    assert module.dim("minilm") == 2


def test_invalid_encoder_or_kind_rejected(embeddings):
    module, _, loads = embeddings
    with pytest.raises(ValueError):
        module.embed(["x"], "invalid")
    with pytest.raises(ValueError):
        module.dim("invalid")
    assert loads == []
    assert module.embed([], "query") == []


def test_fingerprint_reads_revision_after_model_load(embeddings, monkeypatch, tmp_path):
    module, _, _ = embeddings
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    original_model = module._model
    ref = tmp_path / "hub/models--intfloat--multilingual-e5-small/refs/main"

    def load_model(encoder):
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text("revision-cargada\n", encoding="utf-8")
        return original_model(encoder)

    monkeypatch.setattr(module, "dim", lambda encoder: load_model(encoder).get_sentence_embedding_dimension())
    first = module.fingerprint("e5small")
    assert first["revision"] == "revision-cargada"
    assert module.fingerprint("e5small") == first


def test_collection_name():
    from app.config import collection_name
    assert collection_name("e5small", "tok256") == "c_e5small_tok256"
