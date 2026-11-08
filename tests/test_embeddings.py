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
