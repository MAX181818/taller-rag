"""Vectores normalizados y huellas de los codificadores en CPU."""
from functools import lru_cache
from typing import Literal

from app import config


ENCODERS = {
    "e5small": ("intfloat/multilingual-e5-small", "query: ", "passage: "),
    "minilm": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "", ""),
}

REVISIONS = {
    "e5small": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
    "minilm": "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
}


class EncoderMismatch(Exception):
    """La colección pertenece a un codificador con otra huella."""


def _encoder(encoder):
    try:
        return ENCODERS[encoder]
    except KeyError:
        raise ValueError(f"Codificador no soportado: {encoder}") from None


@lru_cache(maxsize=None)
def _model(encoder):
    name, _, _ = _encoder(encoder)
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, device="cpu", revision=REVISIONS[encoder])


def embed(texts: list[str], kind: Literal["query", "passage"],
          encoder: str = config.ENCODER) -> list[list[float]]:
    if kind not in ("query", "passage"):
        raise ValueError(f"Tipo de texto no soportado: {kind}")
    _, query_prefix, passage_prefix = _encoder(encoder)
    if not texts:
        return []
    prefix = query_prefix if kind == "query" else passage_prefix
    vectors = _model(encoder).encode([prefix + text for text in texts], normalize_embeddings=True)
    return [[float(value) for value in vector] for vector in vectors]


def dim(encoder) -> int:
    return int(_model(encoder).get_sentence_embedding_dimension())


def fingerprint(encoder) -> dict:
    model, query_prefix, passage_prefix = _encoder(encoder)
    dimension = dim(encoder)
    revision = REVISIONS[encoder]
    return dict(model=model, revision=revision, query_prefix=query_prefix,
                passage_prefix=passage_prefix, normalize=True, dim=dimension)
